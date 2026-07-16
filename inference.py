import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.build import build_unet
from utils.config import INFERENCE_KEYS, inherited_from_checkpoint, load_inference_config
from utils.dataset import RainIRDataset
from utils.plotting import save_preview
from utils.sampling import sample
from utils.utils import build_pipeline_from_config, set_seed


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

# argparse types for the keys that aren't plain strings.
_FLAG_TYPES = {
    "use_ema": bool, "save_npy": bool, "save_png": bool, "debug": bool,
    "n_steps": int, "n_runs": int, "batch_size": int, "num_workers": int,
    "seed": int, "start_idx": int, "end_idx": int,
    "rtol": float, "atol": float, "guidance_scale": float,
}


def parse_args():
    p = argparse.ArgumentParser(description="Flow Matching inference")
    p.add_argument("--config", required=True, help="inference YAML config")
    for key in INFERENCE_KEYS:
        kind = _FLAG_TYPES.get(key, str)
        if kind is bool:
            p.add_argument(f"--{key}", default=None, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(f"--{key}", default=None, type=kind)

    args = vars(p.parse_args())
    return load_inference_config(args.pop("config"), overrides=args)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_n_cond(state_dict):
    """Read n_cond from the first conv weight shape: total_in_ch - 1 (rain channel)."""
    key = next((k for k in state_dict if "input_blocks.0.0.weight" in k), None)
    if key is None:
        raise ValueError(
            "checkpoint's net_model has no 'input_blocks.0.0.weight' key — "
            "unrecognized architecture or corrupt checkpoint."
        )
    return state_dict[key].shape[1] - 1


def load_model(ckpt, n_cond, num_channel, image_size, channel_mult, attention_resolutions, use_ema, device):
    """Build the UNet and load weights from an already-read checkpoint dict."""
    model = build_unet(num_channel, n_cond, image_size, channel_mult, attention_resolutions, device)

    null_cond = ckpt["null_cond"].to(device) if "null_cond" in ckpt else None

    if use_ema and ckpt.get("ema_model") is not None:
        model.load_state_dict(ckpt["ema_model"])
        print("Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt["net_model"])
        print("Loaded net weights.")

    print(f"Checkpoint step: {ckpt.get('step', 'unknown')}  |  n_cond: {n_cond}"
          + ("" if null_cond is not None else "  (no null_cond — CFG disabled)"))
    model.eval()
    return model, null_cond


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    if cfg["exp_dir"] is None:
        raise ValueError("inference config must set 'exp_dir'.")
    checkpoint = os.path.join(cfg["exp_dir"], "ckpt", f"{cfg['ckpt_name']}.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    tag        = f"{cfg['ckpt_name']}_gs{cfg['guidance_scale']}_runs{cfg['n_runs']}_{cfg['solver']}"
    output_dir = os.path.join(cfg["exp_dir"], "inference", tag)
    os.makedirs(output_dir, exist_ok=True)

    # ── load model + inherit the training config from the checkpoint ─────────
    ckpt      = torch.load(checkpoint, map_location=device, weights_only=False)
    full_cfg  = ckpt.get("config", {})
    train_cfg = inherited_from_checkpoint(full_cfg, checkpoint)
    n_cond    = detect_n_cond(ckpt["net_model"])
    model, null_cond = load_model(
        ckpt, n_cond, train_cfg["num_channel"], train_cfg["image_size"],
        train_cfg["channel_mult"], train_cfg["attention_resolutions"], cfg["use_ema"], device,
    )
    if cfg["guidance_scale"] != 1.0 and full_cfg.get("cfg_drop", 0.0) == 0:
        print(
            "WARNING: guidance_scale != 1.0 but this checkpoint was trained with cfg_drop=0, "
            "so null_cond never received gradient and remains at its zero-init. Classifier-free "
            "guidance will blend against an untrained null condition -- results may be degraded "
            "or meaningless. Retrain with cfg_drop > 0 to use guidance."
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M  |  device: {device}  |  solver: {cfg['solver']}")

    channels = train_cfg["channels"]
    channel_list = channels.split(",")
    if n_cond != len(channel_list):
        raise ValueError(
            f"Checkpoint n_cond={n_cond} does not match its config's {len(channel_list)} "
            f"channels — corrupt or hand-edited checkpoint."
        )
    print(f"n_cond: {n_cond}  |  channels: {len(channel_list)}")

    # ── transform pipeline, inherited from the checkpoint ────────────────────
    pipeline = build_pipeline_from_config({**train_cfg, "random_flip": False}, channel_list)
    print(f"Pipeline: {pipeline}  (from checkpoint config)")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = RainIRDataset.from_csv(
        cond_keys      = channel_list,
        csv_file       = cfg["data_csv"],
        pipeline       = pipeline,
        debug          = cfg["debug"],
    )

    start_idx = cfg["start_idx"]
    end_idx   = cfg["end_idx"] if cfg["end_idx"] >= 0 else len(dataset)
    if start_idx > 0 or end_idx < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(start_idx, end_idx))
        print(f"Slice [{start_idx}, {end_idx})  →  {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], drop_last=False)
    print(f"Dataset: {len(dataset)} samples  |  Runs: {cfg['n_runs']}")

    # ── per-run output dirs ───────────────────────────────────────────────────
    run_dirs = []
    for r in range(cfg["n_runs"]):
        d = os.path.join(output_dir, f"run_{r:02d}")
        os.makedirs(d, exist_ok=True)
        run_dirs.append(d)

    # Full reproducibility: the inference cfg, the resolved checkpoint, and the
    # complete training config embedded in it.
    with open(os.path.join(output_dir, "inference_config.json"), "w") as f:
        json.dump({**cfg, "checkpoint": checkpoint, "n_cond": n_cond,
                   "train_config": full_cfg}, f, indent=2)

    # ── inference loop ────────────────────────────────────────────────────────
    total_nfe  = 0
    global_idx = start_idx
    preview_done = set()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            x1, cond = batch
            x1   = x1.to(device)
            cond = cond.to(device)
            B    = x1.shape[0]

            for run_idx, run_dir in enumerate(run_dirs):
                # Stride exceeds any realistic slice size, so run_idx never collides with
                # global_idx. torch.manual_seed takes up to 2**64-1; do NOT swap in
                # utils.set_seed, whose np.random.seed rejects anything >= 2**32.
                torch.manual_seed(cfg["seed"] + run_idx * 100_000_000 + global_idx)
                x0               = torch.randn_like(x1)
                pred_latent, nfe = sample(model, x0, cond, null_cond, cfg)
                total_nfe       += nfe

                pred_rain, cond_raw = pipeline.post(x=pred_latent, cond=cond)
                x1_rain, _          = pipeline.post(x=x1,          cond=cond)

                if cfg["save_npy"]:
                    for i in range(B):
                        np.save(
                            os.path.join(run_dir, f"sample_{global_idx + i:04d}.npy"),
                            pred_rain[i].cpu().numpy(),
                        )

                if cfg["save_png"] and run_idx not in preview_done:
                    save_preview(
                        x1_rain, pred_rain, cond_raw,
                        os.path.join(run_dir, "preview.png"),
                        channel_list=channel_list,
                    )
                    preview_done.add(run_idx)

            global_idx += B

    n_samples = global_idx - start_idx
    # nfe counts model calls, and one call evaluates a whole batch at once — so
    # this averages over (batch, run) pairs, not (sample, run) pairs, to avoid
    # diluting the cost by batch_size.
    avg_nfe = total_nfe / max(len(loader) * cfg["n_runs"], 1)
    print(f"\nDone.  {n_samples} samples × {cfg['n_runs']} runs  →  {output_dir}")
    print(f"Avg NFE per solve (one batch, one run): {avg_nfe:.1f}")


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
