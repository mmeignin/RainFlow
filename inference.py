import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfm.cfm import sample_prior
from models.build import build_unet
from utils.config import INFERENCE_KEYS, inherited_from_checkpoint, load_inference_config
from utils.dataset import RainIRDataset
from utils.plotting import save_preview
from utils.sampling import sample
from utils.utils import RESIDUAL_DET_KEY, build_pipeline_from_config, set_seed


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

def detect_n_cond(state_dict, residual=False):
    """Read n_cond from the first conv weight shape: total_in_ch minus the rain
    channel and, in residual mode, the extra x_det channel."""
    key = next((k for k in state_dict if "input_blocks.0.0.weight" in k), None)
    if key is None:
        raise ValueError(
            "checkpoint's net_model has no 'input_blocks.0.0.weight' key — "
            "unrecognized architecture or corrupt checkpoint."
        )
    return state_dict[key].shape[1] - 1 - int(residual)


def load_model(ckpt, n_cond, num_channel, image_size, channel_mult, attention_resolutions,
               num_heads, num_head_channels, use_scale_shift_norm, dropout,
               use_ema, device, residual=False):
    """Build the UNet and load weights from an already-read checkpoint dict."""
    model = build_unet(num_channel, n_cond, image_size, channel_mult, attention_resolutions,
                       num_heads, num_head_channels, device,
                       use_scale_shift_norm, dropout, residual=residual)

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
    residual  = full_cfg.get("residual", False)
    prior     = train_cfg["prior"]
    # Required only for the student_t prior, so not an inherited key.
    student_t_nu = full_cfg.get("student_t_nu")
    if prior == "student_t" and student_t_nu is None:
        raise ValueError(
            f"{checkpoint}: prior is 'student_t' but 'student_t_nu' is absent. The "
            f"degrees of freedom must be recorded to reproduce the source distribution."
        )
    n_cond    = detect_n_cond(ckpt["net_model"], residual=residual)
    model, null_cond = load_model(
        ckpt, n_cond, train_cfg["num_channel"], train_cfg["image_size"],
        train_cfg["channel_mult"], train_cfg["attention_resolutions"],
        train_cfg["num_heads"], train_cfg["num_head_channels"],
        train_cfg["use_scale_shift_norm"], train_cfg["dropout"],
        cfg["use_ema"], device, residual=residual,
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
    pipeline_cfg = {**train_cfg, "residual": residual}
    pipeline = build_pipeline_from_config(pipeline_cfg, channel_list)
    print(f"Pipeline: {pipeline}  (from checkpoint config)")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = RainIRDataset.from_csv(
        cond_keys      = channel_list,
        csv_file       = cfg["data_csv"],
        pipeline       = pipeline,
        debug          = cfg["debug"],
        augment        = False,
        det_key        = RESIDUAL_DET_KEY if residual else None,
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
            batch = [t.to(device) for t in batch]
            if residual:
                target, cond, x_det = batch      # target is the true residual r
            else:
                target, cond = batch             # target is x1 in latent space
            B = cond.shape[0]

            if residual:
                # CFG's unconditional branch nulls `cond` but keeps `x_det`; null_res is
                # pre-expanded to the batch.
                cond_res = torch.cat([cond, x_det], dim=1)
                null_res = None
                if null_cond is not None:
                    null_full = null_cond.expand(B, -1, cond.shape[2], cond.shape[3])
                    null_res = torch.cat([null_full, x_det], dim=1)

            for run_idx, run_dir in enumerate(run_dirs):
                # Distinct seed per (run, sample). Not utils.set_seed: numpy rejects seeds >= 2**32.
                torch.manual_seed(cfg["seed"] + run_idx * 100_000_000 + global_idx)
                x0               = sample_prior(target.shape, prior, student_t_nu, device=target.device)
                if residual:
                    pred_latent, nfe = sample(model, x0, cond_res, null_res, cfg)
                    pred_rain, cond_raw = pipeline.post(
                        x=pipeline.from_residual(pred_latent, x_det), cond=cond)
                    x1_rain, _ = pipeline.post(
                        x=pipeline.from_residual(target, x_det), cond=cond)
                else:
                    pred_latent, nfe = sample(model, x0, cond, null_cond, cfg)
                    pred_rain, cond_raw = pipeline.post(x=pred_latent, cond=cond)
                    x1_rain, _          = pipeline.post(x=target, cond=cond)
                total_nfe       += nfe

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
    # nfe counts model calls per batch, so average over (batch, run) pairs.
    avg_nfe = total_nfe / max(len(loader) * cfg["n_runs"], 1)
    print(f"\nDone.  {n_samples} samples × {cfg['n_runs']} runs  →  {output_dir}")
    print(f"Avg NFE per solve (one batch, one run): {avg_nfe:.1f}")


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
