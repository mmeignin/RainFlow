import argparse
import copy
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import trange

from cfm.cfm import ConditionalFlowMatcher
from models.build import build_unet
from utils.checkpoint import load_checkpoint, read_best_val, save_best, save_checkpoint
from utils.config import load_config
from utils.dataset import RainIRDataset
from utils.logger import CSVLogger
from utils.plotting import save_preview
from utils.sampling import euler_sample
from utils.schedule import build_schedule_fn
from utils.utils import build_pipeline_from_config, ema, infiniteloop, set_seed

# Fewer steps than inference
PREVIEW_STEPS = 25


# ──────────────────────────────────────────────────────────────────────────────
# Sample generation for training visualisations
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def gen_and_save(x1, cond, model, pipeline, savedir, step, tag, channel_list):
    model.eval()
    # Unguided: null_cond and guidance_scale are left at their defaults.
    pred, _ = euler_sample(model, torch.randn_like(x1), cond, n_steps=PREVIEW_STEPS)
    model.train()

    pred_rain, cond_raw = pipeline.post(x=pred, cond=cond)
    x1_rain, _ = pipeline.post(x=x1, cond=cond)
    save_preview(x1_rain, pred_rain, cond_raw,
                 os.path.join(savedir, f"{tag}_step_{step:07d}.png"),
                 channel_list=channel_list)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, flow_matcher, device, seed):
    """Mean CFM loss over the full val split.

    The RNG is pinned and restored around the loop, so the noise `x0` and the
    times `t` are identical at every evaluation. Without that the val loss would
    carry its own sampling noise and be uncomparable across steps.
    """
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    model.eval()
    total, batches = 0.0, 0
    for x1, cond in loader:
        x1, cond = x1.to(device), cond.to(device)
        x0 = torch.randn_like(x1)
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
        total += ((model(t, xt, cond) - ut) ** 2).mean().item()
        batches += 1
    model.train()

    torch.set_rng_state(rng_state)
    return total / max(batches, 1)


def build_val_loader(cfg, channel_list):
    """The full val split, unshuffled and unaugmented.

    A subset would be cheaper, but `best.pt` is selected on this number: an
    N-batch subset carries sqrt(total/N) times the standard error, and the first
    N rows of a timestamp-sorted manifest are one season, not the val distribution.
    """
    pipeline = build_pipeline_from_config({**cfg, "random_flip": False}, channel_list)
    dataset = RainIRDataset.from_csv(
        cond_keys=channel_list, csv_file=cfg["val_csv"],
        pipeline=pipeline, debug=cfg["debug"],
    )
    return DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], drop_last=False, pin_memory=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    savedir = os.path.join(cfg["output_dir"], cfg["exp_name"])
    ckptdir = os.path.join(savedir, "ckpt")
    figdir_net = os.path.join(savedir, "figures", "net")
    figdir_ema = os.path.join(savedir, "figures", "ema")
    for d in (ckptdir, figdir_net, figdir_ema):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(savedir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    channel_list = cfg["channels"].split(",")
    n_cond = len(channel_list)

    pipeline = build_pipeline_from_config(cfg, channel_list)
    if not (pipeline.x or pipeline.cond):
        print("WARNING: no value transforms configured — training on raw physical units.", flush=True)

    dataset = RainIRDataset.from_csv(
        cond_keys=channel_list, csv_file=cfg["data_csv"],
        pipeline=pipeline, debug=cfg["debug"],
    )
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        drop_last=True, shuffle=True, pin_memory=True,
    )
    datalooper = infiniteloop(loader)
    val_loader = build_val_loader(cfg, channel_list)

    print(f"Pipeline: {pipeline}")
    print(f"Train: {len(dataset)} samples | Val: {len(val_loader.dataset)} samples "
          f"({len(val_loader)} batches) | n_cond: {n_cond} | device: {device}")
    print(f"Run: {cfg['total_steps']} steps | eval every {cfg['eval_every']} | "
          f"save every {cfg['save_every']} | log every {cfg['log_every']}")

    # ── model ─────────────────────────────────────────────────────────────────
    net_model = build_unet(
        cfg["num_channel"], n_cond, cfg["image_size"],
        cfg["channel_mult"], cfg["attention_resolutions"], device,
    )
    print(f"Model params: {sum(p.numel() for p in net_model.parameters())/1e6:.2f}M")

    # Start maintaining the EMA weights at the halfway point of training.
    ema_start_step = cfg["total_steps"] // 2
    ema_model = copy.deepcopy(net_model)
    ema_active = False

    null_cond = torch.nn.Parameter(torch.zeros(1, n_cond, 1, 1, device=device))
    optim = torch.optim.AdamW(
        list(net_model.parameters()) + [null_cond], lr=cfg["lr"], weight_decay=1e-4,
    )
    schedule_fn = build_schedule_fn(cfg["scheduler"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: schedule_fn(s, cfg["warmup_steps"], cfg["total_steps"])
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_step, best_val = 0, float("inf")
    resume_path = os.path.join(ckptdir, "latest.pt")
    if os.path.exists(resume_path):
        start_step, _ = load_checkpoint(
            resume_path, net_model, ema_model, optim, sched, null_cond, device
        )
        start_step += 1
        best_val = read_best_val(os.path.join(ckptdir, "best.pt"), device)
        ema_active = start_step >= ema_start_step
        print(f"Resuming at step {start_step} (best val {best_val:.4f})")

        set_seed(cfg["seed"] + start_step)
        for _ in range(start_step % len(loader)):
            next(datalooper)

    flow_matcher = ConditionalFlowMatcher(sigma=0.0)
    val_seed = cfg["seed"] + 1  # fixed, so every evaluation sees the same x0 and t

    def checkpoint_args(step):
        return dict(step=step, config=cfg, net_model=net_model,
                    ema_model=ema_model if ema_active else None,
                    optim=optim, sched=sched, null_cond=null_cond)

    # ── training loop ─────────────────────────────────────────────────────────
    metrics = CSVLogger(os.path.join(savedir, "metrics.csv"),
                        ["step", "train_loss", "val_loss", "lr"])
    with metrics, trange(start_step, cfg["total_steps"], dynamic_ncols=True,
                         initial=start_step, total=cfg["total_steps"]) as pbar:
        running, running_n = 0.0, 0

        for step in pbar:
            net_model.train()
            optim.zero_grad()

            x1, cond = [t.to(device) for t in next(datalooper)]
            cond_viz = cond

            if cfg["cfg_drop"] > 0:
                mask = torch.rand(cond.shape[0], 1, 1, 1, device=cond.device) < cfg["cfg_drop"]
                cond = torch.where(mask, null_cond.expand_as(cond), cond)

            x0 = torch.randn_like(x1)
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
            loss = ((net_model(t, xt, cond) - ut) ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), cfg["grad_clip"])
            optim.step()
            sched.step()

            if step == ema_start_step:
                ema_model = copy.deepcopy(net_model)
                ema_active = True
                print(f"\n[step {step}] EMA initialised.")
            if ema_active:
                ema(net_model, ema_model, cfg["ema_decay"])

            running += loss.item()
            running_n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{sched.get_last_lr()[0]:.2e}",
                             ema="on" if ema_active else "off")

            # ── validation ────────────────────────────────────────────────────
            val_loss = None
            if cfg["eval_every"] > 0 and step % cfg["eval_every"] == 0 and step > 0:
                val_loss = evaluate(net_model, val_loader, flow_matcher, device, val_seed)
                best_val = save_best(ckptdir, val_loss, best_val, **checkpoint_args(step))
                pbar.write(f"[step {step}] val {val_loss:.4f} (best {best_val:.4f})")

            # ── logging ───────────────────────────────────────────────────────
            if val_loss is not None or (cfg["log_every"] > 0 and step % cfg["log_every"] == 0):
                metrics.log(step=step,
                            train_loss=f"{running / max(running_n, 1):.6f}",
                            val_loss=f"{val_loss:.6f}" if val_loss is not None else "",
                            lr=f"{sched.get_last_lr()[0]:.6e}")
                running, running_n = 0.0, 0

            # ── checkpointing ─────────────────────────────────────────────────
            if cfg["save_every"] > 0 and step % cfg["save_every"] == 0 and step > 0:
                save_checkpoint(os.path.join(ckptdir, f"step_{step:07d}.pt"), **checkpoint_args(step))
                save_checkpoint(resume_path, **checkpoint_args(step))

                gen_and_save(x1, cond_viz, net_model, pipeline, figdir_net, step, "net",
                             channel_list)
                if ema_active:
                    gen_and_save(x1, cond_viz, ema_model, pipeline, figdir_ema, step, "ema",
                                 channel_list)

    save_checkpoint(os.path.join(ckptdir, "final.pt"), **checkpoint_args(cfg["total_steps"]))
    print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
