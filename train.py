import argparse
import copy
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import trange

from cfm.cfm import ConditionalFlowMatcher, sample_prior, sample_timesteps
from models.build import build_unet
from utils.checkpoint import load_checkpoint, read_best_val, save_best, save_checkpoint
from utils.config import load_config
from utils.dataset import RainIRDataset
from utils.logger import CSVLogger
from utils.plotting import save_preview, save_preview_residual
from utils.sampling import euler_sample
from utils.schedule import build_schedule_fn
from utils.utils import RESIDUAL_DET_KEY, build_pipeline_from_config, ema, infiniteloop, set_seed

# Fewer steps than inference
PREVIEW_STEPS = 25


# ──────────────────────────────────────────────────────────────────────────────
# Sample generation for training visualisations
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def gen_and_save(target, cond, x_det, model, pipeline, savedir, step, tag, channel_list, sampler):
    """Sample from the prior and save a rain preview (unguided).

    `x_det` is the residual baseline, None in non-residual mode.
    """
    model.eval()
    residual = x_det is not None
    model_cond = torch.cat([cond, x_det], dim=1) if residual else cond
    x0 = sample_prior(target.shape, sampler["prior"], sampler["student_t_nu"], device=target.device)
    pred, _ = euler_sample(model, x0, model_cond, n_steps=PREVIEW_STEPS)
    model.train()

    path = os.path.join(savedir, f"{tag}_step_{step:07d}.png")
    if residual:
        pred_rain, cond_raw = pipeline.post(x=pipeline.from_residual(pred, x_det), cond=cond)
        x1_rain, _ = pipeline.post(x=pipeline.from_residual(target, x_det), cond=cond)
        x_det_rain, _ = pipeline.post(x=x_det)
        save_preview_residual(cond_raw, x1_rain, x_det_rain, pred_rain, pred, target,
                              path, channel_list=channel_list)
    else:
        pred_rain, cond_raw = pipeline.post(x=pred, cond=cond)
        x1_rain, _ = pipeline.post(x=target, cond=cond)
        save_preview(x1_rain, pred_rain, cond_raw, path, channel_list=channel_list)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, flow_matcher, device, seed, sampler, residual=False):
    """Mean CFM loss over the full val split, for the weights `model` holds.

    x0 and t are drawn as in training. The RNG is pinned to `seed` and restored
    afterwards, so every evaluation sees the same noise and times.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)

    was_training = model.training
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        batch = [t.to(device) for t in batch]
        if residual:
            target, cond, x_det = batch
            model_cond = torch.cat([cond, x_det], dim=1)
        else:
            target, cond = batch
            model_cond = cond
        x0 = sample_prior(target.shape, sampler["prior"], sampler["student_t_nu"], device=device)
        t = sample_timesteps(target.shape[0], sampler["time_sampling"],
                             sampler["time_sampling_std"], sampler["time_sampling_k"],
                             sampler["time_sampling_mix_prob"], device=device)
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, target, t=t)
        total += ((model(t, xt, model_cond) - ut) ** 2).mean().item()
        batches += 1
    model.train(was_training)

    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return total / max(batches, 1)


def build_loader(cfg, channel_list, pipeline, csv_file, train):
    """Train loader (shuffled, augmented) or val loader (full split, unshuffled)."""
    det_key = RESIDUAL_DET_KEY if cfg.get("residual", False) else None
    dataset = RainIRDataset.from_csv(
        cond_keys=channel_list, csv_file=csv_file,
        pipeline=pipeline, debug=cfg["debug"], augment=train, det_key=det_key,
    )
    return DataLoader(
        dataset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        shuffle=train, drop_last=train, pin_memory=True,
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
    residual = cfg.get("residual", False)

    # Optional keys stay None when absent; the samplers then use their defaults.
    sampler = {
        "prior": cfg["prior"],
        "student_t_nu": cfg.get("student_t_nu"),
        "time_sampling": cfg["time_sampling"],
        "time_sampling_std": cfg.get("time_sampling_std"),
        "time_sampling_k": cfg.get("time_sampling_k"),
        "time_sampling_mix_prob": cfg.get("time_sampling_mix_prob"),
    }

    pipeline = build_pipeline_from_config(cfg, channel_list)
    if not (pipeline.x or pipeline.cond):
        print("WARNING: no value transforms configured — training on raw physical units.", flush=True)

    loader = build_loader(cfg, channel_list, pipeline, cfg["data_csv"], train=True)
    val_loader = build_loader(cfg, channel_list, pipeline, cfg["val_csv"], train=False)
    datalooper = infiniteloop(loader)

    print(f"Pipeline: {pipeline}")
    print(f"Train: {len(loader.dataset)} samples | Val: {len(val_loader.dataset)} samples "
          f"({len(val_loader)} batches) | n_cond: {n_cond} | device: {device}")
    print(f"Run: {cfg['total_steps']} steps | eval every {cfg['eval_every']} | "
          f"save every {cfg['save_every']} | log every {cfg['log_every']}")

    # ── model ─────────────────────────────────────────────────────────────────
    net_model = build_unet(
        cfg["num_channel"], n_cond, cfg["image_size"],
        cfg["channel_mult"], cfg["attention_resolutions"],
        cfg["num_heads"], cfg["num_head_channels"], device,
        cfg["use_scale_shift_norm"], cfg["dropout"],
        residual=residual,
    )
    print(f"Model params: {sum(p.numel() for p in net_model.parameters())/1e6:.2f}M")

    # The EMA starts halfway through training.
    ema_start_step = cfg["total_steps"] // 2
    ema_model = copy.deepcopy(net_model)
    ema_active = False

    null_cond = torch.nn.Parameter(torch.zeros(1, n_cond, 1, 1, device=device))
    optim = torch.optim.AdamW(
        list(net_model.parameters()) + [null_cond],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
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

        # Reseed from the resume step; the batch order is not replayed.
        set_seed(cfg["seed"] + start_step)

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

            batch = [t.to(device) for t in next(datalooper)]
            if residual:
                target, cond, x_det = batch
            else:
                target, cond, x_det = *batch, None
            cond_viz = cond

            # CFG dropout masks only the IR channels; x_det (when present) stays live.
            if cfg["cfg_drop"] > 0:
                mask = torch.rand(cond.shape[0], 1, 1, 1, device=cond.device) < cfg["cfg_drop"]
                cond = torch.where(mask, null_cond.expand_as(cond), cond)

            model_cond = torch.cat([cond, x_det], dim=1) if residual else cond
            x0 = sample_prior(target.shape, sampler["prior"], sampler["student_t_nu"], device=device)
            t = sample_timesteps(target.shape[0], sampler["time_sampling"],
                                 sampler["time_sampling_std"], sampler["time_sampling_k"],
                                 sampler["time_sampling_mix_prob"], device=device)
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, target, t=t)
            loss = ((net_model(t, xt, model_cond) - ut) ** 2).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"step {step}: non-finite loss ({loss.item()}). "
                    f"target nan={torch.isnan(target).float().mean():.4f} "
                    f"cond nan={torch.isnan(cond).float().mean():.4f} "
                    f"xt nan={torch.isnan(xt).float().mean():.4f}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), cfg["grad_clip"])
            optim.step()
            sched.step()

            if step == ema_start_step:
                ema_model = copy.deepcopy(net_model)
                ema_active = True
                print(f"\n[step {step}] EMA initialised — val_loss now scores the EMA "
                      f"weights (not the net), and best.pt selection starts here.")
            if ema_active:
                ema(net_model, ema_model, cfg["ema_decay"])

            running += loss.item()
            running_n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{sched.get_last_lr()[0]:.2e}",
                             ema="on" if ema_active else "off")

            # ── validation ────────────────────────────────────────────────────
            # best.pt is selected on the EMA weights, which inference loads. Before the EMA
            # starts the net is scored for the loss curve only and best.pt is not written,
            # so val_loss is not continuous across ema_start_step.
            val_loss = None
            if cfg["eval_every"] > 0 and step % cfg["eval_every"] == 0 and step > 0:
                val_loss = evaluate(ema_model if ema_active else net_model,
                                    val_loader, flow_matcher, device, val_seed, sampler, residual)
                if ema_active:
                    best_val = save_best(ckptdir, val_loss, best_val, **checkpoint_args(step))
                    pbar.write(f"[step {step}] val[ema] {val_loss:.4f} (best {best_val:.4f})")
                else:
                    pbar.write(f"[step {step}] val[net] {val_loss:.4f} "
                               f"(best.pt not written until EMA starts at {ema_start_step})")

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

                gen_and_save(target, cond_viz, x_det, net_model, pipeline, figdir_net, step,
                             "net", channel_list, sampler)
                if ema_active:
                    gen_and_save(target, cond_viz, x_det, ema_model, pipeline, figdir_ema, step,
                                 "ema", channel_list, sampler)

    save_checkpoint(os.path.join(ckptdir, "final.pt"), **checkpoint_args(cfg["total_steps"]))
    print(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
