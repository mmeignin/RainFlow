import os
import torch


def save_checkpoint(path, step, config, net_model, ema_model, optim, sched, null_cond, best_val=None):
    torch.save(
        dict(
            step=step,
            config=config,
            net_model=net_model.state_dict(),
            ema_model=ema_model.state_dict() if ema_model is not None else None,
            optim=optim.state_dict(),
            sched=sched.state_dict(),
            null_cond=null_cond.data,
            best_val=best_val,
        ),
        path,
    )


def load_checkpoint(path, net_model, ema_model, optim, sched, null_cond, device):
    """Restore training state in place. Returns (step, best_val)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net_model.load_state_dict(ckpt["net_model"])
    if ema_model is not None and ckpt.get("ema_model") is not None:
        ema_model.load_state_dict(ckpt["ema_model"])
    optim.load_state_dict(ckpt["optim"])
    sched.load_state_dict(ckpt["sched"])
    if null_cond is not None and "null_cond" in ckpt:
        null_cond.data.copy_(ckpt["null_cond"])
    return ckpt["step"], ckpt.get("best_val", float("inf"))


def read_best_val(path, device):
    """Read only the `best_val` field from a checkpoint, without touching any model state.

    Used at resume time to recover the true best-so-far from `best.pt` — the one
    file whose `best_val` is guaranteed to describe its own weights — rather than
    from `latest.pt`, which no longer carries a `best_val` of its own.
    """
    if not os.path.exists(path):
        return float("inf")
    return torch.load(path, map_location=device, weights_only=False).get("best_val", float("inf"))


def save_best(ckptdir, val_loss, best_val, **kwargs):
    """Write best.pt when `val_loss` improves. Returns the (possibly updated) best."""
    if not (val_loss < best_val):        # False for NaN — the point of the inversion
        return best_val
    save_checkpoint(os.path.join(ckptdir, "best.pt"), best_val=val_loss, **kwargs)
    return val_loss
