"""Config loading. The YAML is the single source of truth.

Inference splits its settings across two sources:

  - Keys describing the *model and its transform* (`INHERITED_KEYS`) come from
    the config embedded in the checkpoint. They are the only ones guaranteed to
    match the weights being loaded. Reading them from a YAML path instead would
    silently denormalise with the wrong statistics whenever that file drifted.
  - Everything else (`INFERENCE_KEYS`) is an inference concern — solver, output,
    batch size — and comes from the inference config, overridable from the CLI.
"""

import os

import yaml

TRAIN_KEYS = (
    # data
    "data_csv", "val_csv", "norm_csv", "channels",
    # transforms
    "random_flip", "flip_prob", "x_transform", "cond_transform",
    # model
    "num_channel", "image_size", "channel_mult", "attention_resolutions",
    # optimisation
    "total_steps", "batch_size", "num_workers", "lr", "grad_clip",
    "warmup_steps", "scheduler", "ema_decay", "cfg_drop", "seed",
    # evaluation and checkpointing (all cadences in steps)
    "eval_every", "log_every", "save_every",
    # output
    "output_dir", "exp_name", "debug",
)

# Taken from the checkpoint's embedded training config, these must describe the loaded weights exactly.
INHERITED_KEYS = (
    "channels", "norm_csv", "x_transform", "cond_transform",
    "num_channel", "image_size", "channel_mult", "attention_resolutions",
)

INFERENCE_KEYS = (
    # checkpoint: load <exp_dir>/ckpt/<ckpt_name>.pt, write <exp_dir>/inference/<tag>/
    "exp_dir", "ckpt_name", "use_ema",
    # data
    "data_csv", "start_idx", "end_idx",
    # solver
    "solver", "n_steps", "rtol", "atol",
    # classifier-free guidance / ensembling
    "guidance_scale", "n_runs",
    # output
    "save_npy", "save_png",
    # misc
    "batch_size", "num_workers", "seed", "debug",
)


def _load_yaml(path: str) -> dict:
    if not path:
        raise ValueError("a --config path is required; there are no in-code defaults.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a top-level mapping, got {type(config).__name__}.")
    return config


def _validate(config: dict, required, path: str, consumer: str) -> dict:
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    unknown = sorted(set(config) - set(required))
    if unknown:
        print(f"note: {path} has keys not read by {consumer}: {unknown}", flush=True)
    return config


def load_config(path: str) -> dict:
    """Load and validate a training config."""
    config = _validate(_load_yaml(path), TRAIN_KEYS, path, "training")

    eval_every, log_every = config["eval_every"], config["log_every"]
    if eval_every > 0 and log_every > 0 and eval_every % log_every != 0:
        raise ValueError(
            f"{path}: eval_every ({eval_every}) must be a multiple of log_every "
            f"({log_every}) — otherwise an eval-triggered metrics row lands off the "
            f"log_every grid and silently shortens the train_loss averaging window "
            f"for it and the row after it."
        )
    return config


def load_inference_config(path: str, overrides: dict = None) -> dict:
    """Load an inference config, then apply non-None CLI overrides on top."""
    config = _validate(_load_yaml(path), INFERENCE_KEYS, path, "inference")
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return config


def inherited_from_checkpoint(train_cfg: dict, checkpoint: str) -> dict:
    """Pull the model/transform keys out of a checkpoint's embedded training config.

    Fails loudly on a checkpoint written before these keys existed: guessing here
    would denormalise predictions with the wrong statistics and report nothing.
    """
    missing = [k for k in INHERITED_KEYS if k not in train_cfg]
    if missing:
        raise ValueError(
            f"{checkpoint}: embedded training config is missing {missing}. It predates the "
            f"current config schema; re-train, or hand-edit the checkpoint's 'config' dict."
        )
    return {k: train_cfg[k] for k in INHERITED_KEYS}
