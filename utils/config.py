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
    # flow matching
    "prior", "time_sampling",
    # model
    "num_channel", "image_size", "channel_mult", "attention_resolutions",
    "num_heads", "num_head_channels", "use_scale_shift_norm", "dropout",
    # optimisation
    "total_steps", "batch_size", "num_workers", "lr", "weight_decay", "grad_clip",
    "warmup_steps", "scheduler", "ema_decay", "cfg_drop", "seed",
    # evaluation and checkpointing (all cadences in steps)
    "eval_every", "log_every", "save_every",
    # output
    "output_dir", "exp_name", "debug",
)

# Optional training keys: tolerated (no "unknown key" note) but not required.
OPTIONAL_TRAIN_KEYS = (
    "residual",           # bool, default False — two-stage residual training
    "student_t_nu",       # float > 2, default 5 — read only when prior="student_t"
    "time_sampling_std",  # float, default 1.0 — read when time_sampling is
                           # "logit_normal" or "logit_normal_power_mix"
    "time_sampling_k",    # float > 0, default 1.5 (3.0 for the mix) — read when
                           # time_sampling is "power" or "logit_normal_power_mix"
    "time_sampling_mix_prob",  # float in [0, 1], default 0.2 — read only when
                               # time_sampling="logit_normal_power_mix"
)

# Everything needed to rebuild the model as trained, undo its transforms, and sample
# from the source it was trained against. Every config-derived argument to build_unet
# comes from here, except `residual`, which inference reads from the embedded config.
INHERITED_KEYS = (
    "channels", "norm_csv", "x_transform", "cond_transform",
    "num_channel", "image_size", "channel_mult", "attention_resolutions",
    "num_heads", "num_head_channels", "use_scale_shift_norm", "dropout",
    "prior",
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


def _validate(config: dict, required, path: str, consumer: str, optional=()) -> dict:
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    unknown = sorted(set(config) - set(required) - set(optional))
    if unknown:
        print(f"note: {path} has keys not read by {consumer}: {unknown}", flush=True)
    return config


def load_config(path: str) -> dict:
    """Load and validate a training config."""
    config = _validate(_load_yaml(path), TRAIN_KEYS, path, "training", OPTIONAL_TRAIN_KEYS)

    prior = config["prior"]
    if prior not in ("gaussian", "student_t"):
        raise ValueError(f"{path}: prior must be 'gaussian' or 'student_t', got {prior!r}.")
    if prior == "student_t" and config.get("student_t_nu", 5) <= 2:
        raise ValueError(f"{path}: student_t_nu must be > 2 for finite variance.")

    time_sampling = config["time_sampling"]
    if time_sampling not in ("uniform", "logit_normal", "power", "logit_normal_power_mix"):
        raise ValueError(
            f"{path}: time_sampling must be 'uniform', 'logit_normal', 'power' or "
            f"'logit_normal_power_mix', got {time_sampling!r}."
        )
    time_sampling_k = config.get("time_sampling_k", 1.5)
    if time_sampling in ("power", "logit_normal_power_mix") and time_sampling_k <= 0:
        raise ValueError(
            f"{path}: time_sampling_k must be > 0 (k=1 is uniform, k>1 shifts mass "
            f"toward the noise end), got {time_sampling_k!r}."
        )
    time_sampling_mix_prob = config.get("time_sampling_mix_prob", 0.2)
    if time_sampling == "logit_normal_power_mix" and not 0.0 <= time_sampling_mix_prob <= 1.0:
        raise ValueError(
            f"{path}: time_sampling_mix_prob must be in [0, 1], got "
            f"{time_sampling_mix_prob!r}."
        )

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

    No defaulting: a value invented here would denormalise with the wrong statistics, or
    build the wrong architecture, and report nothing.
    """
    missing = [k for k in INHERITED_KEYS if k not in train_cfg]
    if missing:
        raise ValueError(
            f"{checkpoint}: embedded config is missing {missing} — it predates the current "
            f"schema. Re-train, or hand-edit the checkpoint's 'config' dict."
        )
    return {k: train_cfg[k] for k in INHERITED_KEYS}
