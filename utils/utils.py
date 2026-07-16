import os
import random

import numpy as np
import pandas as pd
import torch

from utils.pipeline import Pipeline
from utils.transforms import Log1pZScore, MinMax, Normalize, RandomFlip


def set_seed(seed: int) -> None:
    """Set the seed for the random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infiniteloop(dataloader):
    """Cycle a DataLoader indefinitely, yielding batches forever."""
    while True:
        for batch in dataloader:
            yield batch

@torch.no_grad()
def ema(source, target, decay):
    """In-place exponential moving average update of `target`'s parameters toward `source`'s."""
    for targ, src in zip(target.parameters(), source.parameters()):
        targ.detach().mul_(decay).add_(src.detach(), alpha=1 - decay)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline construction
# ──────────────────────────────────────────────────────────────────────────────

RAIN_CHANNEL = "rain_rate"


def read_norm_stats(csv_path: str, channels: list, columns: list) -> dict:
    """Read `columns` for `channels` from the normalisation CSV.

    Returns {column: [value per channel, in `channels` order]}.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"normalisation CSV not found: {csv_path}")

    table = pd.read_csv(csv_path)
    if "channel" not in table.columns:
        raise ValueError(f"{csv_path}: missing 'channel' column; found {list(table.columns)}.")
    if absent := [c for c in columns if c not in table.columns]:
        raise ValueError(f"{csv_path}: missing columns {absent}; found {list(table.columns)}.")

    table = table.set_index("channel")
    if unknown := [c for c in channels if c not in table.index]:
        raise ValueError(f"{csv_path}: no rows for channels {unknown}; has {list(table.index)}.")

    def cell(channel, column):
        value = table.loc[channel, column]
        if pd.isna(value):
            raise ValueError(f"{csv_path}: column {column!r} is empty for channel {channel!r}.")
        return float(value)

    return {col: [cell(ch, col) for ch in channels] for col in columns}


def _normalize(stats, config):
    return Normalize(stats["mean"], stats["std"], config.get("sigma_factor", 1))


def _minmax(stats, config):
    return MinMax(stats["vmin"], stats["vmax"])


def _log1p(stats, config):
    return Log1pZScore(stats["log1p_mean"][0], stats["log1p_std"][0])


# transform type -> (stats columns it needs, factory, streams it may be used on, extra config keys allowed)
TRANSFORMS = {
    "Normalize":   (["mean", "std"],              _normalize, {"x", "cond"}, {"sigma_factor"}),
    "MinMax":      (["vmin", "vmax"],             _minmax,    {"cond"},      set()),
    "Log1pZScore": (["log1p_mean", "log1p_std"],  _log1p,     {"x"},         set()),
}


def _value_transform(stream: str, config: dict, csv_path: str, channels: list) -> list:
    """Build the zero-or-one value transform for one stream."""
    kind = config.get("type")
    if kind is None:
        return []

    allowed = sorted(k for k, (_, _, streams, _) in TRANSFORMS.items() if stream in streams)
    if kind not in TRANSFORMS:
        raise ValueError(f"unknown {stream} transform type {kind!r}; expected one of {allowed}.")
    if stream not in TRANSFORMS[kind][2]:
        raise ValueError(f"{kind!r} cannot be applied to the {stream!r} stream; expected one of {allowed}.")
    if not csv_path:
        raise ValueError(f"{stream} transform {kind!r} requires a top-level 'norm_csv' path.")

    columns, factory, _, extra_keys = TRANSFORMS[kind]
    unknown = set(config) - {"type"} - extra_keys
    if unknown:
        raise ValueError(
            f"{kind!r} {stream} transform got unknown keys {sorted(unknown)}; "
            f"expected 'type' and {sorted(extra_keys) or 'no extra keys'}."
        )
    return [factory(read_norm_stats(csv_path, channels, columns), config)]


def build_pipeline_from_config(config: dict, channel_list: list) -> Pipeline:
    """Build a `Pipeline` from a config dict.

    All keys optional; an absent transform block leaves that stream in raw units.

        norm_csv:       path to the single normalisation CSV
        random_flip:    bool, with flip_prob (default 0.5)
        x_transform:    {type: Normalize | Log1pZScore}
        cond_transform: {type: Normalize | MinMaxs}

    `x` reads the `rain_rate` row; `cond` reads one row per `channel_list` entry,
    in order. Naming a transform without `norm_csv`, or naming an unknown type,
    raises — silently skipping either would train on unnormalised data.
    """
    csv_path = config.get("norm_csv")

    shared = []
    if config.get("random_flip", False):
        shared.append(RandomFlip(p=config.get("flip_prob", 0.5)))

    ### If you want to add Crops you should do it here 

    return Pipeline(
        shared=shared,
        x=_value_transform("x", config.get("x_transform") or {}, csv_path, [RAIN_CHANNEL]),
        cond=_value_transform("cond", config.get("cond_transform") or {}, csv_path, channel_list),
    )
