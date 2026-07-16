import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _load_payload(path: str) -> dict:
    try:
        raw = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load sample {path!r}") from exc

    data = raw.item() if raw.shape == () else raw
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a dict payload, got {type(data).__name__}.")
    return data


def _as_hw_float32(array, path: str, key: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{path}: {key!r} must be 2-D [H, W], got shape {array.shape}.")
    return np.ascontiguousarray(array, dtype=np.float32)


class RainIRDataset(Dataset):
    """Rain rate (target) conditioned on infrared channels, for flow matching.

    Each sample is a ``.npy`` file holding a pickled dict: one ``rain_key``
    entry plus one 2-D ``[H, W]`` array per conditioning channel. ``file_list``
    holds absolute paths.

    ``__getitem__`` returns float32 ``(x, cond)`` of shape ``[1, H, W]`` and
    ``[C, H, W]``, channels stacked in ``cond_keys`` order; with
    ``return_timestamps``, the manifest timestamp is appended as a third
    element. ``pipeline``, if given, is applied as ``pipeline.pre(x, cond)``.
    """

    def __init__(
        self,
        cond_keys: Sequence[str],
        file_list: Sequence[str],
        rain_key: str = "rain_rate",
        pipeline: Optional[object] = None,
        timestamps: Optional[Sequence[str]] = None,
        return_timestamps: bool = False,
    ):
        self.cond_keys = list(cond_keys)
        self.file_list = list(file_list)
        self.rain_key = rain_key
        self.pipeline = pipeline
        self.timestamps = list(timestamps) if timestamps is not None else None
        self.return_timestamps = return_timestamps
        self._validate()

    def _validate(self) -> None:
        keys, files, ts = self.cond_keys, self.file_list, self.timestamps

        if not keys:
            raise ValueError("cond_keys must be non-empty.")
        if len(set(keys)) != len(keys):
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"cond_keys contains duplicates: {dupes}")
        if self.rain_key in keys:
            raise ValueError(f"cond_keys must not contain the target key {self.rain_key!r}.")
        if not files:
            raise ValueError("file_list is empty; nothing to load.")
        if self.pipeline is not None and not callable(getattr(self.pipeline, "pre", None)):
            raise TypeError(
                f"pipeline must expose a callable .pre(x, cond); "
                f"got {type(self.pipeline).__name__}."
            )
        if self.return_timestamps and ts is None:
            raise ValueError("return_timestamps=True requires timestamps to be provided.")
        if ts is not None and len(ts) != len(files):
            raise ValueError(
                f"timestamps has {len(ts)} entries but file_list has {len(files)}; "
                f"they must be aligned."
            )

        # One stat per sample turns a mid-epoch worker crash into a startup error.
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"{len(missing)} of {len(files)} sample files missing: "
                f"{', '.join(missing[:5])}{more}"
            )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        data = _load_payload(path)

        missing = [k for k in (self.rain_key, *self.cond_keys) if k not in data]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}; available keys: {sorted(data)}")

        rain = _as_hw_float32(data[self.rain_key], path, self.rain_key)
        channels = [_as_hw_float32(data[k], path, k) for k in self.cond_keys]

        bad = {k: c.shape for k, c in zip(self.cond_keys, channels) if c.shape != rain.shape}
        if bad:
            raise ValueError(
                f"{path}: channels must match {self.rain_key} shape {rain.shape}; got {bad}"
            )

        x = torch.from_numpy(rain).unsqueeze(0)
        cond = torch.from_numpy(np.stack(channels))
        if self.pipeline is not None:
            x, cond = self.pipeline.pre(x, cond)

        return (x, cond, self.timestamps[idx]) if self.return_timestamps else (x, cond)

    def __repr__(self) -> str:
        pipeline = type(self.pipeline).__name__ if self.pipeline is not None else None
        return (
            f"{type(self).__name__}(n={len(self)}, cond_keys={self.cond_keys}, "
            f"rain_key={self.rain_key!r}, pipeline={pipeline})"
        )

    @classmethod
    def from_csv(
        cls,
        cond_keys: Sequence[str],
        csv_file: str,
        rain_key: str = "rain_rate",
        pipeline: Optional[object] = None,
        debug: bool = False,
        debug_samples: int = 64,
        return_timestamps: bool = False,
    ) -> "RainIRDataset":
        """Build from a manifest with an absolute-path ``path`` column and an
        optional ``timestamp`` column. ``debug`` keeps only the first
        ``debug_samples`` rows."""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        meta = pd.read_csv(csv_file)
        if "path" not in meta.columns:
            raise ValueError(
                f"{csv_file}: must contain a 'path' column; found {list(meta.columns)}."
            )
        if return_timestamps and "timestamp" not in meta.columns:
            raise ValueError(
                f"{csv_file}: return_timestamps=True requires a 'timestamp' column; "
                f"found {list(meta.columns)}."
            )

        if debug:
            meta = meta.iloc[:debug_samples]
            print(f"DEBUG MODE ACTIVATED — using first {len(meta)} samples", flush=True)

        has_ts = "timestamp" in meta.columns
        return cls(
            cond_keys=cond_keys,
            file_list=meta["path"].tolist(),
            rain_key=rain_key,
            pipeline=pipeline,
            timestamps=meta["timestamp"].astype(str).tolist() if has_ts else None,
            return_timestamps=return_timestamps,
        )


def _summarize(name: str, t: torch.Tensor) -> str:
    finite = t[torch.isfinite(t)]
    rng = f"[{finite.min():.2f}, {finite.max():.2f}]" if finite.numel() else "[all-nan]"
    nan = 100 * torch.isnan(t).sum().item() / t.numel()
    return f"{name:5s} {str(tuple(t.shape)):18s} {rng:22s} nan={nan:.1f}%"


def _main() -> None:
    """Smoke test"""
    import argparse
    from torch.utils.data import DataLoader

    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--cond-keys", default="WV_062,WV_073,IR_087,IR_097,IR_108,IR_120,IR_134")
    p.add_argument("--rain-key", default="rain_rate")
    p.add_argument("--n", type=int, default=8, help="samples to load")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    ds = RainIRDataset.from_csv(
        cond_keys=args.cond_keys.split(","),
        csv_file=args.csv,
        rain_key=args.rain_key,
        debug=True,
        debug_samples=args.n,
        return_timestamps=True,
    )
    print(ds)

    x, cond, ts = ds[0]
    print(f"\nsample 0  ts={ts}\n  {_summarize('x', x)}\n  {_summarize('cond', cond)}")

    loader = DataLoader(ds, args.batch_size, num_workers=args.num_workers)
    bx, bcond, bts = next(iter(loader))
    print(f"\nfirst batch (num_workers={args.num_workers})")
    print(f"  {_summarize('x', bx)}\n  {_summarize('cond', bcond)}\n  ts    {list(bts)}")


if __name__ == "__main__":
    _main()
