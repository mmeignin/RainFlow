
import random

import torch
import torch.nn as nn


def _broadcast(stat_1d: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a 1-D per-channel statistic to broadcast against an `ndim` tensor."""
    if ndim == 3:
        return stat_1d.view(-1, 1, 1)      # [C, H, W]
    if ndim == 4:
        return stat_1d.view(1, -1, 1, 1)   # [B, C, H, W]
    raise ValueError(f"Unsupported tensor ndim={ndim}, expected 3 ([C,H,W]) or 4 ([B,C,H,W])")


class SharedTransform(nn.Module):
    """Applied to both streams. Not invertible."""

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        raise NotImplementedError


class ValueTransform(nn.Module):
    """Applied to one stream. Invertible."""

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ── shared ────────────────────────────────────────────────────────────────────

class RandomFlip(SharedTransform):
    """Independent horizontal and vertical flips, each with probability `p`."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x, cond):
        if random.random() < self.p:
            x, cond = torch.flip(x, dims=[-1]), torch.flip(cond, dims=[-1])
        if random.random() < self.p:
            x, cond = torch.flip(x, dims=[-2]), torch.flip(cond, dims=[-2])
        return x, cond

class RandomCrop(SharedTransform):
    """Crop both streams to `size` at one randomly chosen offset."""

    def __init__(self, size):
        super().__init__()
        self.size = tuple(size)

    def forward(self, x, cond):
        h, w = self.size
        H, W = x.shape[-2:]
        if h > H or w > W:
            raise ValueError(f"Crop size {self.size} larger than image {H}x{W}")
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        return (
            x[..., top:top + h, left:left + w],
            cond[..., top:top + h, left:left + w],
        )

class CenterCrop(SharedTransform):
    """Crop both streams to `size`, centered."""

    def __init__(self, size):
        super().__init__()
        self.size = tuple(size)

    def forward(self, x, cond):
        h, w = self.size
        H, W = x.shape[-2:]
        if h > H or w > W:
            raise ValueError(f"Crop size {self.size} larger than image {H}x{W}")
        top, left = (H - h) // 2, (W - w) // 2
        return (
            x[..., top:top + h, left:left + w],
            cond[..., top:top + h, left:left + w],
        )

# ── value ─────────────────────────────────────────────────────────────────────

class Normalize(ValueTransform):
    """Per-channel z-score.

    forward: (t - mean) / (sigma_factor * std)
    inverse: t * sigma_factor * std + mean

    `mean`/`std` are 1-D, length C, matching the stream's channel count.
    """

    def __init__(self, mean, std, sigma_factor: float = 1.0):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).flatten())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).flatten())
        self.sigma_factor = float(sigma_factor)

    def forward(self, t):
        mean = _broadcast(self.mean, t.dim()).to(t.device)
        scale = _broadcast(self.sigma_factor * self.std, t.dim()).to(t.device)
        return (t - mean) / scale

    def inverse(self, t):
        mean = _broadcast(self.mean, t.dim()).to(t.device)
        scale = _broadcast(self.sigma_factor * self.std, t.dim()).to(t.device)
        return t * scale + mean


class MinMax(ValueTransform):
    """Per-channel min-max into [-1, 1].

    forward: 2 * (t - vmin) / (vmax - vmin) - 1
    inverse: (t + 1) / 2 * (vmax - vmin) + vmin

    `vmin`/`vmax` are 1-D, length C, matching the stream's channel count.
    """

    def __init__(self, vmin, vmax):
        super().__init__()
        self.register_buffer("vmin", torch.as_tensor(vmin, dtype=torch.float32).flatten())
        self.register_buffer("vmax", torch.as_tensor(vmax, dtype=torch.float32).flatten())

    def forward(self, t):
        vmin = _broadcast(self.vmin, t.dim()).to(t.device)
        vmax = _broadcast(self.vmax, t.dim()).to(t.device)
        return 2 * (t - vmin) / (vmax - vmin) - 1

    def inverse(self, t):
        vmin = _broadcast(self.vmin, t.dim()).to(t.device)
        vmax = _broadcast(self.vmax, t.dim()).to(t.device)
        return (t + 1) / 2 * (vmax - vmin) + vmin


class Log1pZScore(ValueTransform):
    """Maps rain [0, inf) to roughly N(0, 1). `mean`/`std` are scalars over log1p pixels.

    forward: (log1p(t) - mean) / std
    inverse: expm1(t * std + mean), clamped at 0
    """

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, t):
        return (torch.log1p(t.clamp_min(0)) - self.mean) / self.std

    def inverse(self, t):
        return torch.expm1(t * self.std + self.mean).clamp_min(0)
