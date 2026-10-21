from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from utils.transforms import SharedTransform, ValueTransform


class Pipeline(nn.Module):
    """Routes transforms to the streams they belong to.

        ``shared`` runs on ``(x, cond)`` together (crops, flips) and is skipped when
        ``augment=False``. ``x`` and ``cond`` hold per-stream ``ValueTransform``s.
    """

    def __init__(
        self,
        shared: Sequence[SharedTransform] = (),
        x: Sequence[ValueTransform] = (),
        cond: Sequence[ValueTransform] = (),
    ):
        super().__init__()
        for t in shared:
            if not isinstance(t, SharedTransform):
                raise TypeError(f"shared expects SharedTransform, got {type(t).__name__}.")
        for t in (*x, *cond):
            if not isinstance(t, ValueTransform):
                raise TypeError(f"x/cond expect ValueTransform, got {type(t).__name__}.")

        self.shared = nn.ModuleList(shared)
        self.x = nn.ModuleList(x)
        self.cond = nn.ModuleList(cond)

    def pre(
        self, x: torch.Tensor, cond: torch.Tensor, augment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw -> model space."""
        if augment:
            for t in self.shared:
                x, cond = t(x, cond)
        for t in self.x:
            x = t(x)
        for t in self.cond:
            cond = t(cond)
        return x, cond

    def post(
        self, x: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Model space -> raw. A stream passed as None is skipped and returned as None."""
        if x is not None:
            for t in reversed(self.x):
                x = t.inverse(x)
        if cond is not None:
            for t in reversed(self.cond):
                cond = t.inverse(cond)
        return x, cond

    @staticmethod
    def _names(modules) -> str:
        return "[" + ", ".join(type(t).__name__ for t in modules) + "]"

    def __repr__(self) -> str:
        return (f"Pipeline(shared={self._names(self.shared)}, x={self._names(self.x)}, "
                f"cond={self._names(self.cond)})")


class ResidualPipeline(Pipeline):
    """Residual variant: r = (L(x1) - L(x_det)) / residual_std, conditioned on x_det.

    x1 and x_det are the same physical quantity, so they share `self.x` — the
    difference is only defined in a single space.

    ``pre`` -> ``(r, cond_t, x_det_t)``; ``from_residual`` then ``post(x=...)``
    rebuild physical units.
    """

    def __init__(self, shared=(), x=(), cond=(), residual_std=1.0):
        super().__init__(shared=shared, x=x, cond=cond)
        if residual_std <= 0:
            raise ValueError(f"residual_std must be > 0, got {residual_std!r}.")
        self.residual_std = float(residual_std)

    def pre(self, x, cond, x_det, augment=True):
        """Raw -> (residual target, transformed cond, transformed x_det baseline).

        The shared geometric transforms must see ``x`` and ``x_det`` in a single draw,
        or a second flip/crop would desync them.
        """
        if augment:
            # stack, not cat: a new leading axis works for [C, H, W] and [B, C, H, W]
            # alike, whereas cat would extend the batch axis and pair up samples 0 and 1.
            pair = torch.stack([x, x_det])
            for t in self.shared:
                pair, cond = t(pair, cond)
            x, x_det = pair[0], pair[1]
        for t in self.x:
            x, x_det = t(x), t(x_det)
        for t in self.cond:
            cond = t(cond)
        return (x - x_det) / self.residual_std, cond, x_det

    def from_residual(self, r, x_det_t):
        """Residual target + transformed baseline -> ``x``-space field.

        Feed the result to ``post(x=...)`` to get physical units.
        """
        return x_det_t + r * self.residual_std

    def __repr__(self):
        return (f"ResidualPipeline(shared={self._names(self.shared)}, x={self._names(self.x)}, "
                f"cond={self._names(self.cond)}, residual_std={self.residual_std:.4f})")
