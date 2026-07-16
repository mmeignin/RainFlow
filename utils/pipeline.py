from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from utils.transforms import SharedTransform, ValueTransform


class Pipeline(nn.Module):
    """Routes transforms to the streams they belong to.
    ``shared`` runs on ``(x, cond)`` together (crops, flips) and is skipped when
    ``augment=False`` — inference wants the geometry left alone. ``x`` and
    ``cond`` hold per-stream ``ValueTransform``s, applied in order by ``pre``
    and undone in reverse by ``post``.
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

    def __repr__(self) -> str:
        def names(m):
            return "[" + ", ".join(type(t).__name__ for t in m) + "]"

        return f"Pipeline(shared={names(self.shared)}, x={names(self.x)}, cond={names(self.cond)})"
