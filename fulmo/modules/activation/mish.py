import torch
import torch.nn as nn

from .functional import mish


class Mish(nn.Module):
    """Mish activation from 'Mish: A Self Regularized Non-Monotonic Activation Function'."""

    def __init__(self) -> None:
        """Create a new instance of Mish operator."""
        super(Mish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return mish(x)


__all__ = ["Mish"]
