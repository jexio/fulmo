from typing import Optional

import torch
import torch.nn as nn

from .functional import adaptive_avgmax_pool2d, adaptive_catavgmax_pool2d


class AdaptiveConcatPool2d(nn.Module):
    """Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."""

    def __init__(self, sz: Optional[int] = None, multiplier: Optional[float] = None, mode: int = 0) -> None:
        """Create a new instance of AdaptiveConcatPool2d."""
        super().__init__()
        self.output_size = sz or 1
        self.multiplier = multiplier or 1.0
        self.mode = mode
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        mp = self.mp(x)
        ap = self.ap(x)
        if self.mode == 0:
            return self.multiplier * torch.cat([mp, ap], 1)
        return self.multiplier * (mp + ap)


class AdaptiveAvgMaxPool2d(nn.Module):
    """Layer that sums AdaptiveAvgPool2d and AdaptiveMaxPool2d"""

    def __init__(self, output_size: int = 1, multiplier: float = 0.5) -> None:
        """Create a new instance of AdaptiveAvgMaxPool2d."""
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return adaptive_avgmax_pool2d(x, self.output_size, self.multiplier)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    """Layer that concats AdaptiveAvgPool2d and AdaptiveMaxPool2d"""

    def __init__(self, output_size: int = 1) -> None:
        """Create a new instance of AdaptiveCatAvgMaxPool2d."""
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return adaptive_catavgmax_pool2d(x, self.output_size)


__all__ = ["AdaptiveConcatPool2d", "AdaptiveAvgMaxPool2d", "AdaptiveCatAvgMaxPool2d"]
