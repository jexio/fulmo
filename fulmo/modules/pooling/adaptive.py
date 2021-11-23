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
        if mode == 0:
            self.pooler = adaptive_catavgmax_pool2d
        else:
            self.pooler = adaptive_avgmax_pool2d

        if multiplier:
            self.weight = multiplier
        else:
            self.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.weight * self.pooler(x, self.output_size)


class AdaptiveAvgMaxPool2d(nn.Module):
    """Layer that sums AdaptiveAvgPool2d and AdaptiveMaxPool2d"""

    def __init__(self, output_size: int = 1, multiplier: Optional[float] = None) -> None:
        """Create a new instance of AdaptiveAvgMaxPool2d."""
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        if multiplier:
            self.weight = multiplier
        else:
            self.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.weight * adaptive_avgmax_pool2d(x, self.output_size)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    """Layer that concats AdaptiveAvgPool2d and AdaptiveMaxPool2d"""

    def __init__(self, output_size: int = 1, multiplier: Optional[float] = None) -> None:
        """Create a new instance of AdaptiveCatAvgMaxPool2d."""
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        if multiplier:
            self.weight = multiplier
        else:
            self.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.weight * adaptive_catavgmax_pool2d(x, self.output_size)


__all__ = ["AdaptiveConcatPool2d", "AdaptiveAvgMaxPool2d", "AdaptiveCatAvgMaxPool2d"]
