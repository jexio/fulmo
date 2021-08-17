from typing import Any

import torch
import torch.nn as nn

from .adaptive import AdaptiveAvgMaxPool2d, AdaptiveCatAvgMaxPool2d, AdaptiveConcatPool2d
from .gem import GeM
from .pyramid import SpatialPyramidPooling
from .scse import SCSEPooling


pool_dict = dict(
    avgmax=AdaptiveAvgMaxPool2d,
    catavgmax=AdaptiveCatAvgMaxPool2d,
    adaptive_concat=AdaptiveConcatPool2d,
    gem=GeM,
    pyramid=SpatialPyramidPooling,
    scse=SCSEPooling,
    max=nn.AdaptiveMaxPool2d,
    avg=nn.AdaptiveAvgPool2d,
)


class Flatten(nn.Module):
    """Flatten operator."""

    def __init__(self, full: bool = False) -> None:
        """Create a new instance of Flatten."""
        super().__init__()
        self.full = full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size."""

    def __init__(
        self, output_size: int = 1, in_channels: int = 1, pool_name: str = "avg", **kwargs: Any  # noqa: ANN003
    ) -> None:
        """Create a new instance of SelectAdaptivePool2d."""
        super(SelectAdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_name = pool_name

        if pool_name not in pool_dict:
            raise KeyError(f"Pooling {pool_name} does not implemented")

        if pool_name == "avgmax":
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_name == "catavgmax":
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_name == "adaptive_concat":
            self.pool = AdaptiveConcatPool2d(output_size, **kwargs)
        elif pool_name == "gem":
            self.pool = GeM(**kwargs)
        elif pool_name == "pyramid":
            self.pool = SpatialPyramidPooling(**kwargs)
        elif pool_name == "scse":
            self.pool = SCSEPooling(in_channels=in_channels, output_size=output_size)
        elif pool_name == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif pool_name == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    @property
    def multiplication_coefficient(self) -> int:
        """Get coefficient related to the type of the selected pooling layer."""
        if self.pool_name == "catavgmax":
            return 2
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.pool(x)


__all__ = ["SelectAdaptivePool2d", "Flatten"]
