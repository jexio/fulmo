import torch
import torch.nn as nn


class SCSEPooling(nn.Module):
    """https://arxiv.org/abs/1803.02579"""

    def __init__(self, in_channels: int, output_size: int = 1) -> None:
        """Create a new instance of SCSEPooling layer."""
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.ap(x * self.sSE(x))


__all__ = ["SCSEPooling"]
