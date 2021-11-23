import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GeM(nn.Module):
    """Generalized Mean Pooling. https://arxiv.org/abs/1902.05509v2"""

    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        """Create a new instance of GeM pooling layer."""
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


__all__ = ["GeM"]
