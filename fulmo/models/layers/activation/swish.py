import torch
import torch.nn as nn

from .functional import swish


class Swish(nn.Module):
    """Swish activation function. https://arxiv.org/abs/1710.05941"""

    def __init__(self, beta: float = 1.0) -> None:
        """Create a new instance of Swish operator."""
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return swish(x, self.beta)


__all__ = ["Swish"]
