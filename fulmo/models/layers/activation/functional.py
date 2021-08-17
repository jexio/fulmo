from typing import Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function


def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish activation."""
    return x * torch.tanh(F.softplus(x))


def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Swish activation."""
    return SwishOP.apply(x, beta)


class SwishOP(Function):
    @staticmethod
    def forward(ctx: Function, tensor: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Run forward pass."""
        ctx.save_for_backward(tensor)
        ctx.beta = beta
        swish = tensor / (1 + torch.exp(-beta * tensor))
        return swish

    @staticmethod
    def backward(ctx: Function, grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Run backward pass."""
        tensor = ctx.saved_tensors[0]
        beta = ctx.beta
        grad_swish = (torch.exp(-beta * tensor) * (1 + beta * tensor) + 1) / (1 + torch.exp(-beta * tensor)) ** 2
        grad_swish = grad_outputs * grad_swish
        return grad_swish, None


__all__ = ["mish", "swish"]
