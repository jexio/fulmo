import torch
import torch.nn.functional as F


def adaptive_avgmax_pool2d(x: torch.Tensor, output_size: int = 1, multiplier: float = 0.5) -> torch.Tensor:
    """Apply avg and max pooling and mean result."""
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return multiplier * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x: torch.Tensor, output_size: int = 1) -> torch.Tensor:
    """Apply avg and max pooling and concatenate it into one tensor."""
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


__all__ = ["adaptive_avgmax_pool2d", "adaptive_catavgmax_pool2d"]
