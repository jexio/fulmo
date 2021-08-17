from typing import List, Optional

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """One of the implementations of the `LabelSmoothing` loss."""

    def __init__(
        self, n_classes: int, smoothing: float = 0.1, dim: int = -1, weight: Optional[List[float]] = None
    ) -> None:
        """Create a new instance of `LabelSmoothingLoss`.

        Args:
            n_classes: Number of classes of the classification problem
            smoothing: smoothing factor
            dim: A dimension along which log_softmax will be computed
            weight: A manual rescaling weight if provided it’s repeated to match input tensor shape

        Raises:
            ValueError: if `smoothing` greater than 1 or less than 0
        """
        super(LabelSmoothingLoss, self).__init__()
        if 0.0 > smoothing > 1.0:
            raise ValueError("smoothing must be in range [0, 1].")

        self._confidence = 1.0 - smoothing
        self._smoothing = smoothing
        self._weight: Optional[torch.Tensor] = torch.FloatTensor(weight) if weight else None
        self._n_classes = n_classes
        self._dim = dim

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        outputs = outputs.log_softmax(dim=self._dim)

        if self._weight is not None:
            outputs = outputs * self._weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(outputs)
            true_dist.fill_(self._smoothing / (self._n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self._confidence)
        return torch.mean(torch.sum(-true_dist * outputs, dim=self._dim))


__all__ = ["LabelSmoothingLoss"]
