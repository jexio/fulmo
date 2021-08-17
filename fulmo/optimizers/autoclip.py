from typing import List

import numpy as np
import torch


class AutoClip:
    """Implements AutoClip from paper https://arxiv.org/pdf/2007.14469.pdf."""

    def __init__(self, percentile: float) -> None:
        """Create a new instance of `AutoClip`."""
        self.grad_history: List[float] = list()
        self.percentile = percentile

    def __call__(self, model: torch.nn.Module) -> None:
        """Compute gradients."""
        grad_norm = self._compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        clip_value = np.percentile(self.grad_history, self.percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    @staticmethod
    def _compute_grad_norm(model: torch.nn.Module) -> float:
        """Compute gradients."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm


__all__ = ["AutoClip"]
