from typing import List, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer: wrapped optimizer.
        end_lr: the initial learning rate which is the lower boundary of the test. Default: 10.
        num_iter: the number of iterations over which the test occurs. Default: 100.
        last_epoch: the index of last epoch. Default: -1.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, end_lr: Optional[float], num_iter: Optional[int], last_epoch: int = -1
    ) -> None:
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        """Compute learning rate using chainable form of the scheduler."""
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


__all__ = ["ExponentialLR"]
