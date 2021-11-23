from dataclasses import dataclass, field
from typing import List, Optional

import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler

from ..dataclass.configurations import LearningRateSchedulerConfigs
from . import register_scheduler


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer: Wrapped optimizer.
        end_lr: The initial learning rate which is the lower boundary of the test. Default: 10.
        num_iter: The number of iterations over which the test occurs. Default: 100.
        last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        end_lr: Optional[float] = 10,
        num_iter: Optional[int] = 100,
        last_epoch: int = -1,
    ) -> None:
        """Create a new instance of ExponentialLR."""
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        """Compute learning rate using chainable form of the scheduler."""
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


@dataclass
class ExponentialLearningRateConfigs(LearningRateSchedulerConfigs):
    """Exponential learning rate scheduler config."""

    name: str = field(default="exponential_learning_rate", metadata={"help": "Name of learning rate scheduler."})
    end_lr: int = field(
        default=10, metadata={"help": "The initial learning rate which is the lower boundary of the test."}
    )
    num_iter: int = field(default=100, metadata={"help": "The number of iterations over which the test occurs."})
    last_epoch: int = field(default=-1, metadata={"help": "The index of last epoch."})

    @staticmethod
    def _scheduler_keys() -> List[str]:
        """Get all keys related to `ExponentialLR` scheduler."""
        return ["end_lr", "num_iter", "last_epoch"]


@register_scheduler("exponential_learning_rate", dataclass=ExponentialLearningRateConfigs)
class ExponentialLearningRateScheduler(ExponentialLR):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Attributes:
        optimizer: Wrapped optimizer.
        configs: Configuration set.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        configs: DictConfig,
    ) -> None:
        """Create a new instance of ExponentialLearningRateScheduler."""
        super(ExponentialLearningRateScheduler, self).__init__(optimizer, **configs.scheduler_parameters())


__all__ = ["ExponentialLearningRateScheduler"]
