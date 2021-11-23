from dataclasses import dataclass, field
from typing import List

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..dataclass.configurations import LearningRateSchedulerConfigs
from . import register_scheduler


@dataclass
class CosineAnnealingWarmRestartsConfigs(LearningRateSchedulerConfigs):
    """CosineAnnealing learning rate scheduler config."""

    name: str = field(default="cosine_annealing_warm_restarts", metadata={"help": "Name of learning rate scheduler."})
    T_0: int = field(default=1, metadata={"help": "Number of iterations for the first restart."})
    T_mult: int = field(default=1, metadata={"help": "A factor increases :math:`T_{i}` after a restart."})
    eta_min: float = field(default=0.0, metadata={"help": "Minimum learning rate."})
    last_epoch: int = field(default=-1, metadata={"help": "The index of last epoch."})
    verbose: bool = field(default=False, metadata={"help": "If ``True``, prints a message to stdout for each update."})

    @staticmethod
    def _scheduler_keys() -> List[str]:
        """Get all keys related to `CosineAnnealingWarmRestarts` scheduler."""
        return ["T_0", "T_mult", "eta_min", "last_epoch", "verbose"]


@register_scheduler("cosine_annealing_warm_restarts", dataclass=CosineAnnealingWarmRestartsConfigs)
class CosineAnnealingWarmRestartsScheduler(CosineAnnealingWarmRestarts):
    """Set the learning rate of each parameter group using a cosine annealing schedule.

    Attributes:
        optimizer: Wrapped optimizer.
        configs: Configuration set.

    References: https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: Optimizer,
        configs: DictConfig,
    ) -> None:
        """Create a new instance of CosineAnnealingWarmRestartsScheduler."""
        super(CosineAnnealingWarmRestartsScheduler, self).__init__(optimizer, **configs.scheduler_parameters)


__all__ = ["CosineAnnealingWarmRestartsScheduler"]
