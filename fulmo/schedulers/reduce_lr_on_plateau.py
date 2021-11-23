from dataclasses import dataclass, field
from typing import List

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..dataclass.configurations import LearningRateSchedulerConfigs
from . import register_scheduler


@dataclass
class ReduceLROnPlateauConfigs(LearningRateSchedulerConfigs):
    """Reduce learning rate on plateau scheduler config."""

    name: str = field(default="reduce_lr_on_plateau", metadata={"help": "Name of learning rate scheduler."})
    mode: str = field(
        default="min",
        metadata={
            "help": "One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; "
            "in max mode it will be reduced when the quantity monitored has stopped increasing."
        },
    )
    factor: float = field(
        default=0.1, metadata={"help": "Factor by which the learning rate will be reduced. new_lr = lr * factor."}
    )
    patience: int = field(
        default=10,
        metadata={
            "help": "Number of epochs with no improvement after which learning rate will be reduced. "
            "For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, "
            "and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then."
        },
    )
    threshold: float = field(
        default=1.0e-4,
        metadata={"help": "Threshold for measuring the new optimum, to only focus on significant changes."},
    )
    threshold_mode: str = field(
        default="rel",
        metadata={
            "help": "One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) "
            "in ‘max’ mode or best * ( 1 - threshold ) in min mode. "
            "In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode."
        },
    )
    cooldown: int = field(
        default=0,
        metadata={"help": "Number of epochs to wait before resuming normal operation after lr has been reduced."},
    )
    min_lr: List[float] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "A scalar or a list of scalars. "
            "A lower bound on the learning rate of all param groups or each group respectively."
        },
    )
    eps: float = field(
        default=1.0e-8,
        metadata={
            "help": "Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, "
            "the update is ignored."
        },
    )
    verbose: bool = field(default=False, metadata={"help": "If True, prints a message to stdout for each update."})


@register_scheduler("reduce_lr_on_plateau", dataclass=ReduceLROnPlateauConfigs)
class ReduceLROnPlateauScheduler(ReduceLROnPlateau):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by
    a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen
    for a ‘patience’ number of epochs, the learning rate is reduced.

    Args:
        optimizer: Wrapped optimizer.
        configs: Configuration set.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        configs: DictConfig,
    ) -> None:
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, **configs.scheduler_parameters)


__all__ = ["ReduceLROnPlateauScheduler"]
