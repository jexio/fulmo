from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

from ..dataclass.configurations import LearningRateSchedulerConfigs
from . import register_scheduler


@dataclass
class OneCycleLRConfigs(LearningRateSchedulerConfigs):
    """OneCycle learning rate scheduler config."""

    name: str = field(default="one_cycle_lr", metadata={"help": "Name of learning rate scheduler."})
    max_lr: List[float] = field(
        default_factory=lambda: [10.0],
        metadata={"help": "Upper learning rate boundaries in the cycle for each parameter group"},
    )
    total_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": " The total number of steps in the cycle. "
            "Note that if a value is not provided here, "
            "then it must be inferred by providing a value for epochs and steps_per_epoch."
        },
    )
    epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": " The number of epochs to train for. This is used along with steps_per_epoch in order to infer "
            "the total number of steps in the cycle if a value for total_steps is not provided."
        },
    )
    steps_per_epoch: Optional[float] = field(
        default=None,
        metadata={
            "help": "The number of steps per epoch to train for. "
            "This is used along with epochs in order to infer the total number of steps in the cycle "
            "if a value for total_steps is not provided."
        },
    )
    pct_start: float = field(
        default=0.3,
        metadata={"help": "The percentage of the cycle (in number of steps) spent increasing the learning rate."},
    )
    anneal_strategy: str = field(
        default="cos",
        metadata={
            "help": "Specifies the annealing strategy: “cos” for cosine annealing, " "“linear” for linear annealing."
        },
    )
    cycle_momentum: bool = field(
        default=True,
        metadata={
            "help": "If True, momentum is cycled inversely to learning rate between "
            "‘base_momentum’ and ‘max_momentum’."
        },
    )
    base_momentum: List[float] = field(
        default_factory=lambda: [0.85],
        metadata={
            "help": " Lower momentum boundaries in the cycle for each parameter group. "
            "Note that momentum is cycled inversely to learning rate; at the peak of a cycle, "
            "momentum is ‘base_momentum’ and learning rate is ‘max_lr’."
        },
    )
    max_momentum: float = field(
        default=0.95,
        metadata={
            "help": "Upper momentum boundaries in the cycle for each parameter group. "
            "Functionally, it defines the cycle amplitude (max_momentum - base_momentum). "
            "Note that momentum is cycled inversely to learning rate; at the start of a cycle, "
            "momentum is ‘max_momentum’ and learning rate is ‘base_lr’."
        },
    )
    div_factor: float = field(
        default=25.0, metadata={"help": "Determines the initial learning rate via initial_lr = max_lr/div_factor."}
    )
    final_div_factor: float = field(
        default=1.0e-4,
        metadata={"help": "Determines the minimum learning rate via min_lr = initial_lr/final_div_factor."},
    )
    three_phase: bool = field(
        default=False,
        metadata={
            "help": "If True, use a third phase of the schedule to annihilate "
            "the learning rate according to ‘final_div_factor’ instead of modifying the second phase "
            "(the first two phases will be symmetrical about the step indicated by ‘pct_start’."
        },
    )
    last_epoch: int = field(
        default=-1,
        metadata={
            "help": "The index of the last batch. This parameter is used when resuming a training job. "
            "Since step() should be invoked after each batch instead of after each epoch, "
            "this number represents the total number of batches computed, "
            "not the total number of epochs computed. When last_epoch=-1, "
            "the schedule is started from the beginning."
        },
    )
    verbose: bool = field(default=False, metadata={"help": "If True, prints a message to stdout for each update."})


@register_scheduler("one_cycle_lr", dataclass=OneCycleLRConfigs)
class OneCycleLRScheduler(OneCycleLR):
    """Sets the learning rate of each parameter group according to the 1cycle learning rate policy.

    The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate
    and then from that maximum learning rate to some minimum learning rate much lower than the initial learning rate.
    This policy was initially described in the paper Super-Convergence:
            Very Fast Training of Neural Networks Using Large Learning Rates.
    The 1cycle learning rate policy changes the learning rate after every batch.

    Step should be called after a batch has been used for training. This scheduler is not chainable.
    Note also that the total number of steps in the cycle can be determined in one of two ways:
    A value for total_steps is explicitly provided.
    A number of epochs (epochs) and a number of steps per epoch (steps_per_epoch) are provided.
    In this case, the number of total steps is inferred by total_steps = epochs * steps_per_epoch
    You must either provide a value for total_steps or provide a value for both epochs and steps_per_epoch.

    Attributes:
        optimizer: Wrapped optimizer.
        configs: Configuration set.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        configs: DictConfig,
    ) -> None:
        """Create a new instance of OneCycleLRScheduler."""
        super(OneCycleLR, self).__init__(optimizer, **configs.scheduler_parameters)


__all__ = ["OneCycleLRScheduler"]
