from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]


class Lookahead(Optimizer):
    """Implements Lookahead optimization algorithm."""

    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        """It has been proposed in `Lookahead Optimizer: k steps forward, 1 step back`.

        Arguments:
            optimizer: base inner optimizers optimize, like Yogi, DiffGrad or Adam.
            k: number of lookahead steps. Default: 5
            alpha: linear interpolation factor. 1.0 recovers the inner optimizers. Default: 5

        Raises:
            ValueError: if `k` or `alpha` are negative numbers.
        """
        if k < 0.0:
            raise ValueError("Invalid number of lookahead steps: {}".format(k))
        if alpha < 0:
            raise ValueError("Invalid linear interpolation factor: {}".format(alpha))

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state: Dict[Any, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def __repr__(self) -> str:
        base_str = self.optimizer.__repr__()
        format_string = self.__class__.__name__ + " ("
        format_string += "\n"
        format_string += "k: {}\n".format(self.k)
        format_string += "alpha: {}\n".format(self.alpha)
        format_string += base_str
        format_string += "\n"
        format_string += ")"
        return format_string

    @property
    def defaults(self) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Get a dict containing default values of optimization."""
        return self.optimizer.defaults

    def _update(self, group: Dict[str, Any]) -> None:
        """Update a group."""
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.clone(fast.data).detach()

            slow = param_state["slow_param"]
            fast.data.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.data.copy_(fast)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the models

        Returns:
            computed losses
        """
        loss = self.optimizer.step(closure=closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self._update(group)
            group["counter"] = (group["counter"] + 1) % self.k
        return loss

    def state_dict(self) -> State:
        """Return the state of the optimizers as a :class:`dict`.

        It contains two entries:
        * state - a dict holding current optimization state. Its content
            differs between optimizers classes.
        * param_groups - a dict containing all parameter groups

        Returns:
            the state of the optimizers
        """
        slow_state_dict = super(Lookahead, self).state_dict()
        fast_state_dict = self.optimizer.state_dict()
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state_dict["state"],
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict: State) -> None:
        """Load the optimizers state.

        Arguments:
            state_dict: optimizers state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def zero_grad(self) -> None:  # type: ignore[override]
        """Clear the gradients of all optimized :class:`torch.Tensor`."""
        self.optimizer.zero_grad()


__all__ = ["Lookahead"]
