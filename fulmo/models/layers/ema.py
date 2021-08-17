from typing import Iterable

import torch


class ModelEma:
    """Maintains (exponential) moving average of a set of parameters."""

    def __init__(self, parameters: torch.nn.Parameter, decay: float, use_num_updates: bool = True) -> None:
        """Create a new instance of ModelEma.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing averages.
        """
        assert 0.0 < decay < 1.0, f"If decay is specified it must be in (0; 1), " f"got {decay}"
        self.collected_parameters = [torch.nn.Parameter]
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters]

    def update(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Usually the same set of parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Copy current parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Save the current parameters for restore.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be temporary stored in.
        """
        self.collected_parameters = [param.clone() for param in parameters if param.requires_grad]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore the parameters from the `store` function.

        Usually used in validation. Store the parameters before the `copy_to` function.
        After the validation(or model saving), restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)


__all__ = ["ModelEma"]
