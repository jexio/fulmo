from typing import Dict, Optional

import torch
import torch.nn as nn


class MetricWrapper(nn.Module):  # type: ignore[misc]
    """Wrap all of your metrics."""

    def __init__(
        self,
        metric: nn.Module,
        output_key: str = "logits",
        target_key: str = "target",
        activation: Optional[str] = "softmax",
        dim: Optional[int] = -1,
        dtype: Optional[torch.Type] = None,
    ) -> None:
        """Create a new instance of MetricWrapper."""
        super().__init__()
        self.metric = metric
        self.output_key = output_key
        self.target_key = target_key
        self.dtype = dtype
        if activation is None:
            self.activation_fn = lambda x: x
        elif activation == "softmax":
            self.activation_fn = lambda x: torch.softmax(x, dim=dim)
        elif activation == "sigmoid":
            self.activation_fn = lambda x: torch.sigmoid(x)
        elif activation == "argmax":
            self.activation_fn = lambda x: torch.argmax(x, dim=dim)
        else:
            raise NotImplementedError(f"Activation - {activation} not implemented")

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass."""
        outputs_ = self.activation_fn(outputs[self.output_key])
        target = batch[self.target_key]
        if self.dtype is not None:
            target = target.type(self.dtype)

        metric: torch.Tensor = self.metric(outputs_, target)
        return metric


__all__ = ["MetricWrapper"]
