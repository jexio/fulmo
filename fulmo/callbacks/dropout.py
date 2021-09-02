from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from .base import BaseCallback


class ScheduledDropoutCallback(BaseCallback):
    """Slowly changes dropout value for `attr_name` each epoch.

    Ref: https://arxiv.org/abs/1703.06229

    Attributes:
        epochs: num epochs to max dropout to fully take effect
        attr_name: name of dropout block in model
    """

    def __init__(self, epochs: int = 10, attr_name: str = "dropout.p") -> None:
        """Create a new instance of `ScheduledDropoutCallback`"""
        super().__init__()
        self.epochs = epochs
        self.attr_name = attr_name
        self._dropout_rates: Dict[torch.nn.Module, float] = dict()

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize dictionary contains initial probabilities."""
        module = getattr(pl_module.model, self.attr_name)
        if isinstance(module, torch.nn.Sequential):
            for layer in module.children():
                if isinstance(layer, torch.nn.Dropout):
                    self._dropout_rates[layer] = layer.p

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional[int] = None
    ) -> None:
        """Changes dropout value."""
        module = getattr(pl_module.model, self.attr_name)
        if isinstance(module, torch.nn.Sequential):
            for layer in module.children():
                if isinstance(layer, torch.nn.Dropout):
                    current_rate = self._dropout_rates[layer] * min(1, pl_module.current_epoch / self.epochs)
                    layer.p = current_rate


__all__ = ["ScheduledDropoutCallback"]
