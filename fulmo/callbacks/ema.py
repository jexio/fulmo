from typing import Optional

import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from ..models.layers.ema import ModelEma
from .base import BaseCallback


class EmaCallback(BaseCallback):
    """Callback that perform EMA."""

    def __init__(self, ema_decay_per_epoch: float = 0.3, every_n_step: int = 10, apply_after_step: int = 1000) -> None:
        """Create a new instance of EmaCallback."""
        super().__init__(None, None)
        self.ema_decay_per_epoch = ema_decay_per_epoch
        self.every_n_step = every_n_step
        self.apply_after_step = apply_after_step
        self.ema: Optional[ModelEma] = None
        self._is_updated = False
        self._decay: Optional[float] = None

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute decay."""
        super().on_before_accelerator_backend_setup(trainer, pl_module)
        self._decay = self.ema_decay_per_epoch ** (1 / len(trainer.datamodule.data_train))  # type: ignore[attr-defined]

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create a new instance of `ModelEma`."""
        super(EmaCallback, self).on_train_start(trainer, pl_module)
        self.ema = ModelEma(pl_module.model.parameters(), self._decay)  # type: ignore[arg-type]

    def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer) -> None:
        """Update currently maintained parameters."""
        gs = trainer.global_step
        if gs % self.every_n_step != 0 or gs <= self.apply_after_step:
            return
        self._is_updated = True
        self.ema.update(pl_module.model.parameters())  # type: ignore[union-attr]

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save current model and validate on ema model."""
        gs = trainer.global_step
        if gs > self.apply_after_step and self._is_updated:
            self.ema.store(pl_module.model.parameters())  # type: ignore[union-attr]
            self.ema.copy_to(pl_module.model.parameters())  # type: ignore[union-attr]

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore saved model."""
        if self._is_updated:
            self.ema.restore(pl_module.model.parameters())  # type: ignore[union-attr]


__all__ = ["EmaCallback"]
