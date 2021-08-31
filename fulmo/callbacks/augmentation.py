# type: ignore[misc, union-attr]
from typing import Optional

import pytorch_lightning as pl

from ..models import BaseModel
from .base import BaseCallback


class AugmentationOnOffCallback(BaseCallback):
    """Enable/disable augmentations for a training dataset."""

    def __init__(self, apply_on_epoch: Optional[int] = None, stop_after_epoch: Optional[int] = None) -> None:
        """Create a new instance of AugmentationOnOffCallback."""
        super().__init__(apply_on_epoch, stop_after_epoch)
        self._disable = True

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Disable augmentations for training data."""
        pl_module.train_dataloader().dataset.apply_transforms = not self._disable
        pl_module.val_dataloader().dataset.apply_transforms = not self._disable
        if isinstance(pl_module.trainer.model, BaseModel):
            pl_module.trainer.model.apply_transforms = not self._disable

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Enable augmentations for training data."""
        super().on_train_epoch_start(trainer, pl_module)
        if not self._disable:
            pl_module.train_dataloader().dataset.apply_transforms = True
            pl_module.val_dataloader().dataset.apply_transforms = True
            if isinstance(pl_module.trainer.model, BaseModel):
                pl_module.trainer.model.apply_transforms = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional = None) -> None:
        """Disable augmentations for training data."""
        super().on_train_epoch_end(trainer, pl_module, unused)
        if self._disable:
            pl_module.train_dataloader().dataset.apply_transforms = False
            pl_module.val_dataloader().dataset.apply_transforms = False
            if isinstance(pl_module.trainer.model, BaseModel):
                pl_module.trainer.model.apply_transforms = False


__all__ = ["AugmentationOnOffCallback"]
