import random
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch

from ..settings import DEFAULT_SETTINGS
from .base import BaseMixCallback


class MixUpCallback(BaseMixCallback):
    """Callback that perform MixUp augmentation on the input batch."""

    def __init__(
        self,
        apply_on_epoch: Optional[int] = None,
        stop_after_epoch: Optional[int] = None,
        alpha: float = 1.0,
        probability: float = 1.0,
        input_key: str = "features",
        target_key: str = "target",
    ) -> None:
        """Create a new instance of MixUpCallback."""
        super(MixUpCallback, self).__init__(apply_on_epoch, stop_after_epoch, input_key, target_key)
        self.alpha = alpha
        self.probability = probability

    def _do_mixup(self, batch: Dict[str, torch.Tensor]) -> None:
        """Performs mixup."""
        self._is_mixed = True
        lam = 1.0
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        features, target = batch[self.input_key], batch[self.target_key]
        batch_size = features.size()[0]
        index = torch.randperm(batch_size).to(target.device)

        mixed_features = lam * features + (1 - lam) * features[index, :]
        lam = torch.tensor([lam]).to(features.device).squeeze(0)
        y_a, y_b = target, target[index]
        batch[self.input_key] = mixed_features
        batch[self.target_key] = y_a
        batch[DEFAULT_SETTINGS.mix_target_key] = y_b
        batch[DEFAULT_SETTINGS.mix_lam_key] = lam

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch begins."""  # noqa: D401
        self._is_mixed = False
        if not self._disable and self.probability > random.random():  # noqa: S311
            self._do_mixup(batch)


class MixUpWHCallback(MixUpCallback):
    """Callback that perform Mixup Without Hesitation https://arxiv.org/abs/2101.04342 on the input batch."""

    def __init__(
        self,
        apply_on_epoch: Optional[int] = None,
        stop_after_epoch: Optional[int] = None,
        alpha: float = 1.0,
        probability: float = 1.0,
        input_key: str = "features",
        target_key: str = "target",
    ) -> None:
        """Create a new instance of MixUpWHCallback."""
        super(MixUpWHCallback, self).__init__(
            apply_on_epoch, stop_after_epoch, alpha, probability, input_key, target_key
        )
        self._num_batches = 0
        self._current_step = 0

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Prepare values."""
        super().on_before_accelerator_backend_setup(trainer, pl_module)
        self._num_batches = len(pl_module.train_dataloader())

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch begins."""  # noqa: D401
        q = np.random.uniform(0, 1)
        p = np.random.uniform(0, q)

        self._is_mixed = False
        if not self._disable:
            if self._current_step <= p * self._num_batches:
                self._do_mixup(batch)
            elif self._current_step <= q * self._num_batches:
                if self._current_step % 2 == 0:
                    self._do_mixup(batch)
            else:
                eps = (self._num_batches - self._current_step) / self._num_batches * (1 - q)
                threshold = np.random.random()
                if threshold < eps:
                    self._do_mixup(batch)
        self._current_step += 1

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional[int] = None
    ) -> None:
        """Reset step count."""
        super().on_train_epoch_end(trainer, pl_module, unused)
        self._current_step = 0


__all__ = ["MixUpCallback", "MixUpWHCallback"]
