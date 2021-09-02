import random
from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from ..settings import DEFAULT_SETTINGS
from .base import BaseMixCallback
from .functional import cutmix_bbox_and_lam


class CutMixCallback(BaseMixCallback):
    """Callback that perform CutMixCallback augmentation on the input batch."""

    def __init__(
        self,
        apply_on_epoch: Optional[int] = None,
        stop_after_epoch: Optional[int] = None,
        alpha: float = 1.0,
        probability: float = 1.0,
        input_key: str = "features",
        target_key: str = "target",
    ) -> None:
        """Create a new instance of CutMixCallback."""
        super().__init__(apply_on_epoch, stop_after_epoch, input_key, target_key)
        self.alpha = alpha
        self.probability = probability

    def _get_bbox(self, batch: Dict[str, torch.Tensor], lam: float) -> Tuple[int, int, int, int, float]:
        """Generate bbox and apply lambda correction."""
        features = batch[self.input_key]
        shape = features.shape[2:]
        x1, y1, x2, y2, lam = cutmix_bbox_and_lam(shape, lam)
        return x1, y1, x2, y2, lam

    def _do_cutmix(self, batch: Dict[str, torch.Tensor], lam: float) -> None:
        """Performs cutmix for classification task."""
        features, target = batch[self.input_key], batch[self.target_key]
        batch_size = features.size()[0]
        index = torch.randperm(batch_size).to(target.device)
        x1, y1, x2, y2, lam = self._get_bbox(batch, lam)
        features[:, :, x1:x2, y1:y2] = features[index, :, x1:x2, y1:y2]
        y_a, y_b = target, target[index]
        lam = torch.tensor([lam]).to(features.device).squeeze(0)
        batch[self.input_key] = features
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
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        self._is_mixed = False
        if not self._disable and self.probability > random.random():  # noqa: S311
            self._is_mixed = True
            self._do_cutmix(batch, lam)


class SegmentationCutMixCallback(CutMixCallback):
    """Callback that perform CutMixCallback augmentation on the input batch."""

    def __init__(
        self,
        apply_on_epoch: Optional[int] = None,
        stop_after_epoch: Optional[int] = None,
        alpha: float = 1.0,
        probability: float = 1.0,
        input_key: str = "features",
        target_key: str = "target",
    ) -> None:
        """Create a new instance of CutMixCallback."""
        super().__init__(apply_on_epoch, stop_after_epoch, alpha, probability, input_key, target_key)

    def _do_cutmix(self, batch: Dict[str, torch.Tensor], lam: float) -> None:
        """Performs cutmix for segmentation task."""
        features, target = batch[self.input_key], batch[self.target_key]
        batch_size = features.size()[0]
        index = torch.randperm(batch_size).to(target.device)
        x1, y1, x2, y2, lam = self._get_bbox(batch, lam)
        features[:, :, x1:x2, y1:y2] = features[index, :, x1:x2, y1:y2]
        target[:, :, x1:x2, y1:y2] = target[index, :, x1:x2, y1:y2]
        batch[self.input_key] = features
        batch[self.target_key] = target

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch begins."""  # noqa: D401
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        self._is_mixed = False
        if not self._disable and self.probability > random.random():  # noqa: S311
            self._do_cutmix(batch, lam)


__all__ = ["CutMixCallback", "SegmentationCutMixCallback"]
