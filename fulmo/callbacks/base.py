from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback


class BaseCallback(Callback):
    """Callback that perform some action on the input batch.

    Attributes:
        apply_on_epoch: Enable transforms on an epoch equal to `apply_on_epoch`
        stop_after_epoch: Disable transforms after an epoch equal to `stop_after_epoch`.
                          If `stop_after_epoch` is equal to -1 then transforms will not be disabled
    Raises:
        ValueError: if `apply_on_epoch` is less than 0 or `stop_after_epoch` is less than -1
    """

    def __init__(self, apply_on_epoch: Optional[int] = None, stop_after_epoch: Optional[int] = None) -> None:
        """Create a new instance of BaseCallback."""
        super().__init__()
        apply_on_epoch = apply_on_epoch or 0
        stop_after_epoch = stop_after_epoch or -1

        if apply_on_epoch < 0:
            raise ValueError("`apply_on_epoch` is less than 0")
        if stop_after_epoch < -1:
            raise ValueError("`stop_after_epoch` is less than -1")
        if apply_on_epoch > stop_after_epoch > -1:
            raise ValueError("`apply_on_epoch` is greater to `stop_after_epoch`")
        self.apply_on_epoch = apply_on_epoch
        self.stop_after_epoch = stop_after_epoch
        self._disable = False

    def __repr__(self) -> str:
        return "; ".join([f"{key}: {value}" for key, value in self._state.items()])

    @property
    def disable(self) -> bool:
        """Show that a callback is disabled or not."""
        return self._disable

    @property
    def _state(self) -> Dict[str, Any]:
        """Return current state."""
        return {"disable": self._disable}

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Disable any transforms for training data."""
        self._disable = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Enable any transforms for training data."""
        if trainer.current_epoch == self.apply_on_epoch:
            self._disable = False

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional[int] = None
    ) -> None:
        """Disable any transforms training data."""
        if trainer.current_epoch == self.stop_after_epoch:
            self._disable = True


class BaseMixCallback(BaseCallback):
    """Callback that perform Mix augmentation on the input batch."""

    def __init__(
        self,
        apply_on_epoch: Optional[int] = None,
        stop_after_epoch: Optional[int] = None,
        input_key: str = "features",
        target_key: str = "target",
    ) -> None:
        """Create a new instance of BaseMixCallback."""
        super().__init__(apply_on_epoch, stop_after_epoch)
        self.input_key = input_key
        self.target_key = target_key
        self._is_mixed = False

    @property
    def is_mixed(self) -> bool:
        """Show that a batch was mixed or not."""
        return self._is_mixed

    @property
    def _state(self) -> Dict[str, Any]:
        """Return current state."""
        return {"disable": self._disable, "is_mixed": self._is_mixed}


__all__ = ["BaseCallback", "BaseMixCallback"]
