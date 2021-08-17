from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback


class BaseCallback(Callback):
    """Callback that perform some action on the input batch."""

    def __init__(self, apply_after_epoch: Optional[int] = None, stop_after_epoch: Optional[int] = None) -> None:
        """Create a new instance of BaseCallback."""
        super().__init__()
        apply_after_epoch = apply_after_epoch or -1
        stop_after_epoch = stop_after_epoch or -1

        if apply_after_epoch < -1:
            raise RuntimeError("`apply_after_epoch` is less than -1")
        if stop_after_epoch < -1:
            raise RuntimeError("`stop_after_epoch` is less than -1")
        if apply_after_epoch >= stop_after_epoch > -1:
            raise RuntimeError("`apply_after_epoch` is greater or equal to `stop_after_epoch`")
        self.apply_after_epoch = apply_after_epoch
        self.stop_after_epoch = stop_after_epoch
        self._disable = False

    def __repr__(self) -> str:
        return ";".join([f"{key}:{value}" for key, value in self._state.items()])

    @property
    def disable(self) -> bool:
        """Show that a callback is disabled or not."""
        return self._disable

    @property
    def _state(self) -> Dict:
        """Return current state."""
        return {"disable": self._disable}

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Disable any transforms for training data."""
        self._disable = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Enable any transforms for training data."""
        if trainer.current_epoch == (self.apply_after_epoch + 1):
            self._disable = False

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional = None) -> None:
        """Disable any transforms training data."""
        if trainer.current_epoch == self.stop_after_epoch:
            self._disable = True


class BaseMixCallback(BaseCallback):
    """Callback that perform Mix augmentation on the input batch."""

    def __init__(
        self,
        apply_after_epoch: Optional[int] = None,
        stop_after_epoch: Optional[int] = None,
        input_key: str = "features",
        target_key: str = "target",
    ) -> None:
        """Create a new instance of BaseMixCallback."""
        super().__init__(apply_after_epoch, stop_after_epoch)
        self.input_key = input_key
        self.target_key = target_key
        self._is_mixed = False

    @property
    def is_mixed(self) -> bool:
        """Show that a batch was mixed or not."""
        return self._is_mixed

    @property
    def _state(self) -> Dict:
        """Return current state."""
        return {"disable": self._disable, "is_mixed": self._is_mixed}


__all__ = ["BaseCallback", "BaseMixCallback"]
