from copy import deepcopy
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from .base import BaseCallback


class EmaCallback(BaseCallback):
    """Callback that perform EMA.

    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.
    """

    def __init__(
        self,
        ema_decay_per_epoch: float = 0.3,
        apply_on_epoch: Optional[int] = None,
        stop_after_epoch: Optional[int] = None,
        every_n_step: Optional[int] = 10,
        use_num_updates: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """Create a new instance of EmaCallback."""
        super().__init__(apply_on_epoch, stop_after_epoch)
        if device is not None and not isinstance(device, str):
            raise MisconfigurationException(f"device is expected to be a torch.device or a str. Found {device}")

        self._every_n_step = every_n_step
        self._device = device
        self._ema_decay_per_epoch = ema_decay_per_epoch
        self._model_contains_batch_norm: Optional[bool] = None
        self._num_updates: Optional[int] = 0 if use_num_updates else None
        self._decay: Optional[float] = None
        self._average_model: Optional[pl.LightningModule] = None
        self._accumulate_grad_batches: Optional[int] = None
        self.momenta: Dict[nn.Module, float] = {}

    @staticmethod
    def _pl_module_contains_batch_norm(pl_module: pl.LightningModule) -> bool:
        """Check if there are _BatchNorm layers."""
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in pl_module.modules())

    @staticmethod
    def _copy_to(src_pl_module: pl.LightningModule, dst_pl_module: pl.LightningModule) -> None:
        """Copy parameters from `src_pl_module` to `dst_pl_module`."""
        for src_param, dst_param in zip(src_pl_module.parameters(), dst_pl_module.parameters()):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def _reset_batch_norm_and_save_state(self, pl_module: pl.LightningModule) -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L154."""
        self.momenta = {}
        for module in pl_module.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            module.running_mean = torch.zeros_like(
                module.running_mean, device=pl_module.device, dtype=module.running_mean.dtype
            )
            module.running_var = torch.ones_like(
                module.running_var, device=pl_module.device, dtype=module.running_var.dtype
            )
            self.momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    def _reset_momenta(self) -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L164-L165."""
        for bn_module in self.momenta:
            bn_module.momentum = self.momenta[bn_module]

    def on_before_accelerator_backend_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Prepare values and a lightning-module."""
        # copy the model before moving it to accelerator device.
        super().on_before_accelerator_backend_setup(trainer, pl_module)
        self._decay = self._ema_decay_per_epoch ** (1 / len(trainer.datamodule.data_train))  # type: ignore[attr-defined]
        with pl_module._prevent_trainer_and_dataloaders_deepcopy():
            self._average_model = deepcopy(pl_module)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when fit begins."""
        optimizers = trainer.optimizers
        lr_schedulers = trainer.lr_schedulers

        if len(optimizers) != 1:
            raise MisconfigurationException("EMA currently works with 1 `optimizer`.")

        if len(lr_schedulers) > 1:
            raise MisconfigurationException("EMA currently not supported for more than 1 `lr_scheduler`.")

        self._model_contains_batch_norm = self._pl_module_contains_batch_norm(pl_module)
        if self._model_contains_batch_norm:
            # virtually increase max_epochs to perform batch norm update on latest epoch.
            trainer.fit_loop.max_epochs += 1

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the train epoch begins."""
        super().on_train_epoch_start(trainer, pl_module)
        if trainer.current_epoch == self.apply_on_epoch:
            # move average model to request device.
            self._average_model = self._average_model.to(self._device or pl_module.device)

        if self.apply_on_epoch <= trainer.current_epoch <= self.stop_after_epoch:
            gs = trainer.global_step
            if gs % self._every_n_step == 0:
                self.update_parameters(self._average_model, pl_module)

        # Note: No > here in case the callback is saved with the model and training continues
        if trainer.current_epoch == self.stop_after_epoch + 1:
            # Transfer weights from average model to pl_module
            self._copy_to(self._average_model, pl_module)
            # Reset BatchNorm for update
            self._reset_batch_norm_and_save_state(pl_module)

            # There is no need to perform either backward or optimizer.step as we are
            # performing only one pass over the train data-loader to compute activation statistics
            # Therefore, we will virtually increase `num_training_batches` by 1 and skip backward.
            trainer.num_training_batches += 1
            trainer.fit_loop._skip_backward = True
            self._accumulate_grad_batches = trainer.accumulate_grad_batches
            trainer.accumulate_grad_batches = trainer.num_training_batches

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the train epoch ends."""
        super().on_train_epoch_end(trainer, pl_module)
        trainer.fit_loop._skip_backward = False

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the train ends."""
        if self._model_contains_batch_norm and trainer.current_epoch == self.stop_after_epoch + 1:
            # BatchNorm epoch update. Reset state
            trainer.accumulate_grad_batches = self._accumulate_grad_batches
            trainer.num_training_batches -= 1
            trainer.fit_loop.max_epochs -= 1
            self._reset_momenta()
        elif trainer.current_epoch == self.stop_after_epoch:
            # Last EMA epoch. Transfer weights from average model to pl_module
            self._copy_to(self._average_model, pl_module)

    def update_parameters(self, average_model: pl.LightningModule, model: pl.LightningModule) -> None:
        """Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of the `optimizer.step()` call.

        Args:
            average_model: Usually shadow parameters.
            model: Usually the same set of parameters used to initialize this object.
        """
        decay = self._decay
        if self._num_updates is not None:
            self._num_updates += 1
            decay = min(decay, (1 + self._num_updates) / (10 + self._num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(average_model.parameters(), model.parameters()):
                param_ = param.to(s_param.device)
                s_param.sub_(one_minus_decay * (s_param - param_))


__all__ = ["EmaCallback"]
