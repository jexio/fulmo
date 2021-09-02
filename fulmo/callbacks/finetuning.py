from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.optim import Optimizer


class FreezeUnfreezeBackboneCallback(BackboneFinetuning):
    """

    Finetune a backbone model based on a learning rate user-defined scheduling.
    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:

        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.

        num_layers: Number of non-freeze layers.

        lambda_func: Scheduling function for increasing backbone learning rate.

        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model

        backbone_initial_lr: Optional, Inital learning rate for the backbone.
            By default, we will use current_learning /  backbone_initial_ratio_lr

        should_align: Wheter to align with current learning rate when backbone learning
            reaches it.

        initial_denom_lr: When unfreezing the backbone, the intial learning rate will
            current_learning_rate /  initial_denom_lr.

        train_bn: Wheter to make Batch Normalization trainable.

        verbose: Display current learning rate for model and backbone

        precision: Precision for displaying learning rate

    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        num_layers: int = -1,
        lambda_func: Callable[[int], int] = lambda x: 2,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        precision: int = 12,
    ) -> None:
        """Create a new instance of `FreezeUnfreezeBackboneCallback`."""
        super().__init__(
            unfreeze_backbone_at_epoch,
            lambda_func,
            backbone_initial_ratio_lr,
            backbone_initial_lr,
            should_align,
            initial_denom_lr,
            train_bn,
            verbose,
            precision,
        )
        self.num_layers = num_layers

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore parameters if needs.

        Args:
            trainer: `pytorch-lightning` trainer
            pl_module: `pytorch-lightning` module

        Raises:
            MisconfigurationException: If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module.model, "backbone") and isinstance(pl_module.model.backbone, torch.nn.Module):
            # restore the param_groups created during the previous training.
            if self._restarting:
                named_parameters = dict(pl_module.named_parameters())
                for opt_idx, optimizer in enumerate(trainer.optimizers):  # type: ignore[arg-type]
                    param_groups = self.__apply_mapping_to_param_groups(
                        self._internal_optimizer_metadata[opt_idx], named_parameters
                    )
                    optimizer.param_groups = param_groups
                self._restarting = False
        else:
            raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        """Freeze modules."""
        modules = pl_module.model.backbone
        if self.num_layers != -1:
            modules = list(modules)[: -self.num_layers]
        self.freeze(modules)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        """Called when the epoch begins."""
        modules = pl_module.model.backbone
        if self.num_layers != -1:
            modules = list(modules)[: -self.num_layers]

        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                modules,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr


__all__ = ["FreezeUnfreezeBackboneCallback"]
