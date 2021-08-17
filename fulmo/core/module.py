from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Type, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.core import LightningModule
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from ..callbacks.base import BaseCallback
from ..losses import CriterionMatcher, CriterionWrapper, OnlineLabelSmoothing
from ..metrics import MetricWrapper
from ..optimizers import Lookahead
from ..settings import Stage, logger
from .exceptions import PipelineCriterionBuildError, PipelineMetricBuildError, PipelineStepError


TOptimizer = Dict[str, Union[Optimizer, Dict[str, Union[int, str, _LRScheduler]]]]


class BaseModule(LightningModule):
    """nn.Module with additional great features."""

    def __init__(self, config: DictConfig) -> None:
        """Create a new instance of BaseModule.

        Args:
            config: Config describing a pipeline.
        """
        super().__init__()
        self.save_hyperparameters()
        OmegaConf.set_struct(config, False)
        self.config = config
        self.model: torch.nn.Module = self._build_model()
        self.criterion_wrapper: CriterionWrapper = self._build_criterion()
        self._train_metrics = self._build_metrics()
        self._test_metrics = self._build_metrics()
        self._val_metrics = self._build_metrics()
        self._keys_to_callbacks = self.config.get("input_keys_to_callback", None) or list()
        self._input_keys = self.config.get("input_keys")
        self._output_keys = self.config.get("output_keys")
        self._sync_dist = self.config.get("sync_dist") or False
        OmegaConf.set_struct(self.config, True)

    def _build_metrics(self) -> torch.nn.ModuleDict:
        """Build metrics.

        Returns:
            `:class:torch.nn.ModuleDict[str, pl_metrics]`

        Raises:
            PipelineMetricBuildError: if `target_key` or `output_key` does not exist
        """
        metrics: Dict[str, MetricWrapper] = {}
        for metric_name, metric_config in self.config["metrics"].items():
            copy_metric_config = deepcopy(metric_config)
            copy_metric_config.pop("wrapper_params")
            wrapper_params = metric_config.get("wrapper_params", None)
            if not wrapper_params:
                raise PipelineMetricBuildError(f"Wrapper parameters key for metric: {metric_name} does not exist")

            target_key = wrapper_params.get("target_key", None)
            output_key = wrapper_params.get("output_key", None)
            if not target_key:
                raise PipelineMetricBuildError(f"Target key for metric: {metric_name} does not exist")
            if not output_key:
                raise PipelineMetricBuildError(f"Output key for metric: {metric_name} does not exist")

            metrics[metric_name] = MetricWrapper(
                hydra.utils.instantiate(copy_metric_config, _convert_="partial"), **wrapper_params
            )
        return torch.nn.ModuleDict(modules=metrics)

    def _build_model(self) -> torch.nn.Module:
        """Build model from config.

        Returns:
            `:class:torch.nn.Module`
        """
        model = hydra.utils.instantiate(self.config.model)
        return model

    def _build_criterion(self) -> CriterionWrapper:
        """Build criterion from config.

        Returns:
            Wrapper on losses

        Raises:
            PipelineCriterionBuildError: if the number of losses is greater than 1 and
                the `strategy' is not a `default`. If `target_key` or `output_key` does not exist
        """
        criterion = {}
        weight = {}
        for loss_name, loss_config in self.config["losses"].items():
            copy_loss_config = deepcopy(loss_config)
            wrapper_params = copy_loss_config.pop("wrapper_params", None)
            matcher_params = copy_loss_config.pop("matcher_params", None)
            if not wrapper_params:
                raise PipelineCriterionBuildError(f"Wrapper parameters key for loss: {loss_name} does not exist")
            if not matcher_params:
                raise PipelineCriterionBuildError(f"Matcher parameters key for loss: {loss_name} does not exist")

            weight[loss_name] = wrapper_params.get("weight", 1.0)
            output_key = matcher_params.get("output_key", None)
            target_key = matcher_params.get("target_key", None)
            if not target_key:
                raise PipelineCriterionBuildError(f"Target key for loss: {loss_name} does not exist")
            elif not output_key:
                raise PipelineCriterionBuildError(f"Output key for loss: {loss_name} does not exist")
            criterion[loss_name] = CriterionMatcher(hydra.utils.instantiate(copy_loss_config), **matcher_params)

        criterion = torch.nn.ModuleDict(modules=criterion)
        criterion = CriterionWrapper(criterion, weight)
        return criterion

    def _compute_metrics(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stage: Stage
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics.

        Args:
            outputs: The output of your model
            batch: The output of your :class:torch.utils.data.DataLoader
            stage: stage name

        Returns:
            Evaluated metrics
        """
        metrics = {Stage.test: self._test_metrics, Stage.train: self._train_metrics, Stage.val: self._val_metrics}[
            stage
        ]
        output_dict = {f"{stage.value}/{key}": metric(outputs, batch) for key, metric in metrics.items()}
        return output_dict

    def _step(self, batch: Dict[str, torch.Tensor], batch_idx: int, stage: Stage) -> Dict[str, torch.Tensor]:
        """Compute losses and some additional metrics.

        Args:
            batch: The output of your :class:torch.utils.data.DataLoader
            batch_idx: Integer displaying index of this batch
            stage: Stage name. ("test", "train", "valid")

        Returns:
            the losses and some additional metrics

        Raises:
            PipelineStepError: if something went wrong.
        """
        outputs = self(batch)
        outputs = self._postprocess_outputs(outputs)
        losses = self.criterion_wrapper(outputs, batch, stage)
        try:
            log_dict = self._compute_metrics(outputs, batch, stage)
        except Exception as e:  # noqa: B902
            logger.exception(
                "Error caught in epoch %s",
                self.trainer.current_epoch,
                extra={"stage": stage.value, "step": "compute_metrics"},
            )
            raise PipelineStepError from e

        output_dict = dict()
        for key, value in outputs.items():
            batch[key] = value

        for key in self._keys_to_callbacks:
            output_dict[key] = batch[key]

        for key in self.criterion_wrapper.names:
            item = losses[key]
            output_dict[f"loss/{key}"] = item
            log_dict[f"{stage.value}/loss/{key}"] = item

        losses = torch.stack(list(losses.values()))
        loss_reduction = self.config.get("loss_reduction", None)
        if loss_reduction == "mean":
            loss = losses.mean()
        elif loss_reduction == "sum":
            loss = losses.sum()
        else:
            raise PipelineStepError(f"Incorrect loss reduction name: {loss_reduction}")

        output_dict["loss"] = loss
        log_dict[f"{stage.value}/loss"] = loss

        return {**log_dict, **output_dict}

    def _postprocess_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Post process model outputs."""
        return outputs

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run forward pass."""
        outputs = self.model(**{key: batch[key] for key in self._input_keys})
        if not isinstance(outputs, List):
            outputs = [outputs]
        return {key: outputs[index] for index, key in enumerate(self._output_keys)}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Compute and log training losses and some additional metrics.

        Args:
            batch: The output of your :class:torch.utils.data.DataLoader
            batch_idx: Integer displaying index of this batch

        Returns:
            the training losses and some additional metrics
        """
        stage = Stage.train
        output_dict = self._step(batch, batch_idx, stage)
        log_dict = {key: value for key, value in output_dict.items() if key.startswith(stage.value)}
        output_dict = {key: value for key, value in output_dict.items() if not key.startswith(stage.value)}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        return output_dict

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Compute validation losses and some additional metrics.

        Args:
            batch: The output of your :class:torch.utils.data.DataLoader
            batch_idx: Integer displaying index of this batch

        Returns:
            the validation losses and some additional metrics
        """
        stage = Stage.val
        output_dict = self._step(batch, batch_idx, stage)
        log_dict = {key: value for key, value in output_dict.items() if key.startswith(stage.value)}
        output_dict = {key: value for key, value in output_dict.items() if not key.startswith(stage.value)}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self._sync_dist)
        return output_dict

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Compute and log test losses and some additional metrics.

        Args:
            batch: The output of your :class:torch.utils.data.DataLoader
            batch_idx: Integer displaying index of this batch

        Returns:
            the test losses
        """
        stage = Stage.test
        output_dict = self._step(batch, batch_idx, stage)
        log_dict = {key: value for key, value in output_dict.items() if key.startswith(stage.value)}
        output_dict = {key: value for key, value in output_dict.items() if not key.startswith(stage.value)}
        self.log_dict(log_dict, on_step=False, on_epoch=True)
        return output_dict

    def on_train_epoch_end(self, unused: int = None) -> None:
        """Called in the training loop at the very end of the epoch."""
        for criterion in self.criterion_wrapper.criterion:
            if isinstance(criterion, OnlineLabelSmoothing):
                criterion.next_epoch()

        if self.config.get("debug", None):
            for callback in self.trainer.callbacks:
                if isinstance(callback, BaseCallback):
                    logger.info(
                        "Callback: %s, epoch: %s, current state: %s",
                        callback.__class__,
                        self.trainer.current_epoch,
                        repr(callback),
                    )

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Sequence[Optimizer], TOptimizer, Sequence[TOptimizer], None]:
        """Build optimizers and schedulers from config."""
        model_parameters = self.model.parameters()
        lookahead: Optional[Dict] = None
        if "optimizer_extension" in self.config:
            model_parameters = list()
            per_type: Optional[
                Dict[Type, Dict[str, Union[Type, Dict[str, float]]]]
            ] = self.config.optimizer_extension.get("per_type", None)

            for module_name, dict_per_module in self.config.optimizer_extension.per_module.items():
                params_per_module = dict_per_module["params"]
                module = getattr(self.model, module_name)
                if per_type:
                    for _, dict_per_type in per_type.items():
                        dtype: Type = dict_per_type["dtype"]
                        params_per_type: Dict[str, float] = dict_per_type["params"]
                        group_related_to_type = [
                            p for layer in module.children() for p in layer.parameters() if isinstance(layer, dtype)
                        ]
                        group_related_to_per_module = [
                            p for layer in module.children() for p in layer.parameters() if not isinstance(layer, dtype)
                        ]
                        if group_related_to_type:
                            model_parameters.append(dict(params=group_related_to_type, **params_per_type))
                        if group_related_to_per_module:
                            model_parameters.append(dict(params=group_related_to_per_module, **params_per_module))
                else:
                    model_parameters.append(dict(params=module.parameters(), **params_per_module))
            model_parameters = iter(model_parameters)
            lookahead = self.config.optimizer_extension.get("lookahead", None)

        optimizer: Optimizer = hydra.utils.instantiate(self.config.optimizer, params=model_parameters)
        if lookahead:
            optimizer = Lookahead(optimizer, **lookahead)

        if self.config.get("use_lr_scheduler", None):
            scheduler_class = self.config.scheduler["_target_"]
            if "ReduceLROnPlateau" in scheduler_class:
                scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer=optimizer)
                scheduler = {"scheduler": scheduler, "monitor": self.config.callback.model_checkpoint.monitor}
            elif "OneCycleLR" in scheduler_class:
                interval = self.config.scheduler.pop("interval", "epoch")
                scheduler = hydra.utils.instantiate(
                    self.config.scheduler, optimizer=optimizer, steps_per_epoch=len(self.datamodule.data_train)
                )
                scheduler = {
                    "scheduler": scheduler,
                    "interval": interval,
                    "frequency": 1,
                }
            else:
                scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer=optimizer)

            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer


__all__ = ["BaseModule"]
