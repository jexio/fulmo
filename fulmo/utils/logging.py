import logging
import warnings

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Initializes python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """Control flow by main config file.

    Args:
        config (DictConfig): [description]
    """
    log = get_logger(__name__)

    # make it possible to add new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info("Forcing ddp friendly configuration! <config.trainer.accelerator=%s>", accelerator)
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

        # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
) -> None:
    """Log hyperparameters."""
    hparams = {}
    hparams.update(config["trainer"])
    hparams.update(config["datamodule"])
    hparams.update(config["model"])
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "strategy" in config:
        hparams["strategy"] = config["strategy"]
    if "lr_per_module" in config:
        hparams["lr_per_module"] = config["lr_per_module"]
    if "optimizer" in config:
        hparams["optimizer"] = config["optimizer"]
    if "scheduler" in config and config.get("use_lr_scheduler", False):
        hparams["scheduler"] = config["scheduler"]
    if "sampler" in config:
        hparams["sampler"] = config["sampler"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    if hasattr(datamodule, "data_train") and datamodule.data_train is not None:
        hparams["train_size"] = len(datamodule.data_train)
    if hasattr(datamodule, "data_val") and datamodule.data_val is not None:
        hparams["val_size"] = len(datamodule.data_val)
    if hasattr(datamodule, "data_test") and datamodule.data_test is not None:
        hparams["test_size"] = len(datamodule.data_test)

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda x: None


__all__ = ["get_logger", "extras", "log_hyperparameters"]
