import pytorch_lightning as pl
import torch


def set_seed(seed: int) -> None:
    """Set the seed value.

    Args:
        seed: seed value
    """
    pl.seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


__all__ = ["set_seed"]
