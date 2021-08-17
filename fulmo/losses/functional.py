from typing import Dict

import torch
import torch.nn as nn

from ..settings import DEFAULT_SETTINGS


def mix_criterion(
    criterion: nn.Module, outputs: torch.Tensor, batch: Dict[str, torch.Tensor], input_target_key: str
) -> torch.Tensor:
    """Compute loss."""
    lam = batch[DEFAULT_SETTINGS.mix_lam_key]
    target = batch[input_target_key]
    second_target = batch[DEFAULT_SETTINGS.mix_target_key]
    return lam * criterion(outputs, target) + (1 - lam) * criterion(outputs, second_target)


__all__ = ["mix_criterion"]
