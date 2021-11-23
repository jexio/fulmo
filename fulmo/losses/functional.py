from typing import Dict

import torch
import torch.nn as nn

from ..settings import DEFAULT_SETTINGS


def mix_criterion(
    criterion: nn.Module, outputs: torch.Tensor, batch: Dict[str, torch.Tensor], input_target_key: str
) -> torch.Tensor:
    """Compute loss."""
    lam_a = batch[DEFAULT_SETTINGS.mix_lam_a_key][0]
    lam_b = batch[DEFAULT_SETTINGS.mix_lam_b_key][0]
    target = batch[input_target_key]
    second_target = batch[DEFAULT_SETTINGS.mix_target_key]
    return lam_a * criterion(outputs, target) + lam_b * criterion(outputs, second_target)


__all__ = ["mix_criterion"]
