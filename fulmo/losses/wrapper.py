from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from ..settings import DEFAULT_SETTINGS, Stage
from .functional import mix_criterion


_criterion_strategy: Dict[str, Callable[[nn.Module, torch.Tensor, Dict[str, torch.Tensor], str], torch.Tensor]] = {
    "mix": mix_criterion,
}


class CriterionMatcher(nn.Module):  # type: ignore[misc]
    """Match your outputs with your target here."""

    def __init__(
        self,
        criterion: nn.Module,
        output_key: str = "logits",
        target_key: str = "target",
        mix_strategy: Optional[str] = None,
    ) -> None:
        """Create a new instance of CriterionMatcher."""
        super().__init__()
        self.criterion = criterion
        self.output_key = output_key
        self.target_key = target_key
        self.mix_strategy = mix_strategy
        if mix_strategy and mix_strategy not in _criterion_strategy:
            raise KeyError(f"strategy - {mix_strategy} does not support.")

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stage: Stage) -> torch.Tensor:
        """Run forward pass."""
        if stage == Stage.val:
            return self.criterion(outputs[self.output_key], batch[self.target_key])
        if self.mix_strategy:
            if DEFAULT_SETTINGS.mix_target_key in batch:
                return _criterion_strategy[self.mix_strategy](
                    self.criterion, outputs[self.output_key], batch, self.target_key
                )
        return self.criterion(outputs[self.output_key], batch[self.target_key])


class CriterionWrapper(nn.Module):  # type: ignore[misc]
    """Wrap all of your criteria."""

    def __init__(
        self, criterion: nn.ModuleDict, reduction: Callable[[torch.Tensor], torch.Tensor], weight: Dict[str, float]
    ) -> None:
        """Create a new instance of CriterionWrapper."""
        super().__init__()
        self._criterion = criterion
        self._reduction = reduction
        self._weight = weight

    def __len__(self) -> int:
        """Get a number of criteria."""
        return len(self._criterion)

    @property
    def names(self) -> Iterator[str]:
        """Get a list of criterion names."""
        yield from self._weight.keys()

    @property
    def criterion(self) -> Iterator[nn.Module]:
        """Iterate over criteria."""
        for _, value in self._criterion.items():
            yield value

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stage: Stage = Stage.train
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run forward pass."""
        losses = {}
        for key, value in self._criterion.items():
            loss = value(outputs, batch, stage) * self._weight[key]
            losses[key] = loss
        return losses, self._reduction(torch.stack(list(losses.values())))


__all__ = ["CriterionMatcher", "CriterionWrapper"]
