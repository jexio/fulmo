from typing import Iterator, List, Union

import numpy as np
from torch.utils.data import Sampler


class BalanceClassSampler(Sampler):
    """Abstraction over loader samplers.

    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(self, labels: List[int], mode: Union[str, int] = "downsampling") -> None:
        """Create instance of BalanceClassSampler.

        Args:
            labels: list of class label
                for each elem in the datasets
            mode: Strategy to balance classes.
                Must be one of [downsampling, upsampling]

        Raises:
            ValueError: if `mode` not in ("downsampling", "upsampling")
        """
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {label: (labels == label).sum() for label in set(labels)}
        self.lbl2idx = {label: np.arange(len(labels))[labels == label].tolist() for label in set(labels)}

        if isinstance(mode, str):
            if mode not in ("downsampling", "upsampling"):
                raise ValueError(f"Mode - {mode} does not support")

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = mode if isinstance(mode, int) else max(samples_per_class.values())
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class: int = samples_per_class
        self.length: int = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """Get element of the datasets.

        Returns:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(self.lbl2idx[key], self.samples_per_class, replace=replace_).tolist()
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Get length of result sample.

        Returns:
             length of result sample
        """
        return self.length


__all__ = ["BalanceClassSampler"]
