from typing import Any, Callable, Dict, Optional, Sequence

import pandas as pd
from torch.utils.data import Dataset

from ..readers.base import Augmentor


class MultiDomainCsvDataset(Dataset):
    """Dataset abstraction for the multi-domain case."""

    def __init__(
        self,
        data: pd.DataFrame,
        open_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        transforms: Optional[Dict[str, Augmentor]] = None,
        preprocessing: Optional[Dict[str, Augmentor]] = None,
        target_key: Optional[str] = "target",
    ) -> None:
        """Create a new instance of MultiDomainCsvDataset.

        Args:
            data: Dataframe with at least the following two columns - "path to data", "target"
            open_fn: Function, that can open your annotations dict and transfer it to data, needed by your network
            transforms: Transforms to use on dict
            preprocessing: Transforms to use on dict
            target_key: Target key. Default: "target"
        """
        self.open_fn = open_fn
        self.transforms = transforms if transforms is not None else {}
        self.preprocessing = preprocessing if preprocessing is not None else {}
        self.target_key = target_key
        self._apply_transforms = True
        self._labels = self._get_labels(data)
        self._length = len(data)
        self.data = data.copy().to_dict("records")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get element of the datasets.

        Args:
            index: index of the element in the datasets

        Returns:
            Single element by index
        """
        item = self.data[index]
        dict_ = self.open_fn(item)
        if self._apply_transforms:
            for _, function in self.transforms.items():
                dict_ = {**dict_, **function(dict_)}

        for _, function in self.preprocessing.items():
            dict_ = {**dict_, **function(dict_)}
        return dict_

    def __len__(self) -> int:
        """Get length of the datasets.

        Returns:
            int: length of the datasets
        """
        return self._length

    @property
    def apply_transforms(self) -> bool:
        """Get `apply_transforms` state."""
        return self._apply_transforms

    @property
    def labels(self) -> Optional[Sequence[int]]:
        """Get labels. Useful for a metric-learning case."""
        return self._labels

    def _get_labels(self, data: pd.DataFrame) -> Optional[Sequence[int]]:
        """Get labels."""
        if self.target_key in data:
            return list(data[self.target_key])
        return None

    @apply_transforms.setter  # type: ignore
    def apply_transforms(self, value: bool) -> None:
        """Set `apply_transforms` state."""
        self._apply_transforms = value


__all__ = ["MultiDomainCsvDataset"]
