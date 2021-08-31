from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from ..settings import Stage


@dataclass(frozen=True)
class BaseDataModuleParameters:
    """Base parameters."""

    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = False
    drop_last: bool = False

    @classmethod
    def from_config(cls, parameters: Dict[str, Any]) -> "BaseDataModuleParameters":
        """Create a new instance of `BaseDataModuleParameters` from `Dict`"""
        return cls(
            parameters["batch_size"], parameters["num_workers"], parameters["pin_memory"], parameters["drop_last"]
        )


class BaseDataModule(LightningDataModule):
    """This is example of lightning datamodule for base case."""

    def __init__(
        self,
        data_dir: str,
        parameters: Union[BaseDataModuleParameters, Dict[str, BaseDataModuleParameters]],
        sampler: Sampler[int] = None,
    ) -> None:
        """Create a new instance of BaseDataModule."""
        super().__init__()
        self.data_dir = data_dir
        self.parameters = parameters
        self._labels: Optional[List[str]] = None
        self.sampler: Optional[Sampler[int]] = sampler
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

    @property
    def batch_size(self) -> int:
        """Get batch_size."""
        if isinstance(self.parameters, BaseDataModuleParameters):
            return self.parameters.batch_size
        return self.parameters[Stage.train.value].batch_size

    @property
    def collate_fn(self) -> Optional[Callable[[Dict[str, Any]], Dict[str, torch.Tensor]]]:
        """Get collate_fn."""
        return None

    @property
    def labels(self) -> Optional[List[str]]:
        """Get class names."""
        return self._labels

    def _get_parameters(self, stage: Stage) -> Dict[str, Union[int, bool, float]]:
        """Get dataloader parameters."""
        if isinstance(self.parameters, BaseDataModuleParameters):
            dict_ = asdict(self.parameters)
        else:
            dict_ = asdict(self.parameters[stage.value])
        return dict_

    def prepare_data(self) -> None:
        """Download data if needed. This method is called only from a single GPU.

        Raises:
            NotImplementedError: if `__class__` is "BaseDataModule"
        """
        raise NotImplementedError("Implement in child.")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: data_train, data_val, data_test.

        Args:
            stage: Stage name.

        Raises:
            NotImplementedError: if `__class__` is "BaseDataModule"
        """
        raise NotImplementedError("Child class must implement method")

    def train_dataloader(self) -> DataLoader:
        """Create dataloader for train stage."""
        shuffle = True if not self.sampler else False
        return DataLoader(
            dataset=self.data_train,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            shuffle=shuffle,
            **self._get_parameters(Stage.train),
        )

    def val_dataloader(self) -> DataLoader:
        """Create dataloader for valid stage."""
        return DataLoader(
            dataset=self.data_val, collate_fn=self.collate_fn, shuffle=False, **self._get_parameters(Stage.val)
        )

    def test_dataloader(self) -> DataLoader:
        """Create dataloader for test stage."""
        if self.data_test:
            return DataLoader(
                dataset=self.data_test, collate_fn=self.collate_fn, shuffle=False, **self._get_parameters(Stage.test)
            )

    def predict_dataloader(self) -> DataLoader:
        """Create dataloader for predict stage."""
        if self.data_predict:
            return DataLoader(
                dataset=self.data_predict, collate_fn=self.collate_fn, shuffle=False, **self._get_parameters(Stage.val)
            )


__all__ = ["BaseDataModule", "BaseDataModuleParameters"]
