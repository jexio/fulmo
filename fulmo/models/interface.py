import abc


class BaseModel(metaclass=abc.ABCMeta):
    """A neural network for classification/regression tasks."""

    @property
    @abc.abstractmethod
    def apply_transforms(self) -> bool:
        """Get `apply_transforms` state."""
        ...

    @property
    @abc.abstractmethod
    def num_parameters(self) -> int:
        """Get number of parameters."""
        ...

    @apply_transforms.setter
    @abc.abstractmethod
    def apply_transforms(self, value: bool) -> None:
        """Set `apply_transforms` state."""
        ...


__all__ = ["BaseModel"]
