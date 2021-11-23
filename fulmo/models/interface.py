import abc


class BaseModel(metaclass=abc.ABCMeta):
    """A neural network for classification/regression tasks."""

    @property
    @abc.abstractmethod
    def num_parameters(self) -> int:
        """Get number of parameters."""
        ...


__all__ = ["BaseModel"]
