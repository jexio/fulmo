from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig


@dataclass
class DeepMMDataclass:
    """DeepMM base dataclass that supported fetching attributes and metas."""

    def _get_all_attributes(self) -> List[str]:
        """Get all attributes."""
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(self, attribute_name: str, meta: str, default: Optional[Any] = None) -> Any:
        """Get metadata of `attribute_name`."""
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        """Get name of `attribute_name`."""
        return self.__dataclass_fields__[attribute_name].name

    def _get_value(self, attribute_name: str) -> Any:
        """Get value of `attribute_name`."""
        return getattr(self, attribute_name)

    def _get_default(self, attribute_name: str) -> Any:
        """Get default value of a field."""
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith("${"):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif getattr(self, attribute_name) != self.__dataclass_fields__[attribute_name].default:
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        """Get type of `attribute_name`."""
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        """Get help field for `attribute_name`."""
        return self._get_meta(attribute_name, "help")


@dataclass
class LearningRateSchedulerConfigs(DeepMMDataclass):
    """Super class of learning rate dataclass."""

    interval: str = field(
        default="epoch",
        metadata={
            "help": "The unit of the scheduler's step size, could also be 'step'. "
            "'epoch' updates the scheduler on epoch end whereas "
            "'step' updates it after a optimizer update.",
        },
    )
    frequency: int = field(
        default=1,
        metadata={
            "help": "How many epochs/steps should pass " "between calls to `scheduler.step()`",
        },
    )

    @property
    def lightning_parameters(self) -> Dict[str, Any]:
        """Get parameters related to pytorch-lightning."""
        return {key: self._get_value(key) for key in self._lightning_keys()}

    @property
    def scheduler_parameters(self) -> Dict[str, Any]:
        """Get parameters related to a scheduler."""
        return {key: self._get_value(key) for key in self._scheduler_keys()}

    @staticmethod
    def _lightning_keys() -> List[str]:
        """Get all keys related to a pytorch-lightning scheduler."""
        return ["interval", "frequency"]

    @staticmethod
    def _scheduler_keys() -> List[str]:
        """Get all keys related to a scheduler."""
        return []

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> "LearningRateSchedulerConfigs":
        """Create a new instance of LearningRateSchedulerConfigs from a `DictConfig` object."""
        d_ = {key: value for key, value in config.items() if key != "_target_"}
        return cls(**d_)


__all__ = ["LearningRateSchedulerConfigs"]
