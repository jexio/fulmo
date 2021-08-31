from enum import Enum
from typing import Optional, Union

from .utils.frozen_class import FrozenClass
from .utils.logging import get_logger


class Stage(str, Enum):
    """Stage names related to `pytorch-lightning` naming convention."""

    train = "train"
    val = "val"
    test = "test"

    def __eq__(self, other: Union[str, Enum]) -> bool:  # type: ignore[override]
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.name)

    @classmethod
    def from_str(cls, value: str) -> Optional["Stage"]:
        """Create a new instance of `Stage` from string."""
        statuses = [status for status in dir(cls) if not status.startswith("_")]
        for st in statuses:
            if st.lower() == value.lower():
                return getattr(cls, st)  # type: ignore[no-any-return]
        return None


class Settings(FrozenClass):
    """Settings for the entire project."""

    def __init__(
        self,
        mix_target_key: str = "mixed_target",
        mix_lam_key: str = "lam",
    ) -> None:
        self.mix_target_key = mix_target_key
        self.mix_lam_key = mix_lam_key


DEFAULT_SETTINGS = Settings()
logger = get_logger("global")


__all__ = ["DEFAULT_SETTINGS", "Settings", "Stage", "logger"]
