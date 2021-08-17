from typing import Any


class FrozenClass:
    """Class which prohibit ``__setattr__`` on existing attributes."""

    __is_frozen = False

    def __setattr__(self, key: Any, value: Any) -> None:
        if self.__is_frozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class for key {key}")
        object.__setattr__(self, key, value)

    def _freeze(self) -> None:
        """Freeze state."""
        self.__is_frozen = True

    def _unfreeze(self) -> None:
        """Unfreeze state."""
        self.__is_frozen = False


__all__ = ["FrozenClass"]
