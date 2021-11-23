import importlib
import os
from typing import Any, Callable


SCHEDULER_REGISTRY = {}
SCHEDULER_DATACLASS_REGISTRY = {}


def register_scheduler(name: str, dataclass: object = None) -> Callable:
    """New scheduler types can be added to OpenSpeech with the :func:`register_scheduler` function decorator.

    For example::
        @register_scheduler('reduce_lr_on_plateau')
        class ReduceLROnPlateau:
            (...)
    .. note:: All scheduler must implement the :class:`cls.__name__` interface.

    Args:
        name: a name of the scheduler
        dataclass: a config of the scheduler

    Returns:
        A decorator function to register schedulers.
    """

    def _register_scheduler_cls(cls: Any) -> Any:
        """Add a scheduler to a registry."""
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"Cannot register duplicate scheduler ({name})")

        SCHEDULER_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in SCHEDULER_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate scheduler ({name})")
            SCHEDULER_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return _register_scheduler_cls


# automatically import any Python files in the models/ directory
scheduler_dir = os.path.dirname(__file__)
for file in os.listdir(scheduler_dir):
    if os.path.isdir(os.path.join(scheduler_dir, file)) and file != "__pycache__":
        for subfile in os.listdir(os.path.join(scheduler_dir, file)):
            path = os.path.join(scheduler_dir, file, subfile)
            if subfile.endswith(".py"):
                scheduler_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"fulmo.schedulers.{scheduler_name}")
        continue

    path = os.path.join(scheduler_dir, file)
    if file.endswith(".py"):
        scheduler_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"fulmo.schedulers.{scheduler_name}")


__all__ = ["SCHEDULER_REGISTRY", "SCHEDULER_DATACLASS_REGISTRY", "register_scheduler"]
