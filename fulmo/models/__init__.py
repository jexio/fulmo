import importlib
import os
from typing import Any, Callable

from .interface import BaseModel


MODEL_REGISTRY = dict()
MODEL_DATACLASS_REGISTRY = dict()


def register_model(name: str, dataclass: object = None) -> Callable:
    """New model types can be added to Fulmo with the :func:`register_model` function decorator.

    For example::
        @register_model('image_classification')
        class ImageClassificationModel(BaseModel):
            (...)
    .. note:: All models must implement the :class:`cls.__name__` interface.

    Args:
        name: a name of the model
        dataclass: a config of the model

    Returns:
        A decorator function to register models.
    """

    def _register_model_cls(cls: Any) -> Any:
        """Add a model to a registry."""
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, BaseModel):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend FulmoModel")

        MODEL_REGISTRY[name] = cls
        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in MODEL_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate model ({name})")
            MODEL_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return _register_model_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    if os.path.isdir(os.path.join(models_dir, file)) and not file.startswith("__"):
        for subfile in os.listdir(os.path.join(models_dir, file)):
            path = os.path.join(models_dir, file, subfile)
            if subfile.endswith(".py"):
                python_file = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"fulmo.models.{file}.{python_file}")
        continue

    path = os.path.join(models_dir, file)
    if file.endswith(".py"):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"fulmo.models.{model_name}")
