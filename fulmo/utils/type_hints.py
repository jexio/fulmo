"""Mypy stubs"""
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    import numpy.typing as npt

    NDArray = npt.NDArray[Any]
    IntegerScalar = npt.NDArray[np.int_]
    NPType = npt.Any
else:
    NDArray = np.ndarray
    IntegerScalar = np.int_
    NPType = np.dtype

__all__ = ["IntegerScalar", "NDArray", "NPType"]
