from typing import Dict, Optional, Union

import numpy as np

from ..utils.type_hints import IntegerScalar, NPType
from .base import IReader


class ScalarReader(IReader):
    """Numeric loader readers abstraction.  Reads a single float, int, str or other from loader."""

    def __init__(
        self,
        input_key: str,
        output_key: str,
        dtype: NPType = np.float32,
        default_value: float = -1.0,
        one_hot_classes: Optional[int] = None,
        smoothing: Optional[float] = None,
    ) -> None:
        """Create a new instance of ScalarReader.

        Args:
            input_key: input key to use from annotation dict
            output_key: output key to use to store the result
            dtype: datatype of scalar values to use
            default_value: default value to use if something goes wrong
            one_hot_classes: number of one-hot classes
            smoothing: if specified applies label smoothing to one_hot classes
        """
        super().__init__(input_key, output_key)
        self.dtype = dtype
        self.default_value = default_value
        self.one_hot_classes = one_hot_classes
        self.smoothing = smoothing
        if one_hot_classes is not None and smoothing is not None:
            assert 0.0 < smoothing < 1.0, f"If smoothing is specified it must be in (0; 1), " f"got {smoothing}"

    def __call__(self, element: Dict[str, Union[int, float, IntegerScalar]]) -> Dict[str, IntegerScalar]:
        """Read a row from your annotations dict and transfer it to a single value.

        Args:
            element: elem in your datasets

        Returns:
            dtype: Scalar value
        """
        scalar = self.dtype(element.get(self.input_key, self.default_value))
        if self.one_hot_classes is not None:
            scalar = get_one_hot(scalar, self.one_hot_classes, smoothing=self.smoothing)
        output = {self.output_key: scalar}
        return output


def get_one_hot(label: int, num_classes: int, smoothing: Optional[float] = None) -> IntegerScalar:
    """Apply OneHot vectorization to a giving scalar, optional with label smoothing.

    Args:
        label: scalar value to be vectorized
        num_classes: total number of classes
        smoothing: if specified applies label smoothing

    Returns:
        a one-hot vector with shape (num_classes,)
    """
    assert num_classes > 0, f"Expect num_classes to be > 0, got {num_classes}"

    assert 0 <= label < num_classes, f"Expect label to be in [0; {num_classes}), got {label}"

    if smoothing is not None:
        assert 0.0 < smoothing < 1.0, f"If smoothing is specified it must be in (0; 1), got {smoothing}"

        smoothed = smoothing / float(num_classes - 1)
        result = np.full((num_classes,), smoothed, dtype=np.float32)
        result[label] = 1.0 - smoothing

        return result

    result = np.zeros(num_classes, dtype=np.float32)
    result[label] = 1.0

    return result


__all__ = ["ScalarReader"]
