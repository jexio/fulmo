import os
from typing import Any, Dict, Optional

import numpy as np

from ..utils.type_hints import NDArray
from .base import IReader


class NpyReader(IReader):
    """Npy array readers abstraction. Reads arrays from a ``csv`` datasets."""

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        rootpath: Optional[str] = None,
    ) -> None:
        """Create a new instance of NpyReader.

        Args:
            input_key: key to use from annotation dict
            output_key: key to use to store the result
            rootpath: path to images datasets root directory
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath

    def __call__(self, element: Dict[str, Any]) -> Dict[str, NDArray]:
        """Read a row from your annotations dict with filename and transfer it to an array.

        Args:
            element: elem in your datasets

        Returns:
            Dict[`output_key`, np.ndarray]
        """
        array_name = str(element[self.input_key])

        if self.rootpath is not None:
            array_name = array_name if array_name.startswith(self.rootpath) else os.path.join(self.rootpath, array_name)
        array = np.load(array_name)

        output = {self.output_key: array}
        return output


__all__ = ["NpyReader"]
