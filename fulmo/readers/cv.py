from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.type_hints import NDArray
from .base import IReader
from .functional import imread, mimread


class ImageReader(IReader):
    """Image readers abstraction. Reads images from a ``csv`` datasets."""

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        rootpath: Optional[str] = None,
        grayscale: bool = False,
    ) -> None:
        """Create a new instance of ImageReader.

        Args:
            input_key: key to use from annotation dict
            output_key: key to use to store the result
            rootpath: path to images datasets root directory
            grayscale: if True, make all images grayscale
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        self.grayscale = grayscale

    def __call__(self, element: Dict[str, Any]) -> Dict[str, NDArray]:
        """Read a row from your annotations dict with filename and  transfer it to an image.

        Args:
            element: elem in your datasets

        Returns:
            Dict[`output_key`, Image]
        """
        image_name = str(element[self.input_key])
        img = imread(image_name, rootpath=self.rootpath, grayscale=self.grayscale)

        output = {self.output_key: np.array(img)}
        return output


class MaskReader(IReader):
    """Mask reader abstraction. Reads masks from a `csv` dataset."""

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        rootpath: Optional[str] = None,
        clip_range: Tuple[int, int] = (0, 1),
    ) -> None:
        """Create a new instance of MaskReader.

        Args:
            input_key: key to use from annotation dict
            output_key: key to use to store the result, default: ``input_key``
            rootpath: path to images dataset root directory (so your can use relative paths in annotations)
            clip_range: lower and upper interval edges, image values outside the interval are clipped
                to the interval edges
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        self.clip = clip_range

    def __call__(self, element: Dict[str, Any]) -> Dict[str, NDArray]:
        """Reads a row from your annotations dict with filename and transfer it to a mask.

        Args:
            element: elem in your dataset.

        Returns:
            np.ndarray: Mask
        """
        mask_name = str(element[self.input_key])
        mask = mimread(mask_name, rootpath=self.rootpath, clip_range=self.clip)

        output = {self.output_key: np.array(mask)}
        return output


__all__ = ["ImageReader", "MaskReader"]
