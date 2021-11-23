import os
import pathlib
from typing import Any, Optional, Tuple, Union

import imageio
import numpy as np
from skimage.color import rgb2gray

from ..utils.type_hints import IntegerScalar, NDArray


def imread(
    uri: str,
    grayscale: bool = False,
    expand_dims: bool = True,
    rootpath: Optional[Union[str, pathlib.Path]] = None,
    **kwargs: Any,
) -> NDArray:
    """Reads an image from the specified file.

    Args:
        uri: the resource to load the image
          from, e.g. a filename, ``pathlib.Path``, http address or file object,
          see ``imageio.imread`` docs for more info
        grayscale: if True, make all images grayscale
        expand_dims: if True, append channel axis to grayscale images
          rootpath: path to the resource with image
          (allows to use relative path)
        rootpath: path to the resource with image
            (allows to use relative path)
        **kwargs: extra params for image read

    Returns:
        np.ndarray: image
    """
    uri = str(uri)

    if rootpath is not None:
        rootpath = str(rootpath)
        uri = uri if uri.startswith(rootpath) else os.path.join(rootpath, uri)

    img: NDArray = imageio.imread(uri, as_gray=grayscale, pilmode="RGB", **kwargs)

    if grayscale:
        img = rgb2gray(img)

    if expand_dims and len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    return img


def mimread(
    uri: str,
    clip_range: Optional[Tuple[int, int]] = None,
    expand_dims: bool = True,
    rootpath: Optional[Union[str, pathlib.Path]] = None,
    **kwargs: Any,
) -> NDArray:
    """Reads multiple images from the specified file.

    Args:
        uri (str, pathlib.Path, bytes, file): the resource to load the image
          from, e.g. a filename, ``pathlib.Path``, http address or file object,
          see ``imageio.mimread`` docs for more info
        clip_range (Tuple[int, int]): lower and upper interval edges,
          image values outside the interval are clipped to the interval edges
        expand_dims: if True, append channel axis to grayscale images
          rootpath (Union[str, pathlib.Path]): path to the resource with image
          (allows to use relative path)
        rootpath (Union[str, pathlib.Path]): path to the resource with image
            (allows to use relative path)
        **kwargs: extra params for image read

    Returns:
        np.ndarray: image
    """
    if rootpath is not None:
        uri = uri if uri.startswith(str(rootpath)) else os.path.join(rootpath, uri)

    image: Union[Any, NDArray] = np.dstack(imageio.mimread(uri, **kwargs))
    if clip_range is not None:
        image = np.clip(image, *clip_range)

    if expand_dims and len(image.shape) < 3:  # grayscale
        image = np.expand_dims(image, -1)

    return image


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


__all__ = [
    "imread",
    "mimread",
    "get_one_hot",
]
