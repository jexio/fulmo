from typing import Optional, Tuple

import numpy as np

from ..utils.type_hints import IntegerScalar


def rand_bbox(
    img_shape: Tuple[int, int], lam: float, margin: float = 0.0, count: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """Get standard CutMix bounding-box.

    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape: Image shape as tuple
        lam: Cutmix lambda value
        margin: Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count: Number of bbox to generate

    Returns:
        top-left and bottom-right coordinates of the box
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)

    yl: IntegerScalar = np.clip(cy - cut_h // 2, 0, img_h)
    yh: IntegerScalar = np.clip(cy + cut_h // 2, 0, img_h)
    xl: IntegerScalar = np.clip(cx - cut_w // 2, 0, img_w)
    xh: IntegerScalar = np.clip(cx + cut_w // 2, 0, img_w)
    return xl.item(), yl.item(), xh.item(), yh.item()


def rand_bbox_minmax(
    img_shape: Tuple[int, int], minmax: Tuple[int, int], count: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """Get Min-Max CutMix bounding-box.

    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.
    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape: Image shape as tuple
        minmax: Min and max bbox ratios (as percent of image size)
        count: Number of bbox to generate

    Returns:
        top-left and bottom-right coordinates of the box
    """
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu: IntegerScalar = yl + cut_h
    xu: IntegerScalar = xl + cut_w
    return xl.item(), yl.item(), xu.item(), yu.item()


def cutmix_bbox_and_lam(
    img_shape: Tuple[int, int],
    lam: float,
    ratio_minmax: Optional[Tuple[int, int]] = None,
    correct_lam: bool = True,
    count: Optional[int] = None,
) -> Tuple[int, int, int, int, float]:
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return yl, yu, xl, xu, lam


__all__ = ["cutmix_bbox_and_lam", "rand_bbox", "rand_bbox_minmax"]
