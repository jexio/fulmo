import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    """Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition. https://arxiv.org/abs/1406.4729"""

    def __init__(self, levels: Tuple[int, ...], mode: str = "max") -> None:
        """General Pyramid Pooling class which uses Spatial Pyramid Pooling by default.

        And holds the static methods for both spatial and temporal pooling.

        Args:
            levels: defines the different divisions to be made in the width and (spatial) height dimension
            mode: defines the underlying pooling mode to be used, can either be "max" or "avg"
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    @staticmethod
    def spatial_pyramid_pool(previous_conv: torch.Tensor, levels: Tuple[int, ...], mode: str) -> torch.Tensor:
        """Apply Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally.

        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.

        Args:
            previous_conv: input tensor of the previous convolutional layer
            levels: defines the different divisions to be made in the width and height dimension
            mode: defines the underlying pooling mode to be used, can either be "max" or "avg"

        Returns:
            a tensor vector with shape [batch x 1 x n],
            where n: sum(filter_amount*level*level) for each level in levels
            which is the concentration of multi-level pooling

        Raises:
            ValueError: if mode not in ("avg", "max")
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and h_pad1 + h_pad2 == (
                h_kernel * levels[i] - previous_conv_size[0]
            )

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2], mode="constant", value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise ValueError('Unknown pooling type: %s, please use "max" or "avg".')
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

    @staticmethod
    def temporal_pyramid_pool(previous_conv: torch.Tensor, out_pool_size: Tuple[int, ...], mode: str) -> torch.Tensor:
        """Apply Static Temporal Pyramid Pooling method, which divides the input Tensor horizontally (last dimensions).

        According to each level in the given levels and pools its value according to the given mode.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly
        and the original height and pools the values inside this stripe

        Args:
            previous_conv: input tensor of the previous convolutional layer
            out_pool_size: defines the different divisions to be made in the width dimension
            mode: defines the underlying pooling mode to be used, can either be "max" or "avg"

        Returns:
            a tensor vector with shape [batch x 1 x n],
            where n: sum(filter_amount*level) for each level in levels
            which is the concentration of multi-level pooling

        Raises:
            ValueError: if mode not in ("avg", "max")
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(out_pool_size)):
            h_kernel = previous_conv_size[0]
            w_kernel = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            w_pad1 = int(math.floor((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * out_pool_size[i] - previous_conv_size[1])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2], mode="constant", value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise ValueError('Unknown pooling type: %s, please use "max" or "avg".')
            x = pool(padded_input)
            if i == 0:
                tpp = x.view(num_sample, -1)
            else:
                tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)

        return tpp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters: int) -> int:
        """Calculate the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels.

        Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers

        Args:
            filters: the amount of filter of output fed into the spatial pyramid pooling

        Returns:
            sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


class SpatialPyramidPooling(PyramidPooling):
    """Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition https://arxiv.org/abs/1406.4729"""

    def __init__(self, levels: Tuple[int, ...], mode: str = "max") -> None:
        """Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally.

        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
        and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)

        Args:
            levels: defines the different divisions to be made in the width dimension
            mode: defines the underlying pooling mode to be used, can either be "max" or "avg"
        """
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters: int) -> int:
        """Calculate the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels.

        Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers

        Args:
             filters: the amount of filter of output fed into the spatial pyramid pooling

        Returns:
             sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


__all__ = ["SpatialPyramidPooling"]
