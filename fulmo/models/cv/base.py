import abc
from typing import Dict, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn

from ..layers.activation import Mish, Swish


activation_dict = dict(lrelu=nn.LeakyReLU, relu=nn.ReLU, elu=nn.ELU, mish=Mish, swish=Swish, identity=nn.Identity)


class AbstractEncoder(metaclass=abc.ABCMeta):
    """Top layer in nn."""

    @property
    @abc.abstractmethod
    def out_features(self) -> int:
        """Get size of tensor before pooling layer."""
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the `x`."""
        raise NotImplementedError()


class AbstractHead(metaclass=abc.ABCMeta):
    """Last layer in nn."""

    @abc.abstractmethod
    def _init_weights(self) -> None:
        """Initialize layers with some weights."""
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the layers."""
        raise NotImplementedError()

    @abc.abstractmethod
    def in_features(self) -> int:
        """Get size of input for last fc layer."""
        raise NotImplementedError()

    @abc.abstractmethod
    def out_channels(self) -> int:
        """Get size of output for last fc layer."""
        raise NotImplementedError()


class Head(AbstractHead, nn.Module):
    """Last layer in nn."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_name: str = "relu",
        num_layers: int = 0,
        dropout: float = 0,
        activation_parameters: Optional[Dict[str, Union[float, int, str]]] = None,
        layers_order: Tuple[str, ...] = ("linear", "bn", "activation", "dropout"),
    ) -> None:
        """Create a new instance of Head.

        Args:
            in_channels: size of each input sample.
            out_channels: size of each output sample.
            activation_name: name of activation.
            num_layers: number of block except the last fc with following layers.
            dropout: append "Dropout" or not.
            activation_parameters: specific parameters for the selected activation type.
            layers_order: Order of layers for the head. Default - ("Linear", "BatchNorm", "Activation, "Dropout")

        Raises:
            KeyError: if `activation_name` does not implemented
            ValueError: if `layers_order` contains an unsupported layer.
        """
        if activation_name not in activation_dict:
            raise KeyError(f"Activation {activation_name} does not implemented")
        super().__init__()

        self._layers: Sequence[nn.ModuleList] = []
        layer_names = layers_order * num_layers
        include_bn = "bn" in layers_order
        activation_parameters = activation_parameters if activation_parameters else {}
        layers = list()

        for name in layer_names:
            if name == "linear":
                out_features = in_channels // 2
                layer = nn.Linear(in_features=in_channels, out_features=out_features, bias=not include_bn)
                in_channels = out_features
            elif name == "bn":
                layer = nn.BatchNorm1d(in_channels)
            elif name == "activation":
                layer = activation_dict[activation_name](**activation_parameters)
            elif name == "dropout":
                layer = nn.Dropout(p=dropout)
            else:
                raise ValueError(f"Invalid name of layer - {name}")

            layers.append(layer)

        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, out_channels, bias=not include_bn)
        self._init_weights()

    @property
    def in_features(self) -> int:
        """Get size of input for last fc layer."""
        return int(self.fc.in_features)

    @property
    def out_channels(self) -> int:
        """Get size of output for last fc layer."""
        return int(self.fc.out_features)

    def _init_weights(self) -> None:
        """Initialize layers with some weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the layers."""
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        x = self.extract_features(x)
        x = self.fc(x)

        return x


class Encoder(AbstractEncoder, nn.Module):
    """Feature extractor based on `timm` library."""

    def __init__(self, name: str, pretrained: bool = False, in_channels: int = 3) -> None:
        """Create a new instance of Encoder.

        Args:
            name: name of models to instantiate.
            pretrained: if true load pretrained weights.
            in_channels: number of input channels.
        """
        super(Encoder, self).__init__()
        self.model = timm.create_model(name, pretrained=pretrained, in_chans=in_channels, num_classes=0, global_pool="")
        self._in_features: int = self.model.num_features

    @property
    def out_features(self) -> int:
        """Get size of tensor before pooling layer."""
        return self._in_features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from layers excluding `pool` and `fc` layers."""
        return self.model.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        x = self.extract_features(x)
        return x


__all__ = ["AbstractEncoder", "Encoder", "Head"]
