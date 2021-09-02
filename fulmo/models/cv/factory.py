from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig

from ..interface import BaseModel
from ..layers.pooling import SelectAdaptivePool2d
from .base import Encoder, Head


class ImageClassificationModel(BaseModel, nn.Module):
    """A neural network for image classification tasks.

    Create nn with the following structure. Encoder -> Pooling -> Head.
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool = False,
        in_channels: int = 3,
        pool_name: str = "avg",
        pool_parameters: Optional[Dict[str, Union[float, int]]] = None,
        num_layers: int = 1,
        dropout_head: float = 0.0,
        activation_head_name: str = "relu",
        activation_head_parameters: Optional[Dict[str, Union[float, int, str]]] = None,
        layers_head_order: ListConfig = ("lin", "bn", "activation", "dp"),
        num_classes: int = 1,
    ) -> None:
        """Create a new instance of ImageClassificationModel.

        Args:
            backbone: name of models to instantiate
            pretrained: load pretrained weights if true
            in_channels: number of input channels for backbone
            pool_name: name of pooling to instantiate
            pool_parameters: specific parameters for the selected pool type
            num_layers: number of block except the last fc with following layers
            dropout_head: append "Dropout" or not.
            activation_head_name: name of activation
            activation_head_parameters: specific parameters for the selected activation type
            layers_head_order: layers order
            num_classes: number of classes to learn
        """
        super(ImageClassificationModel, self).__init__()

        layers_head_order = [str(item) for item in layers_head_order]
        self._apply_transforms = True
        encoder = Encoder(backbone, pretrained=pretrained, in_channels=in_channels)
        pool_in_channels = encoder.out_features
        self.backbone: nn.Module = nn.Sequential(*encoder.model.children())
        pool_parameters = pool_parameters if pool_parameters else {}
        self.pool: nn.Module = SelectAdaptivePool2d(
            in_channels=pool_in_channels,
            pool_name=pool_name,
            **pool_parameters,
        )
        pool_out_channels: int = pool_in_channels * self.pool.multiplication_coefficient
        self.head = Head(
            in_channels=pool_out_channels,
            out_channels=num_classes,
            num_layers=num_layers,
            dropout=dropout_head,
            activation_name=activation_head_name,
            activation_parameters=activation_head_parameters,
            layers_order=tuple(layers_head_order),
        )

    @property
    def apply_transforms(self) -> bool:
        """Get `apply_transforms` state."""
        return self._apply_transforms

    @property
    def in_features(self) -> int:
        """Get size of input tensor for fc layer."""
        return self.head.in_features

    @property
    def num_parameters(self) -> int:
        """Get number of parameters."""
        return sum(param.numel() for param in self.parameters())

    @apply_transforms.setter  # type: ignore
    def apply_transforms(self, value: bool) -> None:
        """Set `apply_transforms` state."""
        self._apply_transforms = value

    def extract_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract features from the layers."""
        features = self.backbone(features)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        features = self.extract_features(features)
        features = self.head(features)
        return features


class ImageClassificationModelExtendsKornia(ImageClassificationModel):
    """Create a neural network for image classification tasks.

    Perform data augmentation using Kornia
    Create nn with the following structure. Transforms -> Encoder -> Pooling -> Head.
    """

    def __init__(
        self,
        transforms: ListConfig,
        backbone: str,
        pretrained: bool = False,
        in_channels: int = 3,
        pool_name: str = "avg",
        pool_parameters: Optional[Dict[str, Union[float, int]]] = None,
        num_layers: int = 1,
        dropout_head: float = 0.0,
        activation_head_name: str = "relu",
        activation_head_parameters: Optional[Dict[str, Union[float, int, str]]] = None,
        layers_head_order: ListConfig = ("lin", "bn", "activation", "dp"),
        num_classes: int = 1,
    ) -> None:
        """Create a new instance of ImageClassificationModelExtendsKornia.

        Args:
            transforms: list of augmentations to instantiate
            backbone: name of models to instantiate
            in_channels: number of input channels for backbone
            pool_name: name of pooling to instantiate
            pool_parameters: specific parameters for the selected pool type
            num_layers: number of block except the last fc with following layers
            dropout_head: append "Dropout" or not.
            activation_head_name: name of activation
            activation_head_parameters: specific parameters for the selected activation type
            layers_head_order: layers order
            num_classes: number of classes to learn
            pretrained: load pretrained ImageNet-1k weights if true
        """
        super(ImageClassificationModelExtendsKornia, self).__init__(
            backbone,
            pretrained,
            in_channels,
            pool_name,
            pool_parameters,
            num_layers,
            dropout_head,
            activation_head_name,
            activation_head_parameters,
            layers_head_order,
            num_classes,
        )
        self.transforms = nn.Sequential(*transforms)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kornia augmentations to `x`."""
        return self.transforms(x)

    def extract_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract features from the layers."""
        if self.apply_transforms:
            features = self._augment(features)
        features = self.backbone(features)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features


__all__ = ["ImageClassificationModel", "ImageClassificationModelExtendsKornia"]
