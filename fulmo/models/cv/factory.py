from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...dataclass.configurations import DeepMMDataclass
from ...modules.pooling import SelectAdaptivePool2d
from .. import register_model
from ..interface import BaseModel
from .base import Encoder, Head


@dataclass
class ImageClassificationModel(DeepMMDataclass):
    """Super class of image classification model."""

    name: str = field(default="image_classification", metadata={"help": "Model name"})
    backbone: str = field(default="resnet18", metadata={"help": "A name of models to instantiate."})
    model_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"pretrained": False, "in_chans": 1, "num_classes": 0, "global_pool": ""},
        metadata={"help": "A timm model parameters."},
    )
    pool_name: Optional[str] = field(default=None, metadata={"help": "A name of pooling to instantiate."})
    pool_parameters: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Specific parameters for the selected pool type.", "value_type": "Union[float, int]"},
    )
    num_layers: int = field(default=1, metadata={"help": "Number of block except the last fc with following layers."})
    dropout_head: float = field(default=0.0, metadata={"help": "Append `Dropout` with passed value or not."})
    activation_head_name: str = field(default="relu", metadata={"help": "A name of activation."})
    activation_head_parameters: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Specific parameters for the selected activation type.",
            "value_type": "Union[float, int, str]",
        },
    )
    layers_head_order: List = field(
        default_factory=lambda: ["lin", "bn", "activation", "dp"], metadata={"help": "Layers order."}
    )
    num_classes: int = field(default=1, metadata={"help": "Number of classes to learn."})


@register_model("image_classification", dataclass=ImageClassificationModel)
class ImageClassificationModel(BaseModel, nn.Module):
    """A neural network for image classification tasks.

    Create nn with the following structure. Encoder -> Pooling -> Head.
    """

    def __init__(
        self,
        config: ImageClassificationModel,
    ) -> None:
        """Create a new instance of ImageClassificationModel.

        Args:
            config: Configuration set.
        """
        super().__init__()

        layers_head_order = config.layers_head_order
        if not isinstance(layers_head_order, Tuple):
            layers_head_order = tuple(layers_head_order)

        self._apply_transforms = True
        self.backbone: nn.Module = Encoder(config.backbone, **config.model_parameters)
        pool_in_channels = self.backbone.out_features
        pool_parameters = config.pool_parameters if config.pool_parameters else {}
        self.pool = nn.Identity()

        if config.pool_name:
            self.pool: nn.Module = SelectAdaptivePool2d(
                in_channels=pool_in_channels,
                pool_name=config.pool_name,
                **pool_parameters,
            )
            pool_in_channels: int = pool_in_channels * self.pool.multiplication_coefficient

        self.head = Head(
            in_channels=pool_in_channels,
            out_channels=config.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout_head,
            activation_name=config.activation_head_name,
            activation_parameters=config.activation_head_parameters,
            layers_order=layers_head_order,
        )

    @property
    def in_features(self) -> int:
        """Get size of input tensor for fc layer."""
        return self.head.in_features

    @property
    def num_parameters(self) -> int:
        """Get number of parameters."""
        return sum(param.numel() for param in self.parameters())

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


__all__ = ["ImageClassificationModel"]
