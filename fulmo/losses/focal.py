# type: ignore[misc]
import torch
import torch.nn.functional as F
from torch import nn


class FocalCosineLoss(nn.Module):
    """Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification.

    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).
    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """

    def __init__(self, alpha: float = 1, gamma: float = 2, xent: float = 0.1, reduction: str = "mean") -> None:
        """Create a new instance of FocalCosineLoss."""
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        cosine_loss = F.cosine_embedding_loss(
            outputs,
            torch.nn.functional.one_hot(target, num_classes=outputs.size(-1)),
            torch.tensor([1], device=target.device),
            reduction=self.reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(outputs), target, reduction="none")
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


__all__ = ["FocalCosineLoss"]
