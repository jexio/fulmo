import torch.nn as nn


def replace_dropout(model: nn.Module) -> None:
    """Change dropout value to zero."""
    for child_name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            setattr(model, child_name, nn.Dropout(p=0.0))
        else:
            replace_dropout(child)


def freeze_bn(model: nn.Module) -> None:
    """Freeze all BatchNorm layers."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def unfreeze_bn(model: nn.Module) -> None:
    """Unfreeze all BatchNorm layers."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.requires_grad = True
            m.bias.requires_grad = True


__all__ = ["freeze_bn", "replace_dropout", "unfreeze_bn"]
