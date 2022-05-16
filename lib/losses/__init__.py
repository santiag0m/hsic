from typing import Optional, Callable

import torch
from torch.nn.functional import mse_loss, l1_loss, cross_entropy

from .hsic import HSIC


def get_criterion(
    loss_criterion: str, target_transform: Optional[Callable] = None, **kwargs
) -> Callable:

    if target_transform is None:
        target_transform = lambda x: x

    def squared_loss(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target_transform(target)
        return mse_loss(preds, target, **kwargs)

    def absolute_loss(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target_transform(target)
        return l1_loss(preds, target, **kwargs)

    def _cross_entropy(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target_transform(target)
        return cross_entropy(preds, target, **kwargs)

    def hsic(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target_transform(target)
        residual = target - preds
        return HSIC(features, residual, **kwargs)

    if loss_criterion == "squared_loss":
        return squared_loss
    elif loss_criterion == "absolute_loss":
        return absolute_loss
    elif loss_criterion == "cross_entropy":
        return _cross_entropy
    elif loss_criterion == "hsic":
        return hsic
    else:
        raise ValueError(f"Loss criterion '{loss_criterion}' is not supported")
