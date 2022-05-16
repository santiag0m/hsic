from typing import Callable

import torch
from torch.nn.functional import mse_loss, l1_loss, cross_entropy, one_hot

from .hsic import HSIC


def get_criterion(loss_criterion: str, **kwargs) -> Callable:

    num_classes = kwargs.pop("num_classes", None)

    def squared_loss(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return mse_loss(preds, target, **kwargs)

    def squared_loss_one_hot(
        features: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = one_hot(target, num_classes=num_classes)
        residual = target - preds
        return mse_loss(preds, residual, **kwargs)

    def absolute_loss(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return l1_loss(preds, target, **kwargs)

    def _cross_entropy(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return cross_entropy(preds, target, **kwargs)

    def hsic(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        residual = target - preds
        return HSIC(features, residual, **kwargs)

    def hsic_one_hot(
        features: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = one_hot(target, num_classes=num_classes)
        residual = target - preds
        return HSIC(features, residual, **kwargs)

    if loss_criterion == "squared_loss":
        return squared_loss
    elif loss_criterion == "squared_loss_one_hot":
        return squared_loss_one_hot
    elif loss_criterion == "absolute_loss":
        return absolute_loss
    elif loss_criterion == "cross_entropy":
        return _cross_entropy
    elif loss_criterion == "hsic":
        return hsic
    elif loss_criterion == "hsic_one_hot":
        return hsic_one_hot
    else:
        raise ValueError(f"Loss criterion '{loss_criterion}' is not supported")
