from typing import Callable

import torch
from torch.nn.functional import mse_loss, l1_loss

from .hsic import HSIC


def get_criterion(loss_criterion: str) -> Callable:
    def squared_loss(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return mse_loss(preds, target)

    def absolute_loss(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return l1_loss(preds, target)

    def hsic(
        features: torch.Tensor, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        residual = target - preds
        return HSIC(features, residual)

    if loss_criterion == "squared_loss":
        return squared_loss
    elif loss_criterion == "absolute_loss":
        return absolute_loss
    elif loss_criterion == "hsic":
        return hsic
    else:
        raise ValueError(f"Loss criterion '{loss_criterion}' is not supported")
