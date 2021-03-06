from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model_eval import model_eval
from .running_average import RunningAverage


def compute_accuracy(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    accuracy = RunningAverage()
    device = next(model.parameters()).device
    with torch.no_grad(), model_eval(model):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            preds = torch.argmax(preds, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())
    return accuracy.value.item()


def compute_mse(
    model: nn.Module,
    dataloader: DataLoader,
    target_transform: Optional[Callable] = None,
) -> torch.Tensor:
    mse = RunningAverage()
    device = next(model.parameters()).device
    with torch.no_grad(), model_eval(model):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if target_transform is not None:
                targets = target_transform(targets)
            preds = model(inputs)
            batch_mse = (targets - preds) ** 2
            batch_mse = batch_mse.sum(axis=1)
            mse.update(batch_mse.cpu())
    return mse.value.item()


def compute_bias(
    model: nn.Module,
    dataloader: DataLoader,
    target_transform: Optional[Callable] = None,
) -> torch.Tensor:
    target_avg = RunningAverage()
    pred_avg = RunningAverage()
    device = next(model.parameters()).device
    with torch.no_grad(), model_eval(model):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if target_transform is not None:
                targets = target_transform(targets)
            preds = model(inputs)
            target_avg.update(targets)
            pred_avg.update(preds)
    bias = target_avg.value - pred_avg.value
    return bias
