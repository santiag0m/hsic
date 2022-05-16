import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .running_average import RunningAverage


def compute_accuracy(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    accuracy = RunningAverage()
    is_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            preds = torch.argmax(preds, dim=-1)
            correct = preds == targets
            accuracy.update(correct.cpu())
    if is_training:
        model.train()
    return accuracy.value.item()


def compute_bias(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    target_avg = RunningAverage()
    pred_avg = RunningAverage()
    is_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            target_avg.update(targets)
            pred_avg.update(preds)
    bias = target_avg.value - pred_avg.value
    if is_training:
        model.train()
    return bias
