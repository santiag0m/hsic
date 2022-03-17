import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class RunningAvg:
    def __init__(self, initial_value: float = 0):
        self.initial_value = initial_value
        self.reset()

    def update(self, batch: torch.Tensor):
        num_samples = batch.shape[0]
        cumulative = batch.sum(dim=0)

        if self.samples == 0:
            self.samples = num_samples
            self.value = cumulative / num_samples
        else:
            new_samples = self.samples + num_samples
            ratio = self.samples / new_samples
            self.value = (self.value * ratio) + (cumulative / new_samples)
            self.samples = new_samples

    def reset(self):
        self.value = self.initial_value
        self.samples = 0


def compute_bias(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    target_avg = RunningAvg()
    pred_avg = RunningAvg()
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
    return bias
