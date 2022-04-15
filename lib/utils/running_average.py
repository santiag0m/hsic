import torch


class RunningAverage:
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
