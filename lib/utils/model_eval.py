import torch.nn as nn


class model_eval:
    def __init__(self, model: nn.Module):
        self.model = model

    def __enter__(self):
        self.is_training = self.model.training
        self.model.eval()

    def __exit__(self, *args):
        if self.is_training:
            self.model.train()
