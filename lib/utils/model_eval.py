import torch.nn as nn


class model_eval:
    def __enter__(self, model: nn.Module):
        self.is_training = model.training
        self.model = model
        self.model.eval()

    def __exit__(self):
        if self.is_training:
            self.model.train()
