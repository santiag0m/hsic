import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.beta = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(data=torch.Tensor([0]), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta(x) + self.bias

    def update_bias(self, bias_value: torch.Tensor):
        if bias_value.shape == self.bias.shape:
            self.bias.copy_(bias_value)
        else:
            raise ValueError("Shape mismatch")
