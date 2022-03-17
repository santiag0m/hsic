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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta(x)
