import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, softmax: bool = False):
        super().__init__()
        self.softmax = softmax
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.fc_1 = nn.Linear(in_features=7 * 7 * 64, out_features=524)
        self.fc_2 = nn.Linear(in_features=524, out_features=10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bias = nn.Parameter(data=torch.Tensor([0] * 10), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv_1(x))
        x = self.pool(x)
        x = self.relu(self.conv_2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        x = x + self.bias
        if self.softmax:
            x = torch.softmax(x, dim=-1)
        return x

    def update_bias(self, bias_value: torch.Tensor):
        if bias_value.shape == self.bias.shape:
            self.bias.copy_(bias_value)
        else:
            raise ValueError("Shape mismatch")
