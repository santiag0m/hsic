from typing import Tuple, Optional

import torch
import numpy as np
from torch.utils.data import TensorDataset


class SyntheticDataset(TensorDataset):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
        noise_distribution: str,
        input_distribution: str,
        beta: Optional[torch.Tensor] = None,
    ):
        if beta is None:
            beta = sample_beta(num_features)
        inputs, targets = generate_dataset(
            num_samples=num_samples,
            beta=beta,
            noise_distribution=noise_distribution,
            input_distribution=input_distribution,
        )
        super().__init__(inputs, targets)

        self.beta = beta
        self.noise_distribution = noise_distribution
        self.input_distribution = input_distribution


def generate_dataset(
    *,
    num_samples: int,
    beta: torch.Tensor,
    noise_distribution: str,
    input_distribution: str,
) -> Tuple[torch.Tensor]:
    in_features = beta.shape[0]
    inputs = sample_inputs(
        num_samples, in_features, input_distribution=input_distribution
    )
    targets = inputs @ beta
    targets = add_noise(targets, noise_distribution=noise_distribution)
    return inputs, targets


def sample_beta(num_features: int, sigma: float = 0.1) -> torch.Tensor:
    beta = torch.randn(size=(num_features, 1)) * sigma
    return beta


def sample_inputs(
    num_samples: int, num_features: int, *, input_distribution: str
) -> torch.Tensor:
    if input_distribution == "uniform":
        inputs = sample_uniform_inputs(num_samples, num_features)
    elif input_distribution == "gaussian":
        inputs = sample_gaussian_inputs(num_samples, num_features)
    else:
        raise ValueError(f"Input distribution '{input_distribution}' is not supported.")
    return inputs


def sample_gaussian_inputs(num_samples: int, num_features: int) -> torch.Tensor:
    inputs = torch.randn(size=(num_samples, num_features))
    return inputs


def sample_uniform_inputs(num_samples: int, num_features: int) -> torch.Tensor:
    inputs = torch.rand(size=(num_samples, num_features))  # U[0, 1]
    inputs = (inputs * 2) - 1  # U[-1, 1]
    return inputs


def add_noise(targets: torch.Tensor, *, noise_distribution: str) -> torch.Tensor:
    if noise_distribution == "gaussian":
        noise = np.random.normal(size=targets.shape)
    elif noise_distribution == "laplacian":
        noise = np.random.laplace(size=targets.shape)
    elif noise_distribution == "shifted_exponential":
        noise = np.random.exponential(size=targets.shape)
    else:
        raise ValueError(f"Noise distribution '{noise_distribution}' is not supported.")
    noise = torch.from_numpy(noise).float()
    targets = targets + noise
    return targets
