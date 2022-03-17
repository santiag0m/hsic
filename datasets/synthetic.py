from typing import Tuple

import torch
from torch.utils.data import TensorDataset
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.distributions.exponential import Exponential


class SyntheticDataset(TensorDataset):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
        noise_distribution: str,
        input_distribution: str,
    ):
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
        targets = add_gaussian_noise(targets)
    elif noise_distribution == "laplacian":
        targets = add_laplacian_noise(targets)
    elif noise_distribution == "shifted_exponential":
        targets = add_shifted_exponential_noise(targets)
    else:
        raise ValueError(f"Noise distribution '{noise_distribution}' is not supported.")
    return targets


def add_gaussian_noise(targets: torch.Tensor, sigma: float = 1) -> torch.Tensor:
    loc = torch.zeros_like(targets)
    scale = sigma * torch.ones_like(targets)
    distribution = Normal(loc, scale)
    noise = distribution.sample()
    return targets + noise


def add_laplacian_noise(targets: torch.Tensor, b: float = 1) -> torch.Tensor:
    loc = torch.zeros_like(targets)
    scale = b * torch.ones_like(targets)
    distribution = Laplace(loc, scale)
    noise = distribution.sample()
    return targets + noise


def add_shifted_exponential_noise(
    targets: torch.Tensor, rate: float = 1
) -> torch.Tensor:
    rate = rate * torch.ones_like(targets)
    distribution = Exponential(rate)
    noise = 1 - distribution.sample()
    return targets + noise
