from typing import Tuple

from torch.utils.data import random_split, Dataset

from .synthetic import SyntheticDataset


def train_val_split(dataset: Dataset, val_percentage: float) -> Tuple[Dataset]:
    len_val = int(len(dataset) * val_percentage)
    len_train = len(dataset) - len_val
    train_dataset, val_dataset = random_split(dataset, [len_train, len_val])
    return train_dataset, val_dataset
