from typing import Tuple, Callable, List, Dict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from lib.models import LinearModel
from lib.losses import get_criterion
from lib.utils.bias import compute_bias
from lib.datasets import SyntheticDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_val_split(dataset: Dataset, val_percentage: float) -> Tuple[Dataset]:
    len_val = int(len(dataset) * val_percentage)
    len_train = len(dataset) - len_val
    train_dataset, val_dataset = random_split(dataset, [len_train, len_val])
    return train_dataset, val_dataset


def train(
    *,
    model: LinearModel,
    criterion: Callable,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
) -> float:
    model.train()
    cum_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (inputs, targets) in pbar:
        optim.zero_grad()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        preds = model(inputs)
        loss = criterion(inputs, preds, targets)
        loss.backward()
        optim.step()

        cum_loss += loss.item()
        avg_loss = cum_loss / (idx + 1)
        pbar.set_description(f"Loss: {avg_loss}")
    return avg_loss


def eval(*, model: LinearModel, criterion: Callable, dataloader: DataLoader) -> float:
    model.eval()
    cum_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for idx, (inputs, targets) in pbar:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(inputs)
            loss = criterion(inputs, preds, targets)

            cum_loss += loss.item()
            avg_loss = cum_loss / (idx + 1)
            pbar.set_description(f"Loss: {avg_loss}")
    return avg_loss


def experiment(
    num_samples: int,
    num_features: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    l2_regularization: float,
    loss_criterion: str,
    noise_distribution: str,
    source_input_distribution: str = "uniform",
    target_input_distribution: str = "gaussian",
):
    # Create Datasets
    source_dataset = SyntheticDataset(
        num_samples=num_samples,
        num_features=num_features,
        noise_distribution=noise_distribution,
        input_distribution=source_input_distribution,
    )
    train_dataset, val_dataset = train_val_split(source_dataset, val_percentage=0.1)
    target_dataset = SyntheticDataset(
        num_samples=num_samples,
        num_features=num_features,
        noise_distribution=noise_distribution,
        input_distribution=target_input_distribution,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
    # Init model
    model = LinearModel(in_features=num_features, out_features=1)
    model.to(DEVICE)
    optim = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=l2_regularization
    )
    criterion = get_criterion(loss_criterion)
    mse_criterion = get_criterion("squared_loss")
    # Train
    train_history = []
    val_history = []
    best_loss = 1e10
    for epoch_idx in range(num_epochs):
        print(f"Epoch {epoch_idx}")
        train_loss = train(
            model=model, criterion=criterion, dataloader=train_dataloader, optim=optim
        )
        val_loss = eval(model=model, criterion=criterion, dataloader=val_dataloader)
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss <= best_loss:
            torch.save(model.state_dict(), "./best.pth")
            best_loss = val_loss
    # Compute bias of the model
    model.load_state_dict(torch.load("./best.pth"))
    if loss_criterion == "hsic":
        bias = compute_bias(model, train_dataloader)
        model.update_bias(bias)
    # Evaluate MSE
    train_mse = eval(model=model, criterion=mse_criterion, dataloader=train_dataloader)
    val_mse = eval(model=model, criterion=mse_criterion, dataloader=val_dataloader)
    target_mse = eval(
        model=model, criterion=mse_criterion, dataloader=target_dataloader
    )

    results = {
        "train_loss": train_history,
        "val_loss": val_history,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "target_mse": target_mse,
    }

    return results


def plot_results(dataset_sizes: List[int], results: List[Dict]):
    plt.ion()
    f, ax = plt.subplots()
    ax.set_xscale("log")

    train_loss = [exp_res["train_mse"] for exp_res in results]
    val_loss = [exp_res["val_mse"] for exp_res in results]
    target_loss = [exp_res["target_mse"] for exp_res in results]

    ax.plot(dataset_sizes, train_loss, label="train")
    ax.plot(dataset_sizes, val_loss, label="val")
    ax.plot(dataset_sizes, target_loss, label="target")
    ax.legend()

    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Train Dataset Size")


def main(
    loss_criterion: str = "hsic",
    noise_distribution: str = "gaussian",
    num_features: int = 100,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    l2_regularization: float = 1e-4,
    log_samples_start: int = 5,
    log_samples_end: int = 13,
):
    dataset_sizes = [2 ** i for i in range(log_samples_start, log_samples_end + 1)]

    results = []
    for num_samples in dataset_sizes:
        exp_results = experiment(
            num_samples=num_samples,
            num_features=num_features,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            loss_criterion=loss_criterion,
            noise_distribution=noise_distribution,
        )
        results.append(exp_results)

    plot_results(dataset_sizes, results)


if __name__ == "__main__":
    main()