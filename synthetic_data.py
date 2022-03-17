from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lib.models import LinearModel
from lib.losses import get_criterion
from lib.utils.bias import compute_bias
from lib.utils.trainer import train, eval
from lib.datasets import SyntheticDataset, train_val_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    verbose: bool = False,
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
        if verbose:
            print(f"Epoch {epoch_idx}")
        train_loss = train(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            optim=optim,
            use_pbar=verbose,
        )
        val_loss = eval(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            use_pbar=verbose,
        )
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
    train_mse = eval(
        model=model,
        criterion=mse_criterion,
        dataloader=train_dataloader,
        use_pbar=verbose,
    )
    val_mse = eval(
        model=model,
        criterion=mse_criterion,
        dataloader=val_dataloader,
        use_pbar=verbose,
    )
    target_mse = eval(
        model=model,
        criterion=mse_criterion,
        dataloader=target_dataloader,
        use_pbar=verbose,
    )

    results = {
        "train_history": train_history,
        "val_history": val_history,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "target_mse": target_mse,
    }

    return results


def multiple_trials(experiment_config: Dict, num_trials: int) -> Dict:
    results = []
    for i in tqdm(range(num_trials)):
        trial_results = experiment(**experiment_config)
        results.append(trial_results)

    train_mse = [trial["train_mse"] for trial in results]
    val_mse = [trial["val_mse"] for trial in results]
    target_mse = [trial["target_mse"] for trial in results]

    results = {
        "train_mse_mean": np.mean(train_mse),
        "train_mse_std": np.std(train_mse),
        "val_mse_mean": np.mean(val_mse),
        "val_mse_std": np.std(val_mse),
        "target_mse_mean": np.mean(target_mse),
        "target_mse_std": np.std(target_mse),
    }

    return results


def plot_results(dataset_sizes: List[int], results: List[Dict]):
    plt.ion()
    f, ax = plt.subplots()
    ax.set_xscale("log")

    train_loss = np.array([exp_res["train_mse_mean"] for exp_res in results])
    val_loss = np.array([exp_res["val_mse_mean"] for exp_res in results])
    target_loss = np.array([exp_res["target_mse_mean"] for exp_res in results])

    train_std = np.array([exp_res["train_mse_std"] for exp_res in results])
    val_std = np.array([exp_res["val_mse_std"] for exp_res in results])
    target_std = np.array([exp_res["target_mse_std"] for exp_res in results])

    [p] = ax.plot(dataset_sizes, train_loss, label="train")
    ax.fill_between(
        dataset_sizes,
        train_loss - train_std,
        train_loss + train_std,
        color=p.get_color(),
        alpha=0.5,
    )
    [p] = ax.plot(dataset_sizes, val_loss, label="val")
    ax.fill_between(
        dataset_sizes,
        val_loss - val_std,
        val_loss + val_std,
        color=p.get_color(),
        alpha=0.5,
    )
    [p] = ax.plot(dataset_sizes, target_loss, label="target")
    ax.fill_between(
        dataset_sizes,
        target_loss - target_std,
        target_loss + target_std,
        color=p.get_color(),
        alpha=0.5,
    )
    ax.legend()

    ax.set_ylabel("Mean Squared Error")
    ax.set_xlabel("Train Dataset Size")

    ax.grid(which="both")


def main(
    num_trials: int = 20,
    loss_criterion: str = "hsic",
    noise_distribution: str = "gaussian",
    num_features: int = 100,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    l2_regularization: float = 1e-3,
    log_samples_start: int = 5,
    log_samples_end: int = 13,
):
    dataset_sizes = [2 ** i for i in range(log_samples_start, log_samples_end + 1)]

    results = []
    for num_samples in dataset_sizes:
        experiment_config = {
            "num_samples": num_samples,
            "num_features": num_features,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization,
            "loss_criterion": loss_criterion,
            "noise_distribution": noise_distribution,
        }
        exp_results = multiple_trials(
            num_trials=num_trials, experiment_config=experiment_config
        )
        results.append(exp_results)

    plot_results(dataset_sizes, results)


if __name__ == "__main__":
    main()
