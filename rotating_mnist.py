from typing import List, Dict

import torch
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from lib.datasets import MNIST
from lib.models import CNN, MLP
from lib.losses import get_criterion
from lib.utils.trainer import train, eval
from lib.utils.metrics import compute_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def experiment(
    batch_size: int,
    learning_rate: float,
    loss_criterion: str,
    num_epochs: int,
    cnn: bool = False,
    mlp_layers: List[int] = [],
    verbose: bool = True,
    **kwargs,
) -> Dict:
    # Init model
    if cnn:
        if mlp_layers:
            raise ValueError(
                "Conflicting models. Either set `cnn` to false or `mlp_layers` to `[]`"
            )
        model = CNN()
    elif mlp_layers:
        model = MLP(in_features=28 * 28, layers=mlp_layers)
    else:
        raise ValueError(
            "No model parameters were provided, please provide values "
            "for either `cnn` OR `mlp_layers` parameters"
        )
    model.to(DEVICE)

    # Create Datasets
    source_dataset = MNIST(rotated=False, train=True)
    train_dataset, val_dataset, _ = random_split(
        source_dataset, [10_000, 1_000, len(source_dataset) - 11_000]
    )
    target_dataset = MNIST(rotated=True, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size)

    # Setup Optimizer
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    if loss_criterion in {"hsic", "squared_loss"}:
        target_transform = lambda x: torch.nn.functional.one_hot(x, num_classes=10)
    else:
        target_transform = None

    if loss_criterion == "hsic":
        criterion = get_criterion(
            loss_criterion, target_transform=target_transform, s_x=22, s_y=1
        )
    else:
        criterion = get_criterion(loss_criterion, target_transform=target_transform)

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

    # Compute Accuracy
    model.load_state_dict(torch.load("./best.pth"))
    train_accuracy = compute_accuracy(model, train_dataloader)
    val_accuracy = compute_accuracy(model, val_dataloader)
    target_accuracy = compute_accuracy(model, target_dataloader)

    results = {
        "train_history": train_history,
        "val_history": val_history,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "target_accuracy": target_accuracy,
    }

    return results


def multiple_trials(experiment_config: Dict, num_trials: int) -> Dict:
    results = []
    for i in tqdm(range(num_trials)):
        trial_results = experiment(**experiment_config)
        results.append(trial_results)

    train_accuracy = [trial["train_accuracy"] for trial in results]
    val_accuracy = [trial["val_accuracy"] for trial in results]
    target_accuracy = [trial["target_accuracy"] for trial in results]

    results = {
        "train": pd.Series(train_accuracy).rename(experiment_config["model_name"]),
        # "val": pd.Series(val_accuracy).rename(experiment_config["model_name"]),
        "target": pd.Series(target_accuracy).rename(experiment_config["model_name"]),
    }

    return results


def group_results(results: List[Dict]) -> pd.DataFrame:
    keys = results[0].keys()

    df_list = []
    for key in keys:
        df = pd.concat([exp_res[key] for exp_res in results], axis=1)
        df = (
            df.stack()
            .rename("Accuracy")
            .rename_axis(index=["exp", "model_name"])
            .reset_index()
        )
        df["model_name"] = df["model_name"].apply(lambda x: x + f"_{key}")
        df_list.append(df)
    df = pd.concat(df_list)

    return df


def plot_results(df: pd.DataFrame, title: str = ""):
    plt.ion()

    ax = sns.boxplot(x="Accuracy", y="model_name", hue="loss_criterion", data=df)
    ax.set(xscale="log")
    ax.set_title(title)
    ax.set_xscale("linear")


def main(
    num_trials: int = 20,
    num_epochs: int = 7,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    models = [
        {"model_name": "CNN", "cnn": True},
        # {"model_name": "MLP 2x256", "mlp_layers": [256, 256, 10]},
        # {"model_name": "MLP 2x524", "mlp_layers": [524, 524, 10]},
        # {"model_name": "MLP 2x1024", "mlp_layers": [1024, 1024, 10]},
        # {"model_name": "MLP 4x256", "mlp_layers": [256, 256, 256, 256, 10]},
        # {"model_name": "MLP 4x524", "mlp_layers": [524, 524, 524, 524, 10]},
        # {"model_name": "MLP 4x1024", "mlp_layers": [1024, 1024, 1024, 1024, 10]},
    ]

    data = []
    for loss_criterion in ["cross_entropy"]:  # ["hsic", "cross_entropy"]:
        results = []
        for model_config in models:
            experiment_config = {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss_criterion": loss_criterion,
            }
            experiment_config = {**experiment_config, **model_config}
            exp_results = multiple_trials(
                num_trials=num_trials, experiment_config=experiment_config
            )
            results.append(exp_results)
        results = group_results(results)
        results["loss_criterion"] = loss_criterion
        data.append(results)
    data = pd.concat(data)
    plot_results(data)
    results.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
