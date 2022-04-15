from typing import Callable

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def train(
    *,
    model: nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    use_pbar: bool = False,
) -> float:
    model.train()
    device = next(model.parameters()).device
    cum_loss = 0
    if use_pbar:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)
    for idx, (inputs, targets) in pbar:
        optim.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = model(inputs)
        inputs = torch.flatten(inputs, start_dim=1)
        loss = criterion(inputs, preds, targets)
        loss.backward()
        optim.step()

        cum_loss += loss.item()
        avg_loss = cum_loss / (idx + 1)
        if use_pbar:
            pbar.set_description(f"Loss: {avg_loss}")
    return avg_loss


def eval(
    *,
    model: nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    use_pbar: bool = False,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    cum_loss = 0
    if use_pbar:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        pbar = enumerate(dataloader)
    with torch.no_grad():
        for idx, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            inputs = torch.flatten(inputs, start_dim=1)
            loss = criterion(inputs, preds, targets)

            cum_loss += loss.item()
            avg_loss = cum_loss / (idx + 1)
            if use_pbar:
                pbar.set_description(f"Loss: {avg_loss}")
    return avg_loss
