import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.combined_loss import combined_loss

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Performs one training epoch. Returns average loss.
    """
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = combined_loss(preds, targets[:, -1, :])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)