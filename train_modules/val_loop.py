import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.combined_loss import combined_loss
from typing import Tuple


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    random_mean: torch.Tensor,
    random_std: torch.Tensor
) -> Tuple[float, float]:
    """
    Evaluates model and random baseline on validation set.
    Returns avg_val_loss and avg_random_loss.
    """
    model.eval()
    total_val, total_rand = 0.0, 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validate", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            true = targets[:, -1, :]
            total_val += combined_loss(preds, true).item()
            rand_preds = torch.randn_like(true) * random_std.to(device) + random_mean.to(device)
            total_rand += combined_loss(rand_preds, true).item()
    return total_val / len(loader), total_rand / len(loader)
