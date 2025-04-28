from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple
import torch
from tqdm import tqdm
from torch import optim
from pathlib import Path
import matplotlib.pyplot as plt
from data.plotstate import PlotState

class CheckpointManager:
    """
    Maintains checkpoints: latest and top-k best by validation loss.
    """
    def __init__(self, directory: Path, top_k: int = 3):
        self.directory = directory
        self.top_k = top_k
        self.best_ckpts: List[Tuple[float, Path]] = []
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self,
             model: nn.Module,
             optimizer: optim.Optimizer,
             epoch: int,
             train_loss: float,
             val_loss: float) -> None:
        """
        Save latest and update top-k best checkpoints.
        """
        # Common checkpoint dict
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        # Save latest
        latest_path = self.directory / 'latest.pt'
        torch.save(ckpt, latest_path)

        # Save candidate with val in filename
        candidate = self.directory / f"val_{val_loss:.4f}_epoch_{epoch}.pt"
        torch.save(ckpt, candidate)

        # Update best list
        self.best_ckpts.append((val_loss, candidate))
        self.best_ckpts.sort(key=lambda x: x[0])
        # Remove excess
        while len(self.best_ckpts) > self.top_k:
            _, worst_path = self.best_ckpts.pop(-1)
            if worst_path.exists():
                worst_path.unlink()

def init_plot(total_epochs: int) -> Tuple[plt.Figure, plt.Axes, PlotState]:
    """
    Initializes an interactive matplotlib plot for tracking losses.
    Returns the figure, axes, and a PlotState object.
    """
    plt.ion()
    fig, ax = plt.subplots()
    epochs = list(range(1, total_epochs + 1))

    # Draw empty lines for each curve
    train_line, = ax.plot([], [], '-o', label='Train Loss')
    val_line,   = ax.plot([], [], '-o', label='Val Loss')
    rand_line,  = ax.plot([], [], '-o', label='Random Baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.show()

    state = PlotState(
        epochs=epochs,
        train_line=train_line,
        val_line=val_line,
        rand_line=rand_line
    )
    return fig, ax, state