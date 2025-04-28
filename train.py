import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.TrajectoryPredictor import TrajectoryPredictor
from data.config import Config
from utils import get_device
from train_modules.utils import create_dataloaders
from train_modules.train_loop import train_epoch
from train_modules.val_loop import validate_epoch
from plotting.CheckpointManager import CheckpointManager
from plotting.CheckpointManager import init_plot


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    random_mean: torch.Tensor,
    random_std: torch.Tensor,
    cfg: Config
) -> None:
    """
    Full training loop with live plotting and checkpoint management.
    """
    device = get_device()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    manager = CheckpointManager(cfg.checkpoint_dir, top_k=3)

    fig, ax, plot_state = init_plot(cfg.num_epochs)

    for epoch in range(1, cfg.num_epochs + 1):
        # Training and validation
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, rand_loss = validate_epoch(model, val_loader, device, random_mean, random_std)

        # Update histories
        plot_state.train_losses.append(train_loss)
        plot_state.val_losses.append(val_loss)
        plot_state.rand_losses.append(rand_loss)

        # Update plot lines
        x = plot_state.epochs[:len(plot_state.train_losses)]
        plot_state.train_line.set_data(x, plot_state.train_losses)
        plot_state.val_line.set_data(x, plot_state.val_losses)
        plot_state.rand_line.set_data(x, plot_state.rand_losses)
        ax.relim(); ax.autoscale_view()
        fig.canvas.draw(); plt.pause(0.1)

        # Save checkpoints
        manager.save(model, optimizer, epoch, train_loss, val_loss)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, rand={rand_loss:.4f}")

    plt.ioff(); plt.show()


def main() -> None:
    """
    Entry point: loads data, builds model, and starts training.
    """
    cfg = Config()
    train_loader, val_loader, random_mean, random_std = create_dataloaders(cfg)
    model = TrajectoryPredictor(
        in_channels=3,
        feat_dim=128,
        hidden_dim=256,
        num_layers=2,
        out_dim=3
    )
    train(model, train_loader, val_loader, random_mean, random_std, cfg)


if __name__ == "__main__":
    torch.backends.mps.benchmark = True
    main()
