import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

from datasets.SemanticKITTIDataset import SemanticKITTIDataset
from datasets.PointSamplerDataset import PointSamplerDataset
from datasets.SequenceDataset import SequenceDataset
from models.TrajectoryPredictor import TrajectoryPredictor
from losses.combined_loss import combined_loss
from utils import get_device


# Playback settings
PLAYBACK_FPS = 25
SEQ_LEN = 5       # number of past frames to stack
N_POINTS = int(3e4)  # points per cloud
BATCH_SIZE = 32   # inference batch size
CHECKPOINT_DIR = Path("checkpoints")
ANIM_PATH = Path("traj_anim.mp4")


def prepare_dataset(data_path: str, seq_len: int, n_points: int, split_ratio: float = 0.8) -> DataLoader:
    """
    Loads the SemanticKITTI dataset, splits into train/val,
    samples points, stacks sequences, and returns a DataLoader for inference.
    """
    # Full deterministic dataset for inference
    base_ds = SemanticKITTIDataset(
        data_path=data_path,
        sample_size=1e5,
        seq_len=seq_len,
        noise_amount=0.0,
        deterministic=True
    )

    # Split indices for validation subset
    total = len(base_ds)
    cut = int(split_ratio * total)
    val_idxs = np.arange(cut, total)
    val_subset = Subset(base_ds, val_idxs.tolist())

    # Sample fixed number of points per cloud
    sampled = PointSamplerDataset(
        base_dataset=val_subset,
        n_points=n_points,
        deterministic=True
    )

    # Stack previous frames into sequences
    seq_ds = SequenceDataset(
        sampler=sampled,
        seq_len=seq_len
    )

    # Collate function (default)
    def collate_fn(batch):
        return torch.utils.data._utils.collate.default_collate(batch)

    # DataLoader for inference
    return DataLoader(
        seq_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )


def load_model(checkpoint_dir: Path, device: torch.device) -> nn.Module:
    """
    Instantiates the TrajectoryPredictor, loads the most recent checkpoint,
    and sets model to eval mode on the given device.
    """
    model = TrajectoryPredictor(
        in_channels=3,
        feat_dim=128,
        hidden_dim=256,
        num_layers=2,
        out_dim=3
    ).to(device)

    # Find the checkpoint with the lowest validation loss
    ckpts = sorted(
        checkpoint_dir.glob("val_*.pt"),
        key=lambda p: float(p.stem.split('_')[1])
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    best_ckpt = ckpts[0]
    print(f"Loading checkpoint {best_ckpt}")
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def compute_error_metrics(
    predictions: np.ndarray,
    ground_truths: np.ndarray
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculates per-sample combined loss, angle error (deg), and length error.
    """
    # Combined loss
    losses = []
    for pred, gt in zip(predictions, ground_truths):
        losses.append(combined_loss(
            torch.from_numpy(pred), torch.from_numpy(gt)
        ).item())

    # Angle error in degrees
    def angle_error(v1: np.ndarray, v2: np.ndarray) -> float:
        eps = 1e-8
        n1 = v1 / (np.linalg.norm(v1) + eps)
        n2 = v2 / (np.linalg.norm(v2) + eps)
        cos_t = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_t))

    angle_err = [angle_error(p, g) for p, g in zip(predictions, ground_truths)]

    # Length error (absolute difference)
    length_err = [abs(np.linalg.norm(p) - np.linalg.norm(g))
                  for p, g in zip(predictions, ground_truths)]

    return losses, angle_err, length_err


def summarize_stats(values: List[float]) -> Tuple[float, float, float]:
    """
    Returns (min, max, average) of a list of floats.
    """
    return float(np.min(values)), float(np.max(values)), float(np.mean(values))


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs model inference over the DataLoader,
    collecting predictions, ground truths, and last-point clouds.
    """
    preds, gts, pcds = [], [], []

    with torch.no_grad():
        for pts_seq, tgt in tqdm(loader, desc="Predicting"):
            pts_seq = pts_seq.to(device)
            tgt_last = tgt[:, -1, :].to(device)
            output = model(pts_seq)

            preds.append(output.cpu().numpy())
            gts.append(tgt_last.cpu().numpy())
            pcds.append(pts_seq[:, -1].cpu().numpy())

    return (
        np.vstack(preds),
        np.vstack(gts),
        np.vstack(pcds)
    )


def create_animation(
    pcd_sequence: np.ndarray,
    gt_vectors: np.ndarray,
    pred_vectors: np.ndarray,
    save_path: Path,
    fps: int = PLAYBACK_FPS
) -> None:
    """
    Generates and saves a dual-view 3D animation of point clouds with
    ground-truth and predicted motion arrows.
    """
    frames = len(pcd_sequence)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    scatter1 = ax1.scatter([], [], [], s=1)
    scatter2 = ax2.scatter([], [], [], s=1)

    def init():
        for ax in (ax1, ax2):
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_zlim(-2, 2)
        ax1.view_init(elev=90, azim=-90)
        ax2.view_init(elev=20, azim=120)
        return scatter1, scatter2

    def animate(i):
        # Update point clouds
        xs, ys, zs = pcd_sequence[i].T
        scatter1._offsets3d = (xs, ys, zs)
        scatter2._offsets3d = (xs, ys, zs)

        # Remove old arrows
        for artist in getattr(animate, 'artists', []):
            artist.remove()
        animate.artists = []

        # Draw arrows: ground truth (red) and prediction (blue)
        for ax, vec_gt, vec_pr in ((ax1, gt_vectors[i], pred_vectors[i]),
                                   (ax2, gt_vectors[i], pred_vectors[i])):
            arrow_gt = ax.quiver(0, 0, 0, *vec_gt, length=4, normalize=True)
            arrow_pr = ax.quiver(0, 0, 0, *vec_pr, length=4, normalize=True)
            animate.artists.extend([arrow_gt, arrow_pr])

        return (scatter1, scatter2) + tuple(animate.artists)

    # Save animation with progress bar
    with tqdm(total=frames, desc="Rendering animation") as pbar:
        def progress(frame, *args):
            pbar.update()

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=frames, interval=1000 / fps,
            blit=False
        )
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(str(save_path), writer=writer, progress_callback=progress)
    plt.close(fig)


def main():
    """
    Entry point: prepares data, loads model, runs inference,
    computes statistics, and visualizes results.
    """
    device = get_device()
    inf_loader = prepare_dataset(
        data_path="SemanticKITTI_00/",
        seq_len=SEQ_LEN,
        n_points=N_POINTS
    )
    model = load_model(CHECKPOINT_DIR, device)

    preds, gts, pcds = run_inference(model, inf_loader, device)
    losses, ang_err, len_err = compute_error_metrics(preds, gts)

    # Print statistics
    for name, values in ("Loss", losses), ("Angle Error", ang_err), ("Length Error", len_err):
        mn, mx, avg = summarize_stats(values)
        print(f"{name} → min: {mn:.4f}, max: {mx:.4f}, avg: {avg:.4f}")

    # Create and save animation
    create_animation(pcds, gts, preds, ANIM_PATH)
    print(f"Animation saved to {ANIM_PATH}")


if __name__ == "__main__":
    main()
