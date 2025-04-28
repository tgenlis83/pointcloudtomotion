import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from datasets.utils import load_calibration, load_poses

class SemanticKITTIDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    PyTorch Dataset for loading sequential LiDAR scans and ground-truth translations
    from the KITTI odometry dataset.

    Each sample returns a single scan (N x 3) and the 3D translation to the next frame.
    """
    def __init__(
        self,
        data_path: Union[str, Path],
        sample_size: int,
        seq_len: int = 1,
        split_start: float = 0.0,
        split_end: float = 1.0,
        deterministic: bool = False,
        noise_amount: float = 0.15
    ) -> None:
        """
        Args:
            data_path: Directory containing 'calib.txt', 'poses.txt', and 'velodyne/*.bin'.
            sample_size: Minimum number of points per scan (unused loader-side).
            seq_len: Number of consecutive frames (currently only used to limit file list).
            split_start: Fractional start index for train/val split.
            split_end: Fractional end index for train/val split.
            deterministic: If False, apply random jitter to point clouds.
            noise_amount: Amount of noise to add to point clouds (if deterministic is False).
        """
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.deterministic = deterministic
        self.min_points = sample_size
        self.noise_amount = noise_amount

        # Load calibration and pose transforms
        calib_file = self.data_path / 'calib.txt'
        pose_file = self.data_path / 'poses.txt'
        self.calib = load_calibration(calib_file)
        self.poses = load_poses(pose_file, self.calib)  # shape: (M,4,4)

        # List of sorted LiDAR binary files
        lidar_dir = self.data_path / 'velodyne'
        all_bins: List[Path] = sorted(lidar_dir.glob('*.bin'))

        # Determine valid start indices based on seq_len and split
        total = len(all_bins) - seq_len
        start = int(np.floor(split_start * total))
        end = min(int(np.ceil(split_end * total)), total)
        # Keep extra frames for full sequences
        self.lidar_files: List[Path] = all_bins[start : end + seq_len]
        self.offset = start

        # Precompute relative translations (i -> i+1)
        # The number of valid samples is the total number of LiDAR files minus the sequence length
        num_samples = len(self.lidar_files) - seq_len

        # Initialize a tensor to store the relative translations for each sample
        # Shape: (num_samples, 3), where each row is a 3D translation vector
        self.translations = torch.empty((num_samples, 3), dtype=torch.float32)

        # Iterate over all valid starting indices for sequences
        for i in range(num_samples):
            # Map the local index to the global index in the pose list
            idx_world = i + self.offset

            # Retrieve the pose of the current frame (i) and the next frame (i+1)
            pose_i = self.poses[idx_world]
            pose_j = self.poses[idx_world + 1]

            # Compute the relative transformation from frame i to frame i+1
            # This is done by multiplying the inverse of pose_i with pose_j
            rel = torch.inverse(pose_i) @ pose_j

            # Extract the translation component (first 3 elements of the last column)
            self.translations[i] = rel[:3, 3]

    def __len__(self) -> int:
        """Number of available starting frames for sequences."""
        return len(self.lidar_files) - self.seq_len

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Load a single scan and its next-frame translation.

        Returns:
            pts: Tensor of shape (N, 3)
            tgt: Tensor of shape (3,), the translation vector to next frame
        """
        bin_path = self.lidar_files[index]
        # Load and reshape point cloud (N,4), discard intensity
        raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        pts = torch.from_numpy(raw[:, :3])  # (N,3)

        # Optionally jitter for data augmentation
        if not self.deterministic:
            noise = torch.randn_like(pts) * self.noise_amount
            pts = pts + noise

        target = self.translations[index]
        return pts.float(), target