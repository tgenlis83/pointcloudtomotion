from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset
import torch

class PointSamplerDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Randomly samples a fixed number of points from each scan.
    """
    def __init__(
        self,
        base_dataset: Dataset[Tuple[Tensor, Tensor]],
        n_points: int,
        deterministic: bool = False,
    ) -> None:
        """
        Args:
            base_dataset: Underlying dataset producing (points, target).
            n_points: Number of points to sample per scan.
            deterministic: If True, take first n_points; else random subset.
        """
        self.base = base_dataset
        self.n_points = n_points
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        pts, tgt = self.base[idx]
        num = pts.shape[0]
        if num < self.n_points:
            msg = f"Scan #{idx} has only {num} points < required {self.n_points}"
            raise ValueError(msg)

        if self.deterministic:
            indices = torch.arange(self.n_points)
        else:
            indices = torch.randperm(num)[: self.n_points]

        return pts[indices], tgt
