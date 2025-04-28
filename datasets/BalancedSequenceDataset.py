from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets.SequenceDataset import SequenceDataset


class BalancedSequenceDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Oversamples a SequenceDataset to balance straight vs. turning motions.

    Ensures ~50/50 split based on angle threshold.
    """
    def __init__(
        self,
        seq_dataset: SequenceDataset,
        angle_threshold: float = 3.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            seq_dataset: Underlying SequenceDataset.
            angle_threshold: Degrees threshold to separate straight vs. turn.
            seed: Optional RNG seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.seq_ds = seq_dataset
        # Extract final-step translations
        n = len(self.seq_ds)
        all_t: Tensor = torch.empty((n, 3), dtype=torch.float32)
        for i in tqdm(range(n), desc="Gathering translations"):
            _, tgt_seq = self.seq_ds[i]
            all_t[i] = tgt_seq[-1]

        dx = all_t[:, 0].numpy()
        dy = all_t[:, 1].numpy()
        angles = np.abs(np.arctan2(dy, dx))  # radians
        thresh = np.deg2rad(angle_threshold)

        # Partition indices
        idx_straight = np.where(angles <= thresh)[0].tolist()
        idx_turning  = np.where(angles >  thresh)[0].tolist()
        count_s, count_t = len(idx_straight), len(idx_turning)
        print(f"Balancing: straight={count_s}, turning={count_t}")

        # Oversample minority
        max_count = max(count_s, count_t)
        if count_s < max_count:
            idx_straight = np.random.choice(idx_straight, max_count, replace=True).tolist()
        if count_t < max_count:
            idx_turning  = np.random.choice(idx_turning,  max_count, replace=True).tolist()

        # Interleave and shuffle
        combined = idx_straight + idx_turning
        np.random.shuffle(combined)
        self.indices = combined

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = self.indices[idx]
        return self.seq_ds[start]
