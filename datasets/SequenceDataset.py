from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple
import torch

class SequenceDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Stacks consecutive samples into sequences of length seq_len.
    """
    def __init__(
        self,
        sampler: Dataset[Tuple[Tensor, Tensor]],
        seq_len: int
    ) -> None:
        """
        Args:
            sampler: Dataset yielding (points, target) per index.
            seq_len: Number of consecutive frames.
        """
        self.sampler = sampler
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Last valid start is len(sampler) - (seq_len - 1)
        return len(self.sampler) - (self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            pts_seq: Tensor (seq_len, N, 3)
            tgt_seq: Tensor (seq_len, 3)
        """
        pts_list: List[Tensor] = []
        tgt_list: List[Tensor] = []
        for offset in range(self.seq_len):
            pts, tgt = self.sampler[idx + offset]
            pts_list.append(pts)
            tgt_list.append(tgt)
        pts_seq = torch.stack(pts_list, dim=0)
        tgt_seq = torch.stack(tgt_list, dim=0)
        return pts_seq, tgt_seq