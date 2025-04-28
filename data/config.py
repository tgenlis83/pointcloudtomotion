from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    """
    Configuration for training: hyperparameters and paths.
    """
    data_path: Path = Path("SemanticKITTI_00/")
    seq_len: int = 5
    n_points: int = 30_000
    batch_size: int = 16
    num_epochs: int = 50
    train_split: float = 0.8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    checkpoint_dir: Path = Path("checkpoints")
    angle_threshold: float = 3.0
    noise_amount: float = 0.15