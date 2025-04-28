from data.config import Config
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import torch
from datasets.SemanticKITTIDataset import SemanticKITTIDataset
from datasets.PointSamplerDataset import PointSamplerDataset
from datasets.SequenceDataset import SequenceDataset
from datasets.BalancedSequenceDataset import BalancedSequenceDataset

def create_dataloaders(
    cfg: Config
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Prepares train and validation DataLoaders along with translation stats
    for the random baseline.
    Returns:
        train_loader, val_loader, random_mean, random_std
    """
    # Load datasets
    full_train = SemanticKITTIDataset(
        str(cfg.data_path), sample_size=1e5, seq_len=cfg.seq_len, noise_amount=cfg.noise_amount, deterministic=False)
    full_val   = SemanticKITTIDataset(
        str(cfg.data_path), sample_size=1e5, seq_len=cfg.seq_len, noise_amount=cfg.noise_amount, deterministic=True)

    # Baseline statistics
    random_mean = full_train.translations.mean(dim=0)
    random_std  = full_train.translations.std(dim=0)

    # Split indices
    n_train = int(cfg.train_split * len(full_train))
    n_val   = len(full_train) - n_train
    train_sub, _ = random_split(full_train, [n_train, n_val])
    _, val_sub   = random_split(full_val,   [n_train, n_val])

    def pipeline(ds, deterministic: bool):
        points = PointSamplerDataset(ds, n_points=cfg.n_points, deterministic=deterministic)
        seq    = SequenceDataset(points, seq_len=cfg.seq_len)
        bls = BalancedSequenceDataset(seq, angle_threshold=cfg.angle_threshold)
        
        return bls

    train_ds = pipeline(train_sub, deterministic=False)
    val_ds   = pipeline(val_sub,   deterministic=True)

    # Create loaders
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=11,
        persistent_workers=True,
        pin_memory=(torch.cuda.is_available()),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, random_mean, random_std