from torch import nn
from torch import Tensor
import torch

class PointNetEncoder(nn.Module):
    """
    Encodes a single LiDAR point-cloud frame into a fixed-length feature vector.

    Architecture:
        - Shared MLP via 1D Convolutions across points
        - Global max pooling to aggregate into feature vector

    Args:
        in_channels: Number of input point channels (e.g., 3 for xyz).
        feat_dim: Dimension of the output feature vector per frame.
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 128) -> None:
        super().__init__()
        # Shared MLP: in_channels -> 64 -> 128 -> feat_dim
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feat_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a batch of single frames.

        Args:
            x: Input point clouds of shape [B, N, C]
        Returns:
            feats: Aggregated features [B, feat_dim]
        """
        # Permute to [B, C, N] for Conv1d
        x = x.permute(0, 2, 1)
        # Apply pointwise MLP
        feats = self.mlp(x)
        # Global max-pooling over points
        feats, _ = torch.max(feats, dim=2)
        return feats