from __future__ import annotations
from typing import Tuple

from torch import Tensor, nn
from models.PointNetEncoder import PointNetEncoder

class TrajectoryPredictor(nn.Module):
    """
    Predicts next-frame translation from a sequence of LiDAR frames.

    Pipeline:
        1. Encode each frame via PointNetEncoder
        2. Model temporal dependencies with an LSTM
        3. Decode the final hidden state to a translation vector

    Args:
        in_channels: Point channel dimensions (C).
        feat_dim: Feature dimension from encoder.
        hidden_dim: Hidden size for LSTM.
        num_layers: Number of LSTM layers.
        out_dim: Output dimension (e.g., 3 for xyz translation).
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        # Frame encoder
        self.encoder: PointNetEncoder = PointNetEncoder(
            in_channels=in_channels,
            feat_dim=feat_dim,
        )
        # Sequence model
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # Decoder from final LSTM hidden state to translation
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pc_seq: Tensor) -> Tensor:
        """
        Forward pass for a batch of LiDAR sequences.

        Args:
            pc_seq: Tensor of shape [B, T, N, C]
                B: batch size
                T: sequence length (frames)
                N: points per frame
                C: point channels

        Returns:
            preds: Predicted translations of shape [B, out_dim]
        """
        B, T, N, C = pc_seq.shape
        # Collapse batch & time for encoding
        flat_seq: Tensor = pc_seq.view(B * T, N, C)
        # Encode each frame → [B*T, feat_dim]
        feats: Tensor = self.encoder(flat_seq)
        # Reshape back to sequence form → [B, T, feat_dim]
        feats = feats.view(B, T, -1)
        # LSTM over time → outputs [B, T, hidden_dim]
        lstm_out, _ = self.lstm(feats)
        # Take final time-step hidden state
        final_hidden: Tensor = lstm_out[:, -1, :]
        # Decode to translation → [B, out_dim]
        preds: Tensor = self.decoder(final_hidden)
        return preds
