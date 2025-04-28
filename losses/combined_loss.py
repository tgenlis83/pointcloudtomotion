from __future__ import annotations
from typing import Optional, Tuple

from torch import Tensor
from losses.distance_loss import distance_loss
from losses.direction_loss import direction_loss


def combined_loss(
    pred: Tensor,
    target: Tensor,
    weights: Optional[Tuple[float, float]] = None
) -> Tensor:
    """
    Combined loss: average of distance and direction losses, with optional weighting.

    Args:
        pred: Predicted tensor of shape (B, D).
        target: Ground-truth tensor of shape (B, D).
        weights: Optional (w_dist, w_dir) weights for distance and direction losses.
                 If None, equal weighting (0.5, 0.5) is used.

    Returns:
        Scalar tensor: weighted sum of distance_loss and direction_loss.
    """
    # Compute individual components
    d_loss = distance_loss(pred, target)
    dir_loss = direction_loss(pred, target)

    # Default equal weights
    w_dist, w_dir = weights if weights is not None else (0.5, 0.5)

    # Combine and return
    loss = w_dist * d_loss + w_dir * dir_loss
    return loss
