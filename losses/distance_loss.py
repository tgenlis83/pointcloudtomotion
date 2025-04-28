from torch import Tensor
import torch

def distance_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Computes the mean L2 distance between prediction and target vectors.

    Args:
        pred: Tensor of shape (B, D).
        target: Tensor of shape (B, D).
    Returns:
        Scalar tensor: mean L2 distance.
    """
    diff = pred - target
    # Norm over last dimension: shape (B,)
    dist = torch.norm(diff, p=2, dim=-1)
    return dist.mean()