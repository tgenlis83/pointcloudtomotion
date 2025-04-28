from torch import Tensor
from torch.nn import functional as F

def direction_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Computes mean directional loss based on cosine similarity.

    Args:
        pred: Tensor of shape (B, D).
        target: Tensor of shape (B, D).
    Returns:
        Scalar tensor: mean direction loss = 1 - cosine_similarity.
    """
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    # Convert similarity to loss (1 for opposite, 0 for identical)
    return (1.0 - cos_sim).mean()