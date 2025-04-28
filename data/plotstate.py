from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List
from dataclasses import field

@dataclass
class PlotState:
    """
    Tracks epochs, loss histories, and matplotlib Line2D handles.
    """
    epochs: List[int]
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    rand_losses: List[float] = field(default_factory=list)
    train_line: plt.Line2D = None  # type: ignore
    val_line: plt.Line2D = None    # type: ignore
    rand_line: plt.Line2D = None   # type: ignore