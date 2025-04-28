import torch
from pathlib import Path
from typing import Dict, List
from torch import Tensor
import numpy as np

def load_calibration(calib_path: Path) -> Dict[str, Tensor]:
    """
    Parse a KITTI calibration file into a dict of 4x4 transformation matrices.
    Args:
        calib_path: Path to calibration text file.
    Returns:
        Dict mapping calibration keys to 4x4 torch.FloatTensor matrices.
    """
    calib: Dict[str, Tensor] = {}
    for line in calib_path.read_text().splitlines():
        key, values = line.split(':', maxsplit=1)
        nums = [float(x) for x in values.strip().split()]
        # Construct 4x4 matrix (row-major)
        P = np.eye(4, dtype=np.float32)
        P[0, :4] = nums[0:4]
        P[1, :4] = nums[4:8]
        P[2, :4] = nums[8:12]
        calib[key] = torch.from_numpy(P)
    return calib


def load_poses(poses_path: Path, calib: Dict[str, Tensor]) -> Tensor:
    """
    Parse KITTI ego-motion poses and transform into a global frame using calibration.
    Args:
        poses_path: Path to poses text file (each line has 12 floats).
        calib: Calibration dict containing 'Tr' key for sensor-to-world transform.
    Returns:
        Tensor of shape (N, 4, 4) giving world-frame poses.
    """
    # Sensor-to-world and its inverse
    Tr = calib['Tr'].numpy()              # shape (4,4)
    Tr_inv = np.linalg.inv(Tr).astype(np.float32)

    pose_list: List[np.ndarray] = []
    for line in poses_path.read_text().splitlines():
        nums = [float(x) for x in line.strip().split()]
        T = np.eye(4, dtype=np.float32)
        T[0, :4] = nums[0:4]
        T[1, :4] = nums[4:8]
        T[2, :4] = nums[8:12]
        # Convert to world frame: Tr_inv @ T @ Tr
        pose_world = Tr_inv @ T @ Tr
        pose_list.append(pose_world)

    # Stack into (N,4,4)
    poses_np = np.stack(pose_list, axis=0)
    return torch.from_numpy(poses_np)