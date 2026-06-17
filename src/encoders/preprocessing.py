import numpy as np
import torch

# robosuite images are (H, W, C) uint8; PyTorch expects (C, H, W) float in [0, 1]

def preprocess_image(obs: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 -> (C, H, W) float32 in [0, 1]."""
    return torch.from_numpy(obs).float().permute(2, 0, 1).div(255.0)


def preprocess_batch(obs: np.ndarray) -> torch.Tensor:
    """(B, H, W, C) uint8 -> (B, C, H, W) float32 in [0, 1]."""
    return torch.from_numpy(obs).float().permute(0, 3, 1, 2).div(255.0)
