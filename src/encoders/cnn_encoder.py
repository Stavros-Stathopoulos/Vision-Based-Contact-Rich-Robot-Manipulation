import torch
import torch.nn as nn

from .base_encoder import BaseEncoder

_CONV_OUT_DIM = 64 * 7 * 7  # spatial output of the 3-layer conv stack on 84x84 input


class CNNEncoder(BaseEncoder):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self._embedding_dim = embedding_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # (3,84,84)  -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (32,20,20) -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64,9,9)   -> (64,7,7)
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(_CONV_OUT_DIM, embedding_dim),
            nn.ReLU(),
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 84, 84) float32 normalized to [0, 1]
        Returns:
            (B, embedding_dim) feature embeddings
        """
        return self.fc(self.conv_layers(x))
