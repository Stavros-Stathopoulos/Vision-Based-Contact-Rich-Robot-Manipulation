import torch
import torch.nn as nn

from base_encoder import BaseEncoder

# Υπολογισμός εξόδου: 
# Layer 1 (Stride 2): (84 - 5)/2 + 1 = 40 -> (32, 40, 40)
# Layer 2 (Stride 2): (40 - 3)/2 + 1 = 19 -> (64, 19, 19)
# Layer 3 (Stride 1): (19 - 3)/1 + 1 = 17 -> (64, 17, 17)
_CONV_OUT_DIM = 64 * 17 * 17  # 18,496 features

class CNNEncoder(BaseEncoder):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self._embedding_dim = embedding_dim

        # Χρησιμοποιούμε μικρότερα kernels και strides για να μην χάσουμε την πληροφορία των contact-rich λεπτομερειών (παξιμάδι/βίδα)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),   # Είσοδος: (3, 84, 84)  -> Έξοδος: (32, 40, 40)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # Είσοδος: (32, 40, 40) -> Έξοδος: (64, 19, 19)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Είσοδος: (64, 19, 19) -> Έξοδος: (64, 17, 17)
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
