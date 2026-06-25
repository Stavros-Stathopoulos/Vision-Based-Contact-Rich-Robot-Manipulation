import torch
import torch.nn as nn

from .base_encoder import BaseEncoder

# Lightweight "NatureCNN" topology. Ο stride-4 πρώτος conv υποδειγματοληπτεί
# επιθετικά ώστε να μειωθούν δραστικά τα FLOPs (απαραίτητο για CPU-only μηχανές).
# Υπολογισμός εξόδου:
# Layer 1 (Stride 4): (84 - 8)/4 + 1 = 20 -> (32, 20, 20)
# Layer 2 (Stride 2): (20 - 4)/2 + 1 = 9  -> (64, 9, 9)
# Layer 3 (Stride 1): (9 - 3)/1 + 1  = 7  -> (64, 7, 7)
_CONV_OUT_DIM = 64 * 7 * 7  # 3,136 features (vs 18,496 του προηγούμενου encoder)

class CNNEncoder(BaseEncoder):
    def __init__(self, observation_space=None, embedding_dim: int = 256):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.features_dim = embedding_dim

        # Σταθμισμένη επιλογή ταχύτητας/λεπτομέρειας: ο μεγαλύτερος stride θυσιάζει
        # λίγη χωρική ανάλυση για ~6x λιγότερο compute και πολύ μικρότερο FC layer.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # Είσοδος: (3, 84, 84)  -> Έξοδος: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Είσοδος: (32, 20, 20) -> Έξοδος: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Είσοδος: (64, 9, 9)   -> Έξοδος: (64, 7, 7)
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
