import torch
import torch.nn as nn

class BehaviorCloningStatePolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 7):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Maps actions directly to [-1, 1] range
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)