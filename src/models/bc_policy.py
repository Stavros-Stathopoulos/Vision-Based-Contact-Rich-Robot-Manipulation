import torch
import torch.nn as nn
from ..encoders.cnn_encoder import CNNEncoder


class BehaviorCloningPolicy(nn.Module):
    ''' 
    A simple behavior cloning policy that uses a CNN encoder to map images to continuous actions. 
    Our goal is to learn a policy that can predict the expert's actions given the current state image.
    Our goal is to do regression, that is, to map a state directly to continuous values of the action space

    '''
    def __init__(self, action_dim: int = 7, embedding_dim: int = 256):
        super().__init__()
        # Χρήση του Spatial-Aware CNN Encoder από το Task 1.3
        self.encoder = CNNEncoder(embedding_dim=embedding_dim)
        
        # Policy Head: Χαρτογραφεί το embedding απευθείας στις 7 continuous δράσεις
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),  # Fully Connected / Linear Layer that maps the embedding (256 dim) to a hidden layer (128 dim). Starting combining the features extracted by the CNN into a more compact representation.
            nn.ReLU(),  # Rectified Linear Unit --> Activation function that introduces non-linearity to the model, allowing it to learn complex mappings from the input to the output.
            nn.Linear(128, action_dim),  # Fully Connected / Linear Layer that maps the hidden layer (128 dim) to the action space (7 dim). It produces the 7 raw logits corresponding to the 7 robot actions (3 for position control, 3 for orientation control, and 1 for gripper control).
            nn.Tanh() # Περιορίζει τις δράσεις στο [-1, 1] όπως απαιτεί το robosuite
        )

    def forward(self, state_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_image: (B, 3, 84, 84) tensor normalized to [0, 1]
        Returns:
            (B, action_dim) predicted continuous actions
        """
        embedding = self.encoder(state_image)
        return self.action_head(embedding)