import torch
import torch.nn as nn
import numpy as np

class CNNEncoder(nn.Module):
    """
    CNN Encoder to process 84x84 images from the robosuite environment
    and map them to a low-dimensional feature embedding.
    """
    def __init__(self, embedding_dim: int = 256):
        super(CNNEncoder, self).__init__()
        
        # robosuite image input shape: (84, 84, 3)
        # PyTorch expects channels first: (3, 84, 84)
        
        self.conv_layers = nn.Sequential(
            # Layer 1: Input (3, 84, 84) -> Output (32, 20, 20)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Layer 2: Input (32, 20, 20) -> Output (64, 9, 9)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Layer 3: Input (64, 9, 9) -> Output (64, 7, 7)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Flatten to 64 * 7 * 7 = 3136 features
            nn.Flatten()
        )
        
        # Fully connected layer to project to the desired embedding size
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 3, 84, 84)
                              Normalized between 0.0 and 1.0
        Returns:
            torch.Tensor: Feature embeddings of shape (Batch_Size, embedding_dim)
        """
        features = self.conv_layers(x)
        embedding = self.fc(features)
        return embedding

# --- Integration and Verification Test ---
if __name__ == "__main__":
    print("Testing CNN Encoder with dummy image batch...")
    
    # 1. Initialize the Encoder
    embedding_size = 256
    encoder = CNNEncoder(embedding_dim=embedding_size)
    print(f"✅ CNN Encoder initialized. Target embedding dimension: {embedding_size}")
    
    # 2. Create a fake batch of 4 images (Simulating 4 steps or batch sampling)
    # Shape: (Batch, Height, Width, Channels) - typical robosuite style
    dummy_robosuite_batch = np.random.randint(0, 255, size=(4, 84, 84, 3), dtype=np.uint8)
    
    # 3. Preprocessing pipeline: Convert to float, transpose to (B, C, H, W) and normalize
    # This is exactly how we will process real environment observations!
    tensor_img = torch.from_numpy(dummy_robosuite_batch).float()
    tensor_img = tensor_img.permute(0, 3, 1, 2)  # Change layout to Channels-First
    tensor_img /= 255.0                          # Normalize to [0, 1]
    
    # 4. Pass through the network
    with torch.no_grad():
        output_embeddings = encoder(tensor_img)
        
    print(f"✅ Success! Output tensor shape: {output_embeddings.shape}")
    print(f"📊 Expected shape: (4, {embedding_size})")