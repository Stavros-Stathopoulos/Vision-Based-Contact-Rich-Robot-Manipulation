import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .cnn_encoder import CNNEncoder


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style random image shift (per-sample), the cheapest, highest-ROI
    regularizer for pixel-based RL. Replicate-pads by `pad` then crops back to
    the original size at a random offset. Square images only.
    Reference: Kostrikov et al., "Image Augmentation Is All You Need" (DrQ).
    """
    n, c, h, w = x.size()
    assert h == w, "random_shift expects square images"
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class MultiModalExtractor(BaseFeaturesExtractor):
    """Features extractor for a Dict observation {image, proprio}.

    * image branch  -> NatureCNN (`CNNEncoder`, handles stacked frames), with
      optional DrQ random-shift augmentation applied only during training.
    * proprio branch -> small MLP over the robot's own state.

    The two embeddings are concatenated. Image normalization (uint8 -> [0,1]) is
    done here, so build the policy with `normalize_images=False`. This also
    matters because a stacked image has 9 channels, which SB3 does NOT recognize
    as an "image" space and would therefore leave un-normalized.
    """

    def __init__(
        self,
        observation_space,
        cnn_dim: int = 256,
        proprio_hidden: int = 128,
        proprio_out: int = 64,
        augment: bool = True,
        shift_pad: int = 4,
    ):
        img_space = observation_space.spaces["image"]
        pro_space = observation_space.spaces["proprio"]
        in_channels = int(img_space.shape[0])
        proprio_in = int(pro_space.shape[0])

        super().__init__(observation_space, features_dim=cnn_dim + proprio_out)

        self.augment = augment
        self.shift_pad = shift_pad

        self.cnn = CNNEncoder(embedding_dim=cnn_dim, in_channels=in_channels)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_in, proprio_hidden),
            nn.ReLU(),
            nn.Linear(proprio_hidden, proprio_out),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        image = observations["image"].float() / 255.0
        if self.augment and self.training:
            image = random_shift(image, self.shift_pad)
        img_feat = self.cnn(image)
        pro_feat = self.proprio_mlp(observations["proprio"].float())
        return torch.cat([img_feat, pro_feat], dim=1)
