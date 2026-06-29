import numpy as np
import torch

from src.controllers.base_controller import BaseController
from src.models.bc_model import BehaviorCloningPolicy
from src import config


class Controller(BaseController):
    """Behavior Cloning baseline behind the common controller interface.

    This is the "meaningful competing method" baseline (learning-based, distinct
    from the final SAC controller). It is single-frame and image-only, matching
    how the BC model was trained (see src/models/train_bc.py). Uses only the
    benchmark-allowed `agentview_image` observation.
    """

    def __init__(self, model_path: str = "bc_model.pth", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BehaviorCloningPolicy().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def reset(self, seed=None):
        pass

    def act(self, observation):
        img = np.asarray(observation[config.IMAGE_KEY], dtype=np.float32)  # (H, W, 3)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            action = self.model(x).squeeze(0).cpu().numpy()
        return action, {}
