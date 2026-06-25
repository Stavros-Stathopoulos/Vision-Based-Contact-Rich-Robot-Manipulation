"""Vision-only deployment controller backed by the trained BC policy.

This is the rules-compliant deliverable: it consumes ONLY the camera image from the
observation dict (no privileged object state) and outputs a 7-DOF action. It wraps
the network distilled from the oracle's demonstrations.
"""
import numpy as np
import torch

from ..base_controller import BaseController
from ...models.bc_model import BehaviorCloningPolicy


class BCController(BaseController):
    def __init__(self, model_path="bc_model.pth", device=None):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BehaviorCloningPolicy().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def reset(self, seed=None):
        pass

    def act(self, observation):
        # vision-only: read pixels, normalize, (H,W,C)->(1,C,H,W)
        img = np.asarray(observation["agentview_image"], dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.model(img_t).squeeze(0).cpu().numpy()
        return action, {}
