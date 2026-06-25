import numpy as np
import torch
from stable_baselines3 import SAC
from src.controllers.base_controller import BaseController
from src.encoders.cnn_encoder import CNNEncoder


class Controller(BaseController):
    def __init__(self, model_path="SAC_100k.zip", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAC.load(model_path, device=self.device)

    def reset(self, seed=None):
        pass

    def act(self, observation):
        img = observation["agentview_image"]
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.uint8)

        action, _ = self.model.predict(img, deterministic=True)
        info = {}
        return action, info
