import torch
from stable_baselines3 import SAC

from src.controllers.base_controller import BaseController
from src.environments.observation import ObservationProcessor
from src import config


class Controller(BaseController):
    """Deploys a trained SAC agent behind the common controller interface.

    Builds the exact same {image, proprio} observation the policy was trained on
    (camera + robot proprioception only — no nut location) and maintains its own
    frame stack across an episode. Call `reset()` between episodes.
    """

    def __init__(self, model_path: str | None = None, device=None, num_stack: int = config.NUM_STACK):
        super().__init__()
        if model_path is None:
            model_path = config.BEST_MODEL_PATH + ".zip"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAC.load(model_path, device=self.device)
        proprio_dim = int(self.model.observation_space["proprio"].shape[0])
        self.processor = ObservationProcessor(num_stack=num_stack, proprio_dim=proprio_dim)
        self._started = False

    def reset(self, seed=None):
        self._started = False

    def act(self, observation):
        if not self._started:
            obs = self.processor.reset(observation)
            self._started = True
        else:
            obs = self.processor.observe(observation)
        action, _ = self.model.predict(obs, deterministic=True)
        return action, {}
