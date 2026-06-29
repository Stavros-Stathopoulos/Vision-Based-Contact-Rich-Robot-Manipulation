import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .observation import ObservationProcessor
from ..config import NUM_STACK


class RobosuiteGymWrapper(gym.Env):
    """Adapts a robosuite env to the Gymnasium API for Stable-Baselines3.

    Key differences from a naive wrapper:

    * Observation is a Dict {image, proprio} built by `ObservationProcessor`
      (image-only + robot proprioception; no object/nut state).
    * Correct episode semantics: `terminated` is set ONLY on true task success
      (`env._check_success()`); reaching the horizon is reported as `truncated`.
      This lets SAC bootstrap past timeouts instead of treating them as real
      terminal states (which otherwise biases the value function downward and
      encourages the "hover near the nut" local optimum).
    * `info["is_success"]` exposes the true success flag for honest metrics.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env, num_stack: int = NUM_STACK):
        super().__init__()
        self.env = env
        self.processor = ObservationProcessor(num_stack=num_stack)

        low, high = env.action_spec
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=low.shape, dtype=np.float32
        )

        # Build the observation space from one real reset, and reuse that
        # observation for the first episode so we don't pay for two MuJoCo
        # resets (resets are expensive and leak renderer memory).
        self._pending_raw = self.env.reset()
        self.observation_space = self.processor.build_spaces(self._pending_raw)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._pending_raw is not None:
            raw, self._pending_raw = self._pending_raw, None
        else:
            raw = self.env.reset()
        return self.processor.reset(raw), {}

    def step(self, action):
        raw, reward, done, info = self.env.step(action)
        success = bool(self.env._check_success())

        obs = self.processor.observe(raw)
        terminated = success
        truncated = bool(done) and not success  # robosuite `done` == horizon hit

        info = dict(info)
        info["is_success"] = success
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
