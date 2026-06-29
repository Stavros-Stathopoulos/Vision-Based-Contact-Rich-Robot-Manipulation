import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RobosuiteGymWrapper(gym.Env):
    """
    Custom Wrapper modifying the robosuite NutAssembly environment into a standard 
    Gymnasium Environment using state-based (privileged) observations.
    """
    def __init__(self, env):
        super(RobosuiteGymWrapper, self).__init__()
        self.env = env
        
        # 1. Action Space Mapping (7 continuous joint/gripper actions)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=self.env.action_spec[0].shape, 
            dtype=np.float32
        )
        
        # 2. Dynamic Observation Space Calculation
        # We perform a dummy reset to dynamically check the size of the combined flat vectors
        raw_obs = self.env.reset()
        flat_obs = self.obs_space_refactor(raw_obs)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32
        )

    def obs_space_refactor(self, robosuite_obs):
        """Extracts and concatenates all vector states excluding images."""
        obs_list = []
        for key, value in robosuite_obs.items():
            # Exclude raw visual modalities to train on ground-truth states
            if "image" not in key and "camera" not in key:
                obs_list.append(np.array(value).flatten())
        
        return np.concatenate(obs_list).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        robosuite_obs = self.env.reset()
        obs = self.obs_space_refactor(robosuite_obs)
        return obs, {}

    def step(self, action):
        robosuite_obs, reward, done, info = self.env.step(action)
        obs = self.obs_space_refactor(robosuite_obs)
        
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()