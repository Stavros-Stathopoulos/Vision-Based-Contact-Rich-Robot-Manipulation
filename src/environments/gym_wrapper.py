import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RobosuiteGymWrapper(gym.Env):
    """
    Custom Wrapper που μετατρέπει ένα περιβάλλον robosuite σε standard Gymnasium Environment
    για εκπαίδευση με αλγορίθμους Reinforcement Learning (SAC or TD3).
    """
    def __init__(self, env):
        super(RobosuiteGymWrapper, self).__init__()
        self.env = env
        
        # 1. Action Space Mapping (7 continuous actions για τον Panda)
        # Κανονικοποιημένος χώρος στο [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=self.env.action_spec[0].shape, 
            dtype=np.float32
        )
        
        # 2. Observation Space Refactoring (PyTorch format: [Channels, Height, Width])
        # Η εικόνα από (84, 84, 3) μετατρέπεται σε (3, 84, 84)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(3, 84, 84), 
            dtype=np.uint8
        )

    def obs_space_refactor(self, robosuite_obs):
        """Μετατρέπει το robosuite observation dict σε standard Gym image tensor."""
        # Απομόνωση της agentview εικόνας
        img = robosuite_obs["agentview_image"]
        
        # Transpose από (H, W, C) σε (C, H, W) για να είναι συμβατό με PyTorch/RL CNNs
        img = np.transpose(img, (2, 0, 1))
        
        # Επιστροφή ως contiguous numpy array
        return np.ascontiguousarray(img, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        """Επαναφέρει το περιβάλλον και επιστρέφει το επεξεργασμένο observation."""
        super().reset(seed=seed)
        
        # Reset στο robosuite environment
        robosuite_obs = self.env.reset()
        
        # Επεξεργασία της αρχικής εικόνας
        obs = self.obs_space_refactor(robosuite_obs)
        info = {}
        
        return obs, info

    def step(self, action):
        """Εκτελεί τη δράση και επιστρέφει την τυποποιημένη 5-άδα του Gymnasium."""
        # Εκτέλεση βήματος στο robosuite
        robosuite_obs, reward, done, info = self.env.step(action)
        
        # Επεξεργασία του νέου state
        obs = self.obs_space_refactor(robosuite_obs)
        
        # Διαχωρισμός του done σε terminated και truncated (standard Gymnasium API)
        terminated = done  # Hard horizon ή success
        truncated = False  # Μπορείς να το συνδέσεις με το μέγιστο όριο βημάτων αν θες
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Αναθέτει το render στο εσωτερικό περιβάλλον."""
        return self.env.render()

    def close(self):
        """Κλείνει το περιβάλλον ελευθερώνοντας τη μνήμη."""
        self.env.close()