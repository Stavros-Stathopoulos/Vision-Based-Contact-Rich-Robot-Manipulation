import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RobosuiteGymWrapper(gym.Env):
    """
    Custom Wrapper που μετατρέπει ένα περιβάλλον robosuite σε standard Gymnasium Environment
    για εκπαίδευση με αλγορίθμους Reinforcement Learning (SAC or TD3).

    Reward shaping:
        Προσθέτει έναν πυκνό (dense) όρο "reaching" που αυξάνεται όσο ο end-effector
        πλησιάζει το κοντινότερο παξιμάδι και μειώνεται όσο απομακρύνεται. Ο όρος
        υπολογίζεται από privileged θέσεις του MuJoCo (gripper site & nut body). Αυτό
        ΔΕΝ παραβιάζει το `use_object_obs=False`: το policy observation παραμένει μόνο
        εικόνα — η privileged πληροφορία χρησιμοποιείται αποκλειστικά στο reward κατά
        την εκπαίδευση, όχι ως είσοδος στην πολιτική.
    """
    def __init__(self, env, reach_reward_scale: float = 0.5, reach_tanh_scale: float = 10.0):
        super(RobosuiteGymWrapper, self).__init__()
        self.env = env

        # Παράμετροι του dense reaching reward.
        self.reach_reward_scale = reach_reward_scale   # 0 -> απενεργοποίηση του shaping
        self.reach_tanh_scale = reach_tanh_scale       # μεγαλύτερο = πιο "αιχμηρό" reward κοντά στον στόχο

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

    def _eef_position(self):
        """
        Θέση του end-effector site. Συμβατό με robosuite 1.4 (`_eef_xpos`) και
        1.5+ (per-arm `eef_site_id` dict -> site_xpos).
        """
        u = self.env
        eef = getattr(u, "_eef_xpos", None)   # robosuite <= 1.4
        if eef is not None:
            return np.asarray(eef)
        robot = u.robots[0]                    # robosuite 1.5+
        site_id = robot.eef_site_id
        if isinstance(site_id, dict):          # {'right': id, ...} σε multi-arm API
            site_id = site_id[robot.arms[0]]
        return np.asarray(u.sim.data.site_xpos[site_id])

    def _reaching_reward(self) -> float:
        """
        Dense reward βασισμένο στην απόσταση gripper -> κοντινότερο παξιμάδι.
        Επιστρέφει τιμή στο [0, 1]: ~1 όταν ο gripper ακουμπάει το παξιμάδι, ->0 μακριά.
        Διαβάζει privileged θέσεις απευθείας από το MuJoCo (μόνο για το reward).
        """
        u = self.env
        try:
            eef_pos = self._eef_position()
            sim_data = u.sim.data
            best = 0.0
            for nut in u.nuts:                        # NutAssembly: square + round nut
                body_id = u.obj_body_id[nut.name]
                nut_pos = sim_data.body_xpos[body_id]
                dist = float(np.linalg.norm(eef_pos - nut_pos))
                best = max(best, 1.0 - np.tanh(self.reach_tanh_scale * dist))
            return best
        except (AttributeError, KeyError, TypeError, IndexError):
            # Αν το env δεν είναι NutAssembly ή αλλάξει το API, απλώς δεν προσθέτουμε shaping.
            return 0.0

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

        # Dense reaching shaping: ψηλότερο reward όσο πλησιάζει το παξιμάδι.
        if self.reach_reward_scale != 0.0:
            reach = self.reach_reward_scale * self._reaching_reward()
            reward += reach
            info["reach_reward"] = reach

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
