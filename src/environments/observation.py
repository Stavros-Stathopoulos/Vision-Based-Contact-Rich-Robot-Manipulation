import numpy as np
from collections import deque
from gymnasium import spaces

from ..config import IMAGE_KEY, PROPRIO_KEY, NUM_STACK, IMG_SIZE


class ObservationProcessor:
    """Turns a raw robosuite observation dict into the policy observation.

    This class is the SINGLE chokepoint that reads the robosuite observation
    dict. It reads exactly two keys and nothing else:

        * IMAGE_KEY   ("agentview_image")      -> the camera the policy sees with
        * PROPRIO_KEY ("robot0_proprio-state") -> the robot's OWN joint / eef /
                                                  gripper state (velocities included)

    It NEVER reads object/nut keys (e.g. "*_pos", "*_quat" of the nut). The nut's
    location must be inferred from pixels, per the assignment constraint. If you
    ever need to relax/audit this rule, this is the only file to inspect.

    The image is stacked over the last `num_stack` frames so the network can
    perceive motion (approach speed, contact), which a single frame cannot show.
    """

    def __init__(self, num_stack: int = NUM_STACK, proprio_dim: int | None = None):
        self.num_stack = num_stack
        self._frames: deque = deque(maxlen=num_stack)
        # Expected proprio length (set from the trained model at deploy time). Used
        # only as a defensive fallback so a benchmark obs dict that omits the
        # proprio key degrades to zeros instead of crashing the controller.
        self._proprio_dim = proprio_dim

    @staticmethod
    def _image(raw: dict) -> np.ndarray:
        # robosuite gives (H, W, C) uint8; PyTorch/CNN wants (C, H, W)
        img = np.asarray(raw[IMAGE_KEY], dtype=np.uint8)
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

    def _proprio(self, raw: dict) -> np.ndarray:
        if PROPRIO_KEY in raw:
            vec = np.asarray(raw[PROPRIO_KEY], dtype=np.float32).ravel()
            self._proprio_dim = vec.shape[0]
            return vec
        if self._proprio_dim is not None:
            return np.zeros(self._proprio_dim, dtype=np.float32)
        raise KeyError(
            f"Observation has no '{PROPRIO_KEY}' and no expected proprio_dim was set"
        )

    def reset(self, raw: dict) -> dict:
        """Start a new episode: fill the frame stack with the first frame."""
        frame = self._image(raw)
        self._frames.clear()
        for _ in range(self.num_stack):
            self._frames.append(frame)
        return self._build(raw)

    def observe(self, raw: dict) -> dict:
        """Push the newest frame and return the stacked observation."""
        self._frames.append(self._image(raw))
        return self._build(raw)

    def _build(self, raw: dict) -> dict:
        image = np.concatenate(list(self._frames), axis=0)  # (num_stack*3, H, W)
        return {"image": image, "proprio": self._proprio(raw)}

    def build_spaces(self, raw: dict) -> spaces.Dict:
        """Derive the Gymnasium observation space from a sample raw observation."""
        proprio_dim = self._proprio(raw).shape[0]
        channels = 3 * self.num_stack
        return spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(channels, IMG_SIZE, IMG_SIZE), dtype=np.uint8
                ),
                "proprio": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32
                ),
            }
        )
