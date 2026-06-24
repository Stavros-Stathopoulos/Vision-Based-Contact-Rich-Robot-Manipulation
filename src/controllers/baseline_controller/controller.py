import numpy as np
from scipy.spatial.transform import Rotation
from ..base_controller import BaseController


# Local offset from nut center to its handle in the nut's body frame.
_HANDLE_OFFSETS = {
    "RoundNut": np.array([0.06, 0.0, 0.0]),
    "SquareNut": np.array([0.054, 0.0, 0.0]),
}

# Peg world-frame XY positions (fixed by the arena).
_PEG_XY = {
    "RoundNut": np.array([0.23, -0.1]),
    "SquareNut": np.array([0.23, 0.1]),
}


class HeuristicBaselineController(BaseController):
    """State-guided expert controller for the NutAssembly task.

    Uses privileged object state (nut position/quaternion, end-effector
    position, gripper qpos) to execute a grasp-lift-place sequence.
    Intended for expert demonstration collection — the collected dataset
    stores only camera images, not the privileged state.
    """

    def __init__(
        self,
        action_dim: int = 7,
        nut_key: str = "RoundNut",
        max_grasp_attempts: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.nut_key = nut_key
        self.handle_offset = _HANDLE_OFFSETS[nut_key]
        self.peg_xy = _PEG_XY[nut_key]
        self.max_grasp_attempts = max_grasp_attempts
        self._reset_state()

    def _reset_state(self) -> None:
        self.phase = "approach_above"
        self.step_counter = 0
        self.phase_counter = 0
        self.grasp_attempts = 0
        self.nut_init_z: float = 0.89

    def reset(self) -> None:
        """Resets the internal state machine for a new episode."""
        self._reset_state()
        print("Baseline Controller reset for a new episode.")

    def _handle_pos(self, obs: dict) -> np.ndarray:
        """Compute world-frame handle position from nut pose."""
        pos = obs[f"{self.nut_key}_pos"]
        quat = obs[f"{self.nut_key}_quat"]
        rot = Rotation.from_quat(quat).as_matrix()
        return pos + rot @ self.handle_offset

    def act(self, obs: dict) -> np.ndarray:
        """Select action using a state-guided finite state machine.

        Requires ``use_object_obs=True`` on the environment so that
        ``obs`` contains nut position, quaternion, and robot proprioception.
        """
        action = np.zeros(self.action_dim)
        eef = obs["robot0_eef_pos"]
        handle = self._handle_pos(obs)
        nut_pos = obs[f"{self.nut_key}_pos"]

        if self.step_counter == 0:
            self.nut_init_z = nut_pos[2]

        if self.phase == "approach_above":
            target = handle.copy()
            target[2] += 0.04
            d = target - eef
            action[0:3] = np.clip(d * 25.0, -1, 1)
            action[6] = -1.0
            if np.linalg.norm(d) < 0.008:
                self.phase = "descend"
                self.phase_counter = 0

        elif self.phase == "descend":
            action[0] = np.clip((handle[0] - eef[0]) * 40.0, -1, 1)
            action[1] = np.clip((handle[1] - eef[1]) * 40.0, -1, 1)
            action[2] = -1.0
            action[6] = -1.0
            self.phase_counter += 1
            if self.phase_counter > 80 or eef[2] <= nut_pos[2] + 0.002:
                self.phase = "grasp"
                self.phase_counter = 0

        elif self.phase == "grasp":
            action[0] = np.clip((handle[0] - eef[0]) * 20.0, -1, 1)
            action[1] = np.clip((handle[1] - eef[1]) * 20.0, -1, 1)
            action[2] = -0.3
            action[6] = 1.0
            self.phase_counter += 1
            if self.phase_counter > 25:
                self.phase = "lift_test"
                self.phase_counter = 0

        elif self.phase == "lift_test":
            action[2] = 1.0
            action[6] = 1.0
            self.phase_counter += 1
            if self.phase_counter > 35:
                if nut_pos[2] > self.nut_init_z + 0.03:
                    self.phase = "lift"
                else:
                    self.grasp_attempts += 1
                    if self.grasp_attempts < self.max_grasp_attempts:
                        self.phase = "release_retry"
                        self.phase_counter = 0
                    else:
                        self.phase = "done"

        elif self.phase == "release_retry":
            action[6] = -1.0
            self.phase_counter += 1
            if self.phase_counter > 15:
                self.phase = "approach_above"
                self.phase_counter = 0

        elif self.phase == "lift":
            action[2] = 1.0
            action[6] = 1.0
            if eef[2] > 1.05:
                self.phase = "move_to_peg"

        elif self.phase == "move_to_peg":
            eef_to_nut = nut_pos - eef
            nut_target = np.array([self.peg_xy[0], self.peg_xy[1], 1.05])
            eef_target = nut_target - eef_to_nut
            d = eef_target - eef
            action[0:3] = np.clip(d * 15.0, -1, 1)
            action[6] = 1.0
            if np.linalg.norm(d[:2]) < 0.015 and abs(d[2]) < 0.02:
                self.phase = "lower_to_peg"

        elif self.phase == "lower_to_peg":
            eef_to_nut = nut_pos - eef
            nut_target = np.array(
                [self.peg_xy[0], self.peg_xy[1], 0.85 + 0.02]
            )
            eef_target = nut_target - eef_to_nut
            d = eef_target - eef
            action[0:3] = np.clip(d * 10.0, -1, 1)
            action[6] = 1.0

        elif self.phase == "done":
            action[6] = 1.0

        self.step_counter += 1
        return action
