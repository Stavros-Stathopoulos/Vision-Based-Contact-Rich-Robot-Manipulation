import numpy as np
from ..base_controller import BaseController

class HeuristicBaselineController(BaseController):
    """
    A scripted heuristic controller (Finite State Machine) for the NutAssembly task.
    Acts as a baseline policy and an expert data generator for Imitation Learning.
    """
    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim
        self.step_counter = 0
        
    def reset(self) -> None:
        """ Resets the internal state machine for a new episode. """
        self.step_counter = 0
        print("Baseline Controller reset for a new episode.")

    def act(self, obs: dict) -> tuple[np.ndarray, dict]:
        """
        Outputs a hardcoded sequence of actions based on the elapsed simulation steps.
        Does NOT use privileged object state coordinates, obeying 'use_object_obs=False'.
        """

        action = np.zeros(self.action_dim)

        # State 1: Move gripper down towards the nut location area (Steps 0 to 25)
        if self.step_counter < 25:
            action[2] = -0.15
            action[6] = -1.0

        # State 2: Close the gripper firmly around the nut (Steps 25 to 45)
        elif self.step_counter < 45:
            action[2] = -0.02
            action[6] = 1.0

        # State 3: Lift the nut up and move towards the peg (Steps 45 to 80)
        elif self.step_counter < 80:
            action[2] = 0.20
            action[0] = 0.05
            action[6] = 1.0

        # State 4: Hold position
        else:
            action[6] = 1.0

        self.step_counter += 1
        info = {"step": self.step_counter}
        return action, info