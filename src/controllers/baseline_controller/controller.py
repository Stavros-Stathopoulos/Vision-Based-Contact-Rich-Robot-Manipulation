import numpy as np
from base_controller import BaseController

class HeuristicBaselineController(BaseController):
    """
    A scripted heuristic controller (Finite State Machine) for the NutAssembly task.
    Acts as a baseline policy and an expert data generator for Imitation Learning.
    """
    def __init__(self, action_dim: int = 7):
        super().__init__(action_dim=action_dim)
        self.step_counter = 0
        
    def reset(self) -> None:
        """ Resets the internal state machine for a new episode. """
        self.step_counter = 0
        print("Baseline Controller reset for a new episode.")

    def act(self, obs: dict) -> np.ndarray:
        """
        Outputs a hardcoded sequence of actions based on the elapsed simulation steps.
        Does NOT use privileged object state coordinates, obeying 'use_object_obs=False'.
        """
        action = np.zeros(self.action_dim)

        """
        The following action sequence is a simple heuristic designed to complete the NutAssembly task:
        # 1. Move down to the nut location
        # 2. Close the gripper to grasp the nut
        # 3. Lift the nut and move towards the peg
        # 4. Hold the position (simulate placing the nut on the peg)
        Remarks: 
        1) control_frequency is assumed to be 20Hz, so each step corresponds to 0.05 seconds (20 steps = 1 second).
        2) action[0:3] corresponds to the XYZ position control of the gripper, and action[6] corresponds to the gripper open/close command.
        """
        
        
        # State 1: Move gripper down towards the nut location area (Steps 0 to 25) --> 1.25 seconds 
        if self.step_counter < 25: 
            action[2] = -0.15  # Move down along Z axis
            action[6] = -1.0   # Keep gripper fully open
            
        # State 2: Close the gripper firmly around the nut (Steps 25 to 45) --> 1 second 
        elif self.step_counter < 45:
            action[2] = -0.02  # Apply slight downward pressure to maintain contact
            action[6] = 1.0    # Close gripper (grasp action)
            
        # State 3: Lift the nut up and move towards the peg (Steps 45 to 80) --> 1.75 seconds
        elif self.step_counter < 80:
            action[2] = 0.20   # Move up along Z axis
            action[0] = 0.05   # Move slightly along X toward the peg setup
            action[6] = 1.0    # Keep gripper tightly closed
            
        # State 4: Terminate episode / Hold position
        else:
            action[6] = 1.0    # Hold object
            
        self.step_counter += 1
        return action