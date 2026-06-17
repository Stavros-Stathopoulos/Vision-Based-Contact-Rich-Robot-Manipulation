import numpy as np
from abc import ABC, abstractmethod

class BaseControllerInterface(ABC):
    """
    Abstract Base Class representing the common interface for all controllers
    in the Intelligent Control project.
    """
    def __init__(self):
        """
        Initializes the controller.
        """
        pass
    
    @abstractmethod

    def reset(self) -> None:
        """
        Resets the internal state of the controller at the beginning of each episode.
        Must be implemented by any subclass.
        """
        pass

    @abstractmethod
    def act(self, obs: dict) -> tuple[np.ndarray, dict]:
        """
        Selects an action based on the current environment observation.
        Must be implemented by any subclass.
        
        Args:
            obs (dict): The observation dictionary from the robosuite environment.
                        Contains 'agentview_image' among other variables.
                        
        Returns:
            np.ndarray: The continuous action vector to be sent to the robot.
            dict: Additional information about the action selection.
        """
        pass