from abc import ABC, abstractmethod
import numpy as np

class BaseControllerInterface(ABC):
    """
    Abstract Base Class representing the common interface for all controllers
    in the Intelligent Control project.
    """
    def __init__(self, action_dim: int):
        """
        Initializes the controller.
        
        Args:
            action_dim (int): The dimension of the robot's action space.
        """
        self.action_dim = action_dim
    
    @abstractmethod

    def reset(self) -> None:
        """
        Resets the internal state of the controller at the beginning of each episode.
        Must be implemented by any subclass.
        """
        pass

    @abstractmethod
    def act(self, obs: dict) -> np.ndarray:
        """
        Selects an action based on the current environment observation.
        Must be implemented by any subclass.
        
        Args:
            obs (dict): The observation dictionary from the robosuite environment.
                        Contains 'agentview_image' among other variables.
                        
        Returns:
            np.ndarray: The continuous action vector to be sent to the robot.
        """
        pass