import numpy as np
from base_controller_interface import BaseControllerInterface

class DummyController(BaseControllerInterface):
    """
    A temporary dummy controller to verify that the BaseController 
    interface enforces correct implementation.
    """
    def reset(self) -> None:
        print("🔄 Controller reset successfully!")

    def act(self, obs: dict) -> np.ndarray:
        print("🧠 Controller received observation and calculating action...")
        # Return a zero action vector of the correct dimension
        return np.zeros(self.action_dim)

# --- Verification Script ---
if __name__ == "__main__":
    print("Testing Controller Interface...")
    
    # Assume action dimension for Panda robot is 7 (6 for arm displacement/rotation + 1 for gripper)
    action_dimension = 7 
    
    try:
        # 1. Try to instantiate the DummyController (Should work)
        controller = DummyController(action_dim=action_dimension)
        
        # 2. Test the methods
        controller.reset()
        
        dummy_obs = {"agentview_image": np.zeros((84, 84, 3))}
        action = controller.act(dummy_obs)
        
        print(f"✅ Success! Generated action shape: {action.shape}")
        
    except TypeError as e:
        print(f"❌ Interface Error: {e}")