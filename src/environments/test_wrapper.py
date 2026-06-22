import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from .gym_wrapper import RobosuiteGymWrapper

def test_wrapper():
    print("Initializing raw robosuite environment")
    controller_config = load_composite_controller_config(controller="BASIC")
    
    raw_env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,  # Raw pixels μόνο
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=50,
    )
    
    print("Wrapping environment into Gymnasium API")
    env = RobosuiteGymWrapper(raw_env)
    
    # Έλεγχος των Spaces
    print("\n--- Spaces Verification ---")
    print(f"Action Space:      {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Δοκιμαστικό Reset
    print("\nTesting env.reset()")
    obs, info = env.reset()
    print(f"Reset Obs Shape: {obs.shape}")
    print(f"Reset Obs Type:  {obs.dtype}")
    
    # Δοκιμαστικό Step με τυχαία δράση
    print("\nTesting env.step() with a random action")
    random_action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(random_action)
    
    print(f"Step Obs Shape:  {obs.shape}")
    print(f"Reward Received:  {reward:.4f}")
    print(f"Terminated:       {terminated}")
    print(f"Truncated:        {truncated}")
    
    print("\nWrapper test complete! Closing environment.")
    env.close()

if __name__ == "__main__":
    test_wrapper()