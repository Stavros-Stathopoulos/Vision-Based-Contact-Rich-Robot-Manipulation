import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from controller import BaselineController

print("Setting up baseline evaluation environment...")

controller_config = load_composite_controller_config(controller="BASIC")

env = suite.make(
    env_name="NutAssembly",
    robots="Panda",
    gripper_types="PandaGripper",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    use_object_obs=False,  # Privileged information is disabled!
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
    control_freq=20,  # Control frequency in Hz (20Hz means each step is 0.05 seconds)
    horizon=100,  # Shortened horizon for quick evaluation
)

# Initialize Controller and Environment
controller = BaselineController(action_dim=env.action_spec[0].shape[0])
obs = env.reset()
controller.reset()

total_reward = 0.0
done = False
step = 0

print("🏃 Running evaluation episode...")

while not done and step < 100:
    # 1. Get action from our baseline policy -> a = pi(s)
    action = controller.act(obs)
    
    # 2. Step the environment MDP -> (s', r, done)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    step += 1

print("\n--- Evaluation Results ---")
print(f"Total Steps Executed: {step}")
print(f"Cumulative Reward (Return G): {total_reward}")
if total_reward > 0:
    print("Success! The baseline controller successfully interacted with the object.")
else:
    print("Baseline completed. Performance logged as minimum benchmark.")

env.close()