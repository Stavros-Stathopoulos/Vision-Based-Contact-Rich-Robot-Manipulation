import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

print("Initializing environment...")

# Load composite operational-space controller config
controller_config = load_composite_controller_config(controller="BASIC")

# Create environment based on the project requirements
env = suite.make(
    env_name="NutAssembly",
    robots="Panda",                  # Franka Emika Panda arm
    gripper_types="PandaGripper",    # PandaGripper
    controller_configs=controller_config,
    has_renderer=False,              # No live window
    has_offscreen_renderer=True,     # Enabled for camera observations
    use_camera_obs=True,             # Using camera observations
    use_object_obs=False,            # Privileged object-state is disabled
    camera_names="agentview",        # Benchmark camera setup
    camera_heights=84,               # Height resolution
    camera_widths=84,                # Width resolution
    control_freq=20,                 # Control frequency
    horizon=500,                     # Episode horizon
)

# Reset environment to get initial observation
obs = env.reset()

# Verify that the image tensor is received correctly
if "agentview_image" in obs:
    image = obs["agentview_image"]
    print("✅ Success! Environment initialized correctly.")
    print(f"📸 Image tensor shape: {image.shape}")
    print(f"📊 Image data type: {image.dtype}")
else:
    print("❌ Error: 'agentview_image' not found in observations.")

# Step through the environment with a zero action vector
action = np.zeros(env.action_spec[0].shape)
obs, reward, done, info = env.step(action)

print(f"🏃 Single step execution works. Reward received: {reward}")

env.close()
print("Environment closed cleanly.")