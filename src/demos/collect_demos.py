import os
import gc
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from src.controllers.baseline_controller import HeuristicBaselineController

NUM_SUCCESSFUL_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
MAX_TOTAL_EPISODES = 100
OUTPUT_FILE = "expert_demos.npz"
REWARD_THRESHOLD = 10.0

print("Initializing environment for data collection")
controller_config = load_composite_controller_config(controller="BASIC")

env = suite.make(
    env_name="NutAssembly",
    robots="Panda",
    gripper_types="PandaGripper",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    use_object_obs=True,
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
    control_freq=20,
    horizon=MAX_STEPS_PER_EPISODE,
    reward_shaping=True,
)

controller = HeuristicBaselineController(action_dim=env.action_spec[0].shape[0])

images = []
actions = []

successful_episodes_counter = 0
episode_idx = 0

print(
    f"Starting data collection. Target: {NUM_SUCCESSFUL_EPISODES} successful "
    f"episodes or max {MAX_TOTAL_EPISODES} attempts."
)

while (
    successful_episodes_counter < NUM_SUCCESSFUL_EPISODES
    and episode_idx < MAX_TOTAL_EPISODES
):
    episode_idx += 1
    obs = env.reset()
    controller.reset()

    ep_images = []
    ep_actions = []
    ep_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        img = obs["agentview_image"]
        ep_images.append(img)

        action = controller.act(obs)
        ep_actions.append(action)

        obs, reward, done, info = env.step(action)
        ep_reward += reward

        if done:
            break

    if ep_reward > REWARD_THRESHOLD:
        successful_episodes_counter += 1
        images.extend(ep_images)
        actions.extend(ep_actions)
        print(
            f"Episode {episode_idx}/{MAX_TOTAL_EPISODES}: SUCCESS! "
            f"Reward: {ep_reward:.2f}. "
            f"Total successful collected: {successful_episodes_counter}/"
            f"{NUM_SUCCESSFUL_EPISODES}"
        )
    else:
        print(
            f"Episode {episode_idx}/{MAX_TOTAL_EPISODES}: Failed. "
            f"(Reward: {ep_reward:.2f}). Discarding trajectory."
        )

    del ep_images
    del ep_actions
    gc.collect()

if len(images) > 0:
    images = np.array(images, dtype=np.uint8)
    actions = np.array(actions, dtype=np.float32)

    np.savez_compressed(OUTPUT_FILE, states=images, actions=actions)

    print("\n--- Collection Complete ---")
    print(f"Dataset saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Total Image Frames Stored: {images.shape}")
    print(f"Total Action Vectors Stored: {actions.shape}")
else:
    print("\n--- Collection Stopped ---")
    print(
        f"No successful episodes were collected in {MAX_TOTAL_EPISODES} "
        "attempts. Dataset was not saved."
    )

env.close()
