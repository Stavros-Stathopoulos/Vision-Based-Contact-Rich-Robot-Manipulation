import os
import numpy as np
from openai import images
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from controllers.baseline_controller import HeuristicBaselineController

# Ρυθμίσεις συλλογής
NUM_SUCCESSFUL_EPISODES = 25  # Πόσες πλήρεις και επιτυχημένες τροχιές θέλουμε να αποθηκεύσουμε στο dataset μας
MAX_STEPS_PER_EPISODE = 100  # Μέγιστος αριθμός βημάτων ανά επεισόδιο (5 δευτερόλεπτα με 20Hz). Hard limit για αποφυγή infinite loops σε περίπτωση αποτυχίας.
OUTPUT_FILE = "expert_demonstrations.npz"

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
    use_object_obs=False,
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
    control_freq=20,
    horizon=MAX_STEPS_PER_EPISODE,
)

controller = HeuristicBaselineController(action_dim=env.action_spec[0].shape[0])

# Λίστες για την αποθήκευση όλων των επιτυχημένων τροχιών
images = []
actions = []

successful_episodes_counter = 0
episode_idx = 0

print(f"Starting data collection. Target: {NUM_SUCCESSFUL_EPISODES} successful episodes.")

while successful_episodes_counter < NUM_SUCCESSFUL_EPISODES:
    episode_idx += 1
    obs = env.reset()
    controller.reset()
    
    # Προσωρινές λίστες για το τρέχον επεισόδιο
    ep_images = []
    ep_actions = []
    ep_reward = 0.0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        # Αποθήκευση της τρέχουσας εικόνας
        img = obs["agentview_image"]
        ep_images.append(img)
        
        # Λήψη δράσης από τον expert controller
        action = controller.act(obs)
        ep_actions.append(action)
        
        # Εκτέλεση βήματος στο MDP
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        
        if done:
            break
            
    # Έλεγχος αν το επεισόδιο ήταν επιτυχές
    if ep_reward > 0.0:
        successful_episodes_counter += 1
        images.extend(ep_images)
        actions.extend(ep_actions)
        print(f"Episode {episode_idx}: SUCCESS! Reward: {ep_reward:.2f}. Total successful collected: {successful_episodes_counter}/{NUM_SUCCESSFUL_EPISODES}")
    else:
        print(f"Episode {episode_idx}: Failed (Reward: {ep_reward:.2f}). Discarding trajectory.")

# Μετατροπή σε numpy arrays
images = np.array(images, dtype=np.uint8)
actions = np.array(actions, dtype=np.float32)

# Αποθήκευση στο δίσκο
np.savez_compressed(OUTPUT_FILE, states=images, actions=actions)

print("\n--- Collection Complete ---")
print(f"Dataset saved to: {os.path.abspath(OUTPUT_FILE)}")
print(f"Total Image Frames Stored: {images.shape}")
print(f"Total Action Vectors Stored: {actions.shape}")

env.close()