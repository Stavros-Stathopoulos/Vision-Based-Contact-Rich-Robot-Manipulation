import os
import platform
if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import gc
import robosuite as suite
from ..environments.make_env import make_controller_config
from ..controllers.baseline_controller import HeuristicBaselineController

# Ρυθμίσεις συλλογής
NUM_SUCCESSFUL_EPISODES = 10  # Πόσες πλήρεις και επιτυχημένες τροχιές θέλουμε να αποθηκεύσουμε στο dataset μας
MAX_STEPS_PER_EPISODE = 500  # Μέγιστος αριθμός βημάτων ανά επεισόδιο (25 δευτερόλεπτα με 20Hz). Matches benchmark horizon.
MAX_TOTAL_EPISODES = 100 # Μέγιστος αριθμός επεισοδίων
OUTPUT_FILE = "expert_demos.npz"  # Αρχείο όπου θα αποθηκευτούν οι εικόνες και οι δράσεις των επιτυχημένων τροχιών

print("Initializing environment for data collection")
controller_config = make_controller_config()

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

print(f"Starting data collection. Target: {NUM_SUCCESSFUL_EPISODES} successful episodes or max {MAX_TOTAL_EPISODES} attempts.")

while successful_episodes_counter < NUM_SUCCESSFUL_EPISODES and episode_idx < MAX_TOTAL_EPISODES:
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
        action, _ = controller.act(obs)
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
        print(f"Episode {episode_idx}/{MAX_TOTAL_EPISODES}: SUCCESS! Reward: {ep_reward:.2f}. Total successful collected: {successful_episodes_counter}/{NUM_SUCCESSFUL_EPISODES}")
    else:
        print(f"Episode {episode_idx}/{MAX_TOTAL_EPISODES}: Failed. (Reward: {ep_reward:.2f}). Discarding trajectory.")
    
    # Καθαρισμός μνήμης RAM σε κάθε επανάληψη
    del ep_images
    del ep_actions
    gc.collect()

# Έλεγχος αν όντως μαζεύτηκαν δεδομένα πριν την αποθήκευση
if len(images) > 0:
    # Μετατροπή σε numpy arrays
    images = np.array(images, dtype=np.uint8)
    actions = np.array(actions, dtype=np.float32)

    np.savez_compressed(OUTPUT_FILE, states=images, actions=actions)   # Αποθήκευση στο δίσκο

    print("\n--- Collection Complete ---")
    print(f"Dataset saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Total Image Frames Stored: {images.shape}")
    print(f"Total Action Vectors Stored: {actions.shape}")
else:
    print("\n--- Collection Stopped ---")
    print("No successful episodes were collected in 100 attempts. Dataset was not saved.")

env.close()