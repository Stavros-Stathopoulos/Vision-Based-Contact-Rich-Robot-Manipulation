import os
import platform
if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import gc
import numpy as np
import torch
import robosuite as suite
from ..environments.make_env import make_controller_config
from .bc_model import BehaviorCloningPolicy

def evaluate_bc_model(model_path="bc_model.pth", num_episodes=20, max_steps=500):
    """
    Live αξιολόγηση της εκπαιδευμένης Behavior Cloning πολιτικής στο περιβάλλον NutAssembly.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading trained Behavior Cloning policy from {model_path}...")
    # Αρχικοποίηση μοντέλου και φόρτωση βαρών
    model = BehaviorCloningPolicy().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  
    
    print("Initializing evaluation environment...")
    controller_config = make_controller_config()
    
    env = suite.make(
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
        horizon=max_steps,
    )

    success_count = 0
    episode_rewards = []
    print(f"\nRunning {num_episodes} Evaluation Episodes...")

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            img = obs["agentview_image"] / 255.0
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action_tensor = model(img_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        if episode_reward > 0.0:
            success_count += 1
            print(f"Episode {episode}: SUCCESS (Reward: {episode_reward:.2f})")
        else:
            print(f"Episode {episode}: FAILED (Reward: {episode_reward:.2f})")
            
        # Καθαρισμός μνήμης MuJoCo/PyTorch ανά επεισόδιο
        if 'img_tensor' in locals():
            del img_tensor
        gc.collect()
        
    success_rate = (success_count / num_episodes) * 100
    mean_reward = np.mean(episode_rewards)
    print("\n--- Evaluation Results ---")
    print(f"Total Episodes Tested: {num_episodes}")
    print(f"Successful Episodes: {success_count}")
    print(f"Final Success Rate: {success_rate:.2f}%")
    print(f"Mean Reward: {mean_reward:.4f}")

    env.close()
    return success_rate

if __name__ == "__main__":
    # Αυτό εκτελείται ΜΟΝΟ αν τρέξεις το αρχείο απευθείας από το terminal
    evaluate_bc_model(model_path="bc_model.pth", num_episodes=20, max_steps=500)