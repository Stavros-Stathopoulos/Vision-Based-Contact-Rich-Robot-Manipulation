import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from src.environments.gym_wrapper import RobosuiteGymWrapper

def test_rl_agent(model_path="SAC.zip", num_episodes=5):
    controller_config = load_composite_controller_config(controller="BASIC")
    
    print("Αρχικοποίηση περιβάλλοντος MuJoCo με LIVE Rendering\n")
    # Ενεργοποιούμε το has_renderer=True για να ανοίξει το παράθυρο στα Windows σου
    raw_env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=True,             # <--- ΑΝΟΙΓΕΙ ΤΟ ΠΑΡΑΘΥΡΟ ΣΤΑ WINDOWS
        has_offscreen_renderer=True,   # <--- ΕΠΙΤΡΕΠΕΙ ΣΤΗΝ ΚΑΜΕΡΑ ΝΑ ΒΛΕΠΕΙ
        use_camera_obs=True,
        use_object_obs=False,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=500,
    )
    
    # Στην αξιολόγηση απενεργοποιούμε το dense reaching shaping ώστε το reported
    # reward να αντικατοπτρίζει την πραγματική απόδοση στο task (όχι το shaping).
    env = RobosuiteGymWrapper(raw_env, reach_reward_scale=0.0)

    if not os.path.exists(model_path):
        print(f"Δεν βρέθηκε το αρχείο μοντέλου: {model_path}.\n")
        return
        
    print(f"Φόρτωση του εκπαιδευμένου SAC Agent από: {model_path}\n")
    model = SAC.load(model_path, env=env)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        print(f"Έναρξη Episode {episode + 1} στην οθόνη\n")
        
        while not done:
            # deterministic=True για να δούμε τι ακριβώς έμαθε ο Agent στα 10k steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
            # Σχεδιάζει live το animation στο παράθυρο των Windows
            raw_env.render() 
            
        print(f"Episode {episode + 1} Finished. Total Reward: {ep_reward:.4f}\n")
        episode_rewards.append(ep_reward)
        
    env.close()
    
    # --- ΣΧΕΔΙΑΣΗ ΔΙΑΓΡΑΜΜΑΤΟΣ ΑΠΕΥΘΕΙΑΣ ΑΠΟ ΤΑ LIVE REWARDS ---
    print("\nΠαραγωγή διαγράμματος\n")
    plt.figure(figsize=(8, 4))
    plt.bar([f"Ep {i+1}" for i in range(num_episodes)], episode_rewards, color='teal', alpha=0.8)
    plt.title("Evaluation Rewards per Episode (SAC 10k Steps Checkpoint)")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs("./rl_logs", exist_ok=True)
    plt.savefig("./rl_logs/evaluation_live_plot.png")
    print("Το διάγραμμα αποθηκεύτηκε επιτυχώς στο: ./rl_logs/evaluation_live_plot.png\n")

if __name__ == "__main__":
    test_rl_agent()