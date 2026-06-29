import os
import argparse
import torch
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from gym_wrapper import RobosuiteGymWrapper
from bc_model import BehaviorCloningStatePolicy

def run_simulation(mode="sac", model_path=None, num_episodes=5):
    controller_config = load_composite_controller_config(controller="BASIC")
    
    print(f"Initializing MuJoCo environment with LIVE Rendering for {mode.upper()} Agent...")
    raw_env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=True,             # ΑΝΟΙΓΕΙ ΤΟ ΠΑΡΑΘΥΡΟ ΣΤΑ WINDOWS
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,           # State-based tracking
        control_freq=20,
        horizon=200,
    )
    
    env = RobosuiteGymWrapper(raw_env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Φόρτωση κατάλληλου μοντέλου
    if mode == "sac":
        if model_path is None: model_path = "SAC_state_based_50k.zip"
        model = SAC.load(model_path)
    elif mode == "bc":
        if model_path is None: model_path = "bc_state_model.pth"
        input_dim = env.observation_space.shape[0]
        model = BehaviorCloningStatePolicy(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        print(f"\n--- Starting Episode {episode + 1} ---")
        
        while not done:
            if mode == "sac":
                action, _ = model.predict(obs, deterministic=True)
            elif mode == "bc":
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(obs_tensor).squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            env.render() # Live update of the MuJoCo window
            
        print(f"Episode {episode + 1} Finished. Accumulated Reward: {episode_reward:.4f}")
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sac", choices=["sac", "bc"], help="Agent mode to simulate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of simulation episodes")
    args = parser.parse_args()
    
    run_simulation(mode=args.mode, num_episodes=args.episodes)