"""Evaluate the trained behaviour-cloning policy on NutAssembly (vision-only).

Success is measured with the env's own `_check_success()` (nut placed on the peg),
NOT reward > 0 -- with reward shaping the reward is positive even on failure.
"""
import gc

import numpy as np
import torch

from src.environments.make_env import make_nut_env
from .bc_model import BehaviorCloningPolicy


def evaluate_bc_model(model_path="bc_model.pth", num_episodes=20, horizon=900):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading trained Behavior Cloning policy from {model_path}...")
    model = BehaviorCloningPolicy().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Initializing evaluation environment (round nut, vision-only)...")
    env = make_nut_env(use_camera_obs=True, horizon=horizon)

    success_count = 0
    print(f"\nRunning {num_episodes} evaluation episodes (horizon={horizon})...")
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        while not done:
            img = obs["agentview_image"] / 255.0
            img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(img_t).squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action)

        ok = bool(env._check_success())
        success_count += ok
        print(f"Episode {episode}: {'SUCCESS' if ok else 'failed'}")
        gc.collect()

    rate = 100.0 * success_count / num_episodes
    print("\n--- Evaluation Results ---")
    print(f"Episodes: {num_episodes} | Successes: {success_count} | Success Rate: {rate:.1f}%")
    env.close()
    return rate


if __name__ == "__main__":
    evaluate_bc_model(model_path="bc_model.pth", num_episodes=20)
