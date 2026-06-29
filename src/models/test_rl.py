import os
import platform

if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # avoid tkinter backend crashes on Windows
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from src import config
from src.environments.make_env import make_env


def test_rl_agent(model_path: str | None = None, num_episodes: int = 5):
    # Prefer the best-by-true-success checkpoint, fall back to the final model.
    if model_path is None:
        best = config.BEST_MODEL_PATH + ".zip"
        model_path = best if os.path.exists(best) else config.MODEL_PATH + ".zip"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Train first with `python -m src.models.train_rl`.")
        return

    print(f"Opening live MuJoCo window and loading agent from {model_path}\n")
    env = make_env(has_renderer=True)
    model = SAC.load(model_path, env=env)

    episode_rewards = []
    successes = 0

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_success = False
        print(f"Episode {episode}/{num_episodes}...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_success = ep_success or bool(info.get("is_success", False))
            done = terminated or truncated
            env.render()  # live animation

        successes += int(ep_success)
        episode_rewards.append(ep_reward)
        print(f"  reward={ep_reward:.3f}  success={ep_success}")

    env.close()

    print(f"\nTrue success rate: {successes / num_episodes:.1%} "
          f"({successes}/{num_episodes})  mean reward: {np.mean(episode_rewards):.3f}\n")

    # --- Reward plot ---
    colors = ["seagreen" if r == max(episode_rewards) else "teal" for r in episode_rewards]
    plt.figure(figsize=(8, 4))
    plt.bar([f"Ep {i+1}" for i in range(num_episodes)], episode_rewards, color=colors, alpha=0.85)
    plt.title(f"SAC Evaluation — {config.TASK_NAME} (success {successes}/{num_episodes})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    out = os.path.join(config.LOG_DIR, "evaluation_live_plot.png")
    plt.savefig(out)
    plt.close()
    print(f"Plot saved to {out}\n")


if __name__ == "__main__":
    test_rl_agent()
