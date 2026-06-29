"""Run the trained SAC agent and produce visual output:
  1. A live MuJoCo window (if available)
  2. An MP4 video per episode
  3. Start/Mid/End snapshot composites
  4. A reward bar chart
All output goes to ./visual_results/.
"""

import os
import sys
import platform

if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from stable_baselines3 import SAC

from src import config
from src.environments.make_env import make_raw_env
from src.environments.gym_wrapper import RobosuiteGymWrapper

OUTPUT_DIR = "./visual_results"
NUM_EPISODES = 3
RENDER_HEIGHT = 480
RENDER_WIDTH = 480


def resolve_model_path():
    for candidate in (config.BEST_MODEL_PATH + ".zip", config.MODEL_PATH + ".zip"):
        if os.path.exists(candidate):
            return candidate
    return None


def capture_frame(raw_env):
    frame = raw_env.sim.render(width=RENDER_WIDTH, height=RENDER_HEIGHT, camera_name=config.CAMERA_NAME)
    return np.flipud(frame)


def run_demo():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = resolve_model_path()
    if model_path is None:
        print("ERROR: no trained model found. Run `python -m src.models.train_rl` first.")
        print("Looked for:", config.BEST_MODEL_PATH + ".zip", "and", config.MODEL_PATH + ".zip")
        sys.exit(1)

    try:
        raw_env = make_raw_env(has_renderer=True)
        live_render = True
        print("Live MuJoCo rendering window enabled.")
    except Exception as e:
        print(f"Live rendering unavailable ({e}); offscreen only.")
        raw_env = make_raw_env(has_renderer=False)
        live_render = False

    env = RobosuiteGymWrapper(raw_env)
    print(f"Loading model from {model_path}")
    model = SAC.load(model_path, env=env)

    # Warm up the offscreen renderer.
    env.reset()
    for _ in range(5):
        capture_frame(raw_env)
        if live_render:
            raw_env.render()

    episode_rewards = []
    successes = 0

    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_success = False
        frames = []
        step_count = 0
        print(f"\n--- Episode {ep}/{NUM_EPISODES} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_success = ep_success or bool(info.get("is_success", False))
            done = terminated or truncated
            step_count += 1

            if step_count % 2 == 0:
                frames.append(capture_frame(raw_env))
            if live_render:
                raw_env.render()

        successes += int(ep_success)
        episode_rewards.append(ep_reward)
        print(f"Episode {ep}: reward={ep_reward:.2f} steps={step_count} success={ep_success}")

        if frames:
            video_path = os.path.join(OUTPUT_DIR, f"episode_{ep}.mp4")
            imageio.mimsave(video_path, frames, fps=15)
            print(f"  Video: {video_path}")

            snap_path = os.path.join(OUTPUT_DIR, f"episode_{ep}_snapshot.png")
            mid = len(frames) // 2
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, idx, label in zip(axes, [0, mid, -1], ["Start", "Mid", "End"]):
                ax.imshow(frames[idx])
                ax.set_title(label)
                ax.axis("off")
            fig.suptitle(f"Episode {ep} — reward {ep_reward:.2f} — success {ep_success}")
            plt.tight_layout()
            plt.savefig(snap_path, dpi=100)
            plt.close(fig)
            print(f"  Snapshots: {snap_path}")

    env.close()

    plt.figure(figsize=(8, 4))
    bars = plt.bar([f"Ep {i+1}" for i in range(NUM_EPISODES)], episode_rewards, color="teal", alpha=0.85)
    for bar, val in zip(bars, episode_rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    plt.title(f"SAC Evaluation — {config.TASK_NAME} (success {successes}/{NUM_EPISODES})")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "reward_chart.png")
    plt.savefig(chart_path, dpi=100)
    plt.close()

    print(f"\nResults in {OUTPUT_DIR}/  | true success {successes}/{NUM_EPISODES} "
          f"| mean reward {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    run_demo()
