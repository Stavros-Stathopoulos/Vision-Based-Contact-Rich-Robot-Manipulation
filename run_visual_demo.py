"""
Run the trained SAC agent and produce visual output:
  1. A live MuJoCo window showing the robot in action
  2. A saved MP4 video of the episode
  3. A reward bar chart across episodes
"""

import os
import sys
import platform

if platform.system() == "Windows":
    os.environ["MUJOCO_GL"] = "glfw"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC

from src.environments.gym_wrapper import RobosuiteGymWrapper
from src.encoders.cnn_encoder import CNNEncoder

OUTPUT_DIR = "./visual_results"
MODEL_PATH = "SAC_100k.zip"
NUM_EPISODES = 3
HORIZON = 500
RENDER_HEIGHT = 480
RENDER_WIDTH = 480


def create_env(live_render=False):
    controller_config = load_composite_controller_config(controller="BASIC")
    raw_env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=live_render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=HORIZON,
        reward_shaping=True,
    )
    return raw_env


def capture_frame(raw_env):
    frame = raw_env.sim.render(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        camera_name="agentview",
    )
    return np.flipud(frame)


def run_demo():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Available .zip files:", [f for f in os.listdir(".") if f.endswith(".zip")])
        sys.exit(1)

    # Try with live rendering first, fall back to offscreen only
    try:
        raw_env = create_env(live_render=True)
        live_render = True
        print("Live MuJoCo rendering window enabled.")
    except Exception as e:
        print(f"Live rendering not available ({e}), using offscreen only.")
        raw_env = create_env(live_render=False)
        live_render = False

    gym_env = RobosuiteGymWrapper(raw_env)

    print(f"Loading trained SAC model from: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH, env=gym_env)

    episode_rewards = []

    # Warmup: render a few frames so the offscreen renderer initializes properly
    obs, _ = gym_env.reset()
    for _ in range(5):
        capture_frame(raw_env)
        if live_render:
            raw_env.render()

    for ep in range(NUM_EPISODES):
        obs, _ = gym_env.reset()
        done = False
        ep_reward = 0.0
        frames = []
        step_count = 0

        print(f"\n--- Episode {ep + 1}/{NUM_EPISODES} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step_count += 1

            if step_count % 2 == 0:
                frame = capture_frame(raw_env)
                frames.append(frame)

            if live_render:
                raw_env.render()

        print(f"Episode {ep + 1} finished: reward={ep_reward:.2f}, steps={step_count}")
        episode_rewards.append(ep_reward)

        if frames:
            video_path = os.path.join(OUTPUT_DIR, f"episode_{ep + 1}.mp4")
            imageio.mimsave(video_path, frames, fps=15)
            print(f"  Video saved: {video_path}")

            snapshot_path = os.path.join(OUTPUT_DIR, f"episode_{ep + 1}_snapshot.png")
            mid = len(frames) // 2
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, idx, label in zip(axes, [0, mid, -1], ["Start", "Mid", "End"]):
                ax.imshow(frames[idx])
                ax.set_title(label)
                ax.axis("off")
            fig.suptitle(f"Episode {ep + 1} — Reward: {ep_reward:.2f}")
            plt.tight_layout()
            plt.savefig(snapshot_path, dpi=100)
            plt.close(fig)
            print(f"  Snapshots saved: {snapshot_path}")

    gym_env.close()

    # Reward bar chart
    plt.figure(figsize=(8, 4))
    bars = plt.bar(
        [f"Ep {i+1}" for i in range(NUM_EPISODES)],
        episode_rewards,
        color="teal",
        alpha=0.8,
    )
    for bar, val in zip(bars, episode_rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    plt.title("SAC Agent Evaluation — NutAssembly")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "reward_chart.png")
    plt.savefig(chart_path, dpi=100)
    plt.close()
    print(f"\nReward chart saved: {chart_path}")

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    run_demo()
