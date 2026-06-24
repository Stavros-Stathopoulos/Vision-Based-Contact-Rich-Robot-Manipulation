import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from src.environments.gym_wrapper import RobosuiteGymWrapper


def test_rl_agent(model_path="SAC_10k.zip", num_episodes=5):
    controller_config = load_composite_controller_config(controller="BASIC")

    print("Initializing MuJoCo environment with LIVE Rendering\n")
    raw_env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=500,
        reward_shaping=True,
    )

    env = RobosuiteGymWrapper(raw_env)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}.")
        env.close()
        return

    print(f"Loading trained SAC Agent from: {model_path}\n")
    model = SAC.load(model_path, env=env)

    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        print(f"Starting Episode {episode + 1}\n")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            raw_env.render()

        print(f"Episode {episode + 1} Finished. Total Reward: {ep_reward:.4f}\n")
        episode_rewards.append(ep_reward)

    env.close()

    print("\nGenerating evaluation plot\n")
    plt.figure(figsize=(8, 4))
    plt.bar(
        [f"Ep {i+1}" for i in range(num_episodes)],
        episode_rewards,
        color="teal",
        alpha=0.8,
    )
    plt.title(f"Evaluation Rewards per Episode (SAC {len(episode_rewards)} eps)")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    os.makedirs("./rl_logs", exist_ok=True)
    plt.savefig("./rl_logs/evaluation_live_plot.png")
    print("Plot saved to: ./rl_logs/evaluation_live_plot.png\n")


if __name__ == "__main__":
    test_rl_agent()
