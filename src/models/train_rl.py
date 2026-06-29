import os
import platform

# Choose the MuJoCo GL backend before robosuite is imported (via make_env).
if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import gc
import ctypes

import numpy as np
import torch
from stable_baselines3 import SAC

from src import config
from src.environments.make_env import make_env
from src.encoders.multimodal_encoder import MultiModalExtractor

# --- Training budget --------------------------------------------------------
# Pixel-based SAC on contact-rich manipulation needs *a lot* of steps. This is a
# real run; 500k-1M is better still. Strongly prefer the CUDA torch build here.
TOTAL_TIMESTEPS = 300_000

# Iterative Block Training: MuJoCo's offscreen renderer leaks memory, so we close
# and rebuild the env between blocks while keeping the model (and its replay
# buffer + optimizers) alive via reset_num_timesteps=False.
STEPS_PER_BLOCK = 2_000
EVAL_EVERY_BLOCKS = 10        # evaluate true success every 20k steps
CHECKPOINT_EVERY_BLOCKS = 25  # periodic safety checkpoint every 50k steps
EVAL_EPISODES = 10

# SAC hyperparameters. buffer_size dominates RAM: a 9-channel 84x84 uint8 image
# stores obs+next_obs, so ~40k transitions ~= 5 GB (fits a 32 GB box). A larger
# buffer is the main fix for the value divergence seen on long runs with a tiny one.
BUFFER_SIZE = 40_000
BATCH_SIZE = 256
LEARNING_STARTS = 5_000  # longer random warmup -> diverse buffer before the policy commits


def purge_memory():
    gc.collect()
    if platform.system() == "Linux":
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_true_success(model, n_episodes: int = EVAL_EPISODES):
    """Honest evaluation: success == env._check_success(), NOT reward > 0.

    Builds and tears down its own env so no long-lived renderer accumulates
    leaked memory across the whole run.
    """
    env = make_env(has_renderer=False)
    successes = 0
    rewards = []
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_success = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_success = ep_success or bool(info.get("is_success", False))
                done = terminated or truncated
            successes += int(ep_success)
            rewards.append(ep_reward)
    finally:
        env.close()
        purge_memory()
    return successes / n_episodes, float(np.mean(rewards))


def train_rl_agent():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Configuring SAC agent (device={device}, task={config.TASK_NAME})\n")

    env = make_env(monitor_dir=config.LOG_DIR)

    policy_kwargs = dict(
        features_extractor_class=MultiModalExtractor,
        features_extractor_kwargs=dict(cnn_dim=256, augment=True),
        normalize_images=False,  # the extractor normalizes the image itself
        net_arch=[256, 256],
    )

    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,  # calmer critic updates -> less value divergence over a long run
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        learning_starts=LEARNING_STARTS,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        # Demand higher policy entropy than the default (-action_dim = -7) so the
        # temperature does not collapse to 0 and kill exploration before a grasp
        # is ever discovered (the reach-and-hover local optimum).
        target_entropy=-3.0,
        # Correct timeout handling: horizon hits arrive as `truncated` and are
        # bootstrapped, not treated as terminal. Requires optimize_memory_usage off.
        replay_buffer_kwargs=dict(handle_timeout_termination=True),
        optimize_memory_usage=False,
        tensorboard_log=config.LOG_DIR,
        device=device,
    )

    num_blocks = TOTAL_TIMESTEPS // STEPS_PER_BLOCK
    print(f"Training {TOTAL_TIMESTEPS} steps in {num_blocks} blocks of {STEPS_PER_BLOCK}\n")

    best_score = (-1.0, float("-inf"))  # (success_rate, mean_reward)
    for block in range(1, num_blocks + 1):
        print(f"\n=== Block {block}/{num_blocks} ===")
        model.learn(
            total_timesteps=STEPS_PER_BLOCK,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        # Tear down the leaky renderer between blocks.
        env.close()
        purge_memory()

        if block % EVAL_EVERY_BLOCKS == 0 or block == num_blocks:
            success_rate, mean_reward = evaluate_true_success(model)
            model.logger.record("eval/true_success_rate", success_rate)
            model.logger.record("eval/mean_reward", mean_reward)
            model.logger.dump(model.num_timesteps)
            print(f"[block {block}] true_success={success_rate:.1%} mean_reward={mean_reward:.3f}")
            # Rank by success first, then mean reward — so while success is stuck at
            # 0 we still keep the highest-reward checkpoint, not the first one.
            score = (success_rate, mean_reward)
            if score > best_score:
                best_score = score
                model.save(config.BEST_MODEL_PATH)
                print(f"  new best (success={success_rate:.1%}, reward={mean_reward:.2f}) -> {config.BEST_MODEL_PATH}.zip")

        if block % CHECKPOINT_EVERY_BLOCKS == 0:
            model.save(config.MODEL_PATH)  # overwrite a resumable snapshot
            print(f"  checkpoint saved -> {config.MODEL_PATH}.zip ({model.num_timesteps} steps)")

        if block < num_blocks:
            env = make_env(monitor_dir=config.LOG_DIR)
            model.set_env(env)

    model.save(config.MODEL_PATH)
    print(f"\nTraining complete. Final model: {config.MODEL_PATH}.zip | "
          f"best success: {best_score[0]:.1%} (reward {best_score[1]:.2f})\n")


if __name__ == "__main__":
    train_rl_agent()
