import os
import gc
import torch
import ctypes
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src.environments.gym_wrapper import RobosuiteGymWrapper

TOTAL_TIMESTEPS = 10000
STEPS_PER_BLOCK = 1000
LOG_DIR = "./rl_logs"
MODEL_SAVE_PATH = "SAC_10k"

torch.set_num_threads(2)


def clean_memory():
    """Force Python and C-level memory release."""
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass


def create_env():
    """Create a fresh robosuite environment wrapped for SB3."""
    controller_config = load_composite_controller_config(controller="BASIC")
    raw_env = suite.make(
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
        horizon=100,
        reward_shaping=True,
    )
    env = RobosuiteGymWrapper(raw_env)
    env = Monitor(env, LOG_DIR)
    return env


def train_rl_agent():
    os.makedirs(LOG_DIR, exist_ok=True)

    clean_memory()
    env = create_env()

    print("RL Train: Configuring SAC Agent with CNN Policy\n")
    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1000,
        batch_size=32,
        learning_starts=100,
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        device="cpu",
    )

    trained_steps = 0
    block_count = 1

    print(f"Starting Iterative SAC Training for {TOTAL_TIMESTEPS} timesteps\n")

    while trained_steps < TOTAL_TIMESTEPS:
        print(
            f"Starting Block {block_count} "
            f"({trained_steps} -> {trained_steps + STEPS_PER_BLOCK} timesteps)\n"
        )

        model.learn(total_timesteps=STEPS_PER_BLOCK, reset_num_timesteps=False)
        trained_steps += STEPS_PER_BLOCK
        block_count += 1

        print("Hard resetting environment and purging MuJoCo memory buffers\n")
        env.close()
        clean_memory()

        env = create_env()
        model.set_env(env)

    model.save(MODEL_SAVE_PATH)
    print(f"Success! RL Policy saved to {MODEL_SAVE_PATH}.zip")
    env.close()
    clean_memory()


if __name__ == "__main__":
    train_rl_agent()
