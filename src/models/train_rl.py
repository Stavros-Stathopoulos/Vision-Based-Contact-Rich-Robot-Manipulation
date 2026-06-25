import os
import platform
if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import gc
import ctypes
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from src.environments.gym_wrapper import RobosuiteGymWrapper
from src.encoders.cnn_encoder import CNNEncoder

TOTAL_TIMESTEPS = 50000
STEPS_PER_BLOCK = 2000
LOG_DIR = "./rl_logs"
MODEL_SAVE_PATH = "SAC_50k"


def create_env():
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
        horizon=500,
        reward_shaping=True,
        placement_initializer=None
    )
    env = RobosuiteGymWrapper(raw_env)
    env = Monitor(env, LOG_DIR)
    return env


def purge_memory():
    gc.collect()
    if platform.system() == "Linux":
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except Exception:
            pass
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def train_rl_agent():
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    env = create_env()

    policy_kwargs = dict(
        features_extractor_class=CNNEncoder,
        features_extractor_kwargs=dict(embedding_dim=256),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Configuring SAC Agent (device={device})\n")

    model = SAC(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=20000,
        batch_size=64,
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        optimize_memory_usage=True,
        device=device,
    )

    num_blocks = TOTAL_TIMESTEPS // STEPS_PER_BLOCK
    print(f"Training {TOTAL_TIMESTEPS} timesteps in {num_blocks} blocks of {STEPS_PER_BLOCK}\n")

    for block in range(1, num_blocks + 1):
        print(f"\n=== Block {block}/{num_blocks} ===")
        model.learn(
            total_timesteps=STEPS_PER_BLOCK,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        # Purge MuJoCo rendering memory between blocks
        env.close()
        purge_memory()

        if block < num_blocks:
            env = create_env()
            model.set_env(env)

        # Checkpoint every 5 blocks
        if block % 5 == 0:
            checkpoint_path = f"{MODEL_SAVE_PATH}_block{block}"
            model.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    model.save(MODEL_SAVE_PATH)
    print(f"\nTraining completed. Model saved to {MODEL_SAVE_PATH}\n")


if __name__ == "__main__":
    train_rl_agent()
