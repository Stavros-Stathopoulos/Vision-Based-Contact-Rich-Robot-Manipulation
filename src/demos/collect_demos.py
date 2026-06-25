"""Collect expert demonstrations for behaviour cloning.

Uses the scripted closed-loop OracleController (privileged-state expert) to perform
the task, and records (agentview_image, action) pairs from SUCCESSFUL episodes only.
The recorded observation is pixels only (agentview_image) so the cloned policy is
vision-based; the oracle's privileged-state access never enters the dataset.
"""
import functools
import gc
import os

import numpy as np

from src.environments.make_env import make_nut_env
from src.controllers.oracle_controller import OracleController

# Tunable via env vars so the pipeline stays tractable on CPU-only machines.
NUM_SUCCESSFUL_EPISODES = int(os.environ.get("DEMO_EPISODES", "15"))
MAX_TOTAL_EPISODES = int(os.environ.get("DEMO_MAX_ATTEMPTS", "25"))
HORIZON = int(os.environ.get("DEMO_HORIZON", "900"))
OUTPUT_FILE = "expert_demos.npz"

print = functools.partial(print, flush=True)  # unbuffered progress when piped


def collect():
    print("Initializing environment + oracle expert for data collection")
    env = make_nut_env(use_camera_obs=True, horizon=HORIZON)
    oracle = OracleController(env)

    images, actions = [], []
    successful = 0
    attempt = 0

    print(f"Target: {NUM_SUCCESSFUL_EPISODES} successful episodes "
          f"(max {MAX_TOTAL_EPISODES} attempts).")
    while successful < NUM_SUCCESSFUL_EPISODES and attempt < MAX_TOTAL_EPISODES:
        attempt += 1
        env.reset()
        oracle.reset()

        ep_images, ep_actions = [], []

        def on_step(action, obs, reward, done, info):
            # record the image observed BEFORE this action and the action taken
            ep_images.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
            ep_actions.append(np.asarray(action, dtype=np.float32))

        ok = oracle.place_nut(env.nut_id, on_step=on_step)

        if ok:
            successful += 1
            images.extend(ep_images)
            actions.extend(ep_actions)
            print(f"Attempt {attempt}/{MAX_TOTAL_EPISODES}: SUCCESS "
                  f"({len(ep_images)} frames). Collected {successful}/{NUM_SUCCESSFUL_EPISODES}.")
        else:
            print(f"Attempt {attempt}/{MAX_TOTAL_EPISODES}: failed. Discarding trajectory.")

        del ep_images, ep_actions
        gc.collect()

    env.close()

    if not images:
        print("\nNo successful episodes collected. Dataset NOT saved.")
        return

    images = np.asarray(images, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.float32)
    np.savez_compressed(OUTPUT_FILE, states=images, actions=actions)
    print("\n--- Collection Complete ---")
    print(f"Saved {OUTPUT_FILE}: states={images.shape}, actions={actions.shape}")
    print(f"({successful} successful episodes, {len(images)} total frames)")


if __name__ == "__main__":
    collect()
