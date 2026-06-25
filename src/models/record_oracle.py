"""Record the oracle expert solving NutAssembly to an MP4, so you can watch it.

Uses a dedicated higher-resolution camera ONLY for the video (the oracle drives the
env from privileged state, independent of camera resolution). Run:
    python -m src.models.record_oracle --episodes 2 --out oracle_demo.mp4
"""
import argparse

import imageio
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from src.controllers.oracle_controller import OracleController


def record(out="oracle_demo.mp4", episodes=2, cam="frontview", size=512, horizon=900, fps=40):
    cfg = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        env_name="NutAssembly", robots="Panda", gripper_types="PandaGripper",
        controller_configs=cfg, has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, use_object_obs=False,
        camera_names=cam, camera_heights=size, camera_widths=size,
        control_freq=20, horizon=horizon,
        single_object_mode=2, nut_type="round", reward_shaping=True,
    )
    oracle = OracleController(env)
    frames = []

    for ep in range(episodes):
        env.reset()
        oracle.reset()

        def on_step(action, obs, reward, done, info):
            img = np.asarray(obs[f"{cam}_image"], dtype=np.uint8)
            frames.append(img[::-1])  # robosuite images are vertically flipped

        ok = oracle.place_nut(env.nut_id, on_step=on_step)
        print(f"Episode {ep + 1}: {'SUCCESS' if ok else 'failed'} ({len(frames)} frames so far)")

    env.close()
    imageio.mimsave(out, frames, fps=fps)
    print(f"\nSaved {len(frames)} frames to {out} ({len(frames) / fps:.1f}s at {fps} fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="oracle_demo.mp4")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--camera", default="frontview")
    args = p.parse_args()
    record(out=args.out, episodes=args.episodes, cam=args.camera)
