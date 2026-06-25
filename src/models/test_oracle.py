"""Evaluate the scripted oracle expert on NutAssembly and report success rate.

This is the baseline that demonstrably PERFORMS the task (closed-loop control using
privileged simulator state). Run:  python -m src.models.test_oracle
"""
import argparse

from src.environments.make_env import make_nut_env
from src.controllers.oracle_controller import OracleController


def evaluate_oracle(num_episodes=10, horizon=900, render=False):
    env = make_nut_env(use_camera_obs=False, has_renderer=render, horizon=horizon)
    oracle = OracleController(env)

    successes = 0
    print(f"Evaluating oracle for {num_episodes} episodes (round nut)...")
    for ep in range(1, num_episodes + 1):
        env.reset()
        oracle.reset()
        ok = oracle.place_nut(env.nut_id)
        if render:
            env.render()
        successes += ok
        print(f"Episode {ep}: {'SUCCESS' if ok else 'failed'}")

    rate = 100.0 * successes / num_episodes
    print(f"\nOracle success rate: {successes}/{num_episodes} ({rate:.1f}%)")
    env.close()
    return rate


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    evaluate_oracle(num_episodes=args.episodes, render=args.render)
