"""Unified benchmark evaluation + baseline comparison.

Runs any controller that implements the common interface
(`reset(seed)` / `act(observation) -> (action, info)`) on the **unmodified**
robosuite benchmark env and reports success rate, mean reward, and episode length.

Compliance notes:
* Uses `make_raw_env` (the bare benchmark env), NOT the training wrapper — so the
  benchmark's own termination/horizon are in effect. We only *read*
  `env._check_success()` for scoring; we never change when episodes end.
* Controllers receive the raw benchmark observation dict, exactly as the official
  scoring script would pass it.
* `randomization_scale=1.0` (default here) = full benchmark randomization, to
  mirror official / stress-test conditions.

Examples:
    python -m src.evaluate --controller all --episodes 20
    python -m src.evaluate --controller sac --episodes 50 --render
"""

import os
import platform

if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import numpy as np

from src.environments.make_env import make_raw_env
from src import config

CONTROLLERS = ("simple", "heuristic", "bc", "sac")


def build_controller(name: str, env):
    if name == "simple":
        from src.controllers import SimpleController
        return SimpleController(env.action_spec)
    if name == "heuristic":
        from src.controllers import HeuristicBaselineController
        return HeuristicBaselineController(action_dim=env.action_spec[0].shape[0])
    if name == "bc":
        from src.controllers import BCController
        return BCController()
    if name == "sac":
        from src.controllers import RLController
        return RLController()
    raise ValueError(f"unknown controller '{name}'")


def evaluate(
    name: str,
    episodes: int = 20,
    horizon: int = config.HORIZON,
    randomization_scale: float = 1.0,
    render: bool = False,
    seed0: int = 0,
):
    env = make_raw_env(has_renderer=render, horizon=horizon, randomization_scale=randomization_scale)
    controller = build_controller(name, env)

    successes = 0
    rewards, lengths = [], []
    for ep in range(episodes):
        np.random.seed(seed0 + ep)  # reproducibility for OUR eval (not official seeds)
        obs = env.reset()
        controller.reset(seed=seed0 + ep)
        done = False
        ep_reward = 0.0
        steps = 0
        success = False
        while not done:
            action, _ = controller.act(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1
            if env._check_success():  # read-only scoring; does not alter termination
                success = True
            if render:
                env.render()
        successes += int(success)
        rewards.append(ep_reward)
        lengths.append(steps)
        print(f"  [{name}] ep {ep + 1}/{episodes}: success={success} reward={ep_reward:.3f} len={steps}")

    env.close()
    result = dict(
        controller=name,
        success_rate=successes / episodes,
        mean_reward=float(np.mean(rewards)),
        mean_len=float(np.mean(lengths)),
    )
    print(f"=== {name}: success_rate={result['success_rate']:.1%} "
          f"mean_reward={result['mean_reward']:.3f} mean_len={result['mean_len']:.1f}\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="all", choices=CONTROLLERS + ("all",))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=config.HORIZON)
    parser.add_argument("--randomization-scale", type=float, default=1.0,
                        help="1.0 = full benchmark randomization (default)")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    names = list(CONTROLLERS) if args.controller == "all" else [args.controller]
    results = []
    for name in names:
        try:
            results.append(evaluate(
                name,
                episodes=args.episodes,
                horizon=args.horizon,
                randomization_scale=args.randomization_scale,
                render=args.render,
            ))
        except Exception as exc:  # missing model file, etc. — skip but keep going
            print(f"  [{name}] skipped: {exc}\n")

    if results:
        print("==== Comparison (benchmark, full randomization) ====")
        print(f"{'controller':<12}{'success':>10}{'reward':>12}{'ep_len':>9}")
        for r in results:
            print(f"{r['controller']:<12}{r['success_rate']:>9.1%}{r['mean_reward']:>12.3f}{r['mean_len']:>9.1f}")


if __name__ == "__main__":
    main()
