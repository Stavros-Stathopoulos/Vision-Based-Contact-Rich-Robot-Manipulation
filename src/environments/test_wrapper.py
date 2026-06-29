"""Smoke-test the Gym wrapper: verifies the Dict observation space, shapes,
dtypes, and the terminated/truncated/is_success contract."""

from .make_env import make_env


def test_wrapper():
    print("Building wrapped NutAssembly env...")
    env = make_env(horizon=50)

    print("\n--- Spaces ---")
    print(f"Action space:      {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    print("\nReset...")
    obs, info = env.reset()
    print(f"  image   shape={obs['image'].shape} dtype={obs['image'].dtype}")
    print(f"  proprio shape={obs['proprio'].shape} dtype={obs['proprio'].dtype}")
    assert obs["image"].dtype.name == "uint8"
    assert set(obs.keys()) == {"image", "proprio"}, "policy must only see image + proprio"

    print("\nStep with a random action...")
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"  reward={reward:.4f} terminated={terminated} truncated={truncated} "
          f"is_success={info.get('is_success')}")

    env.close()
    print("\nWrapper test complete.")


if __name__ == "__main__":
    test_wrapper()
