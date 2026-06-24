# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vision-based control of a 7-DOF Franka Emika Panda robot for the NutAssembly task using robosuite/MuJoCo. The robot must grasp, transport, and insert a nut onto a peg using only camera images (`agentview_image` at 84x84) — no privileged object state (`use_object_obs=False`).

## Build & Run

All commands run from repo root with the `.venv` virtual environment active. The package is installed as an editable install (`robot_manipulation`).

```bash
pip install -r requirements.txt       # Install deps (mujoco, robosuite, torch, stable-baselines3)
python main.py                        # Run default episode with SimpleController
```

### Full Pipeline

`run_pipeline.py` orchestrates all training/eval stages as subprocesses (so MuJoCo memory is fully reclaimed between stages):

```bash
python run_pipeline.py              # Run all stages (skips live render)
python run_pipeline.py --all        # Run all stages including live render
python run_pipeline.py --from rl    # Resume from RL training onward
python run_pipeline.py --only bc    # Run only BC training
```

Stages in order: `collect` → `bc` → `eval_bc` → `rl` → `test_rl`

### Individual Training Scripts

```bash
python src/demos/collect_demos.py     # Collect expert demos → expert_demos.npz
python src/models/train_bc.py         # Train BC on demos → bc_model.pth
python src/models/evaluate_bc.py      # Evaluate BC policy
python src/models/train_rl.py         # Train SAC agent → SAC_10k.zip
python src/models/test_rl.py          # Test trained RL agent (requires display)
```

Each script also runs as a module (the form used in `instructions.txt`), e.g. `python -m src.models.train_bc` or `python -m src.environments.test_wrapper`. The editable install puts `src` on the path, so both forms work from repo root.

### Tests

```bash
pytest                                                          # Run all tests
pytest src/controllers/baseline_controller/test_baseline_controller.py  # Single test file
pytest src/environments/test_wrapper.py -k "test_name"          # Single test by name
```

## Architecture

### Data Flow

```
robosuite env (NutAssembly, Panda, 84x84 agentview)
    ↓
RobosuiteGymWrapper (src/environments/gym_wrapper.py)
    — converts robosuite dict obs → Gymnasium (C,H,W) uint8 image space
    — maps terminated/truncated for SB3 compatibility
    ↓
Controller.act(obs) → 7D action vector [delta_pos(3), delta_orient(3), gripper(1)]
    — BASIC composite controller uses OSC_POSE internally
    — OSC maps input [-1, 1] to output [-0.05, 0.05] m per step
    — gripper: -1 = open, +1 = close
```

### Key Abstractions

- **BaseController** (`src/controllers/base_controller.py`): ABC requiring `reset() → None` and `act(obs: dict) → (np.ndarray, dict)`. All controllers must inherit from this.
- **BaseEncoder** (`src/encoders/base_encoder.py`): ABC extending `nn.Module`, requiring `embedding_dim` property and `forward(x) → Tensor`. All feature extractors must inherit from this.
- **CNNEncoder** (`src/encoders/cnn_encoder.py`): 3-layer conv net (84x84x3 → 256-dim embedding). Used by both BC and RL pipelines.
- **RobosuiteGymWrapper** (`src/environments/gym_wrapper.py`): Bridges robosuite's dict-based API to Gymnasium's standard interface for SB3 compatibility.

### Controller Implementations

- `simple_controller/`: Midpoint of action space — used in `main.py` for smoke testing.
- `baseline_controller/`: State-guided FSM expert that uses privileged object state (nut position/quaternion) to grasp the nut handle and place it on the peg. Requires `use_object_obs=True` on the env. Used for demo collection.
- `rl_controller/`: Stub for the RL-trained policy.

### Training Details

- **BC pipeline**: Collects demos via `HeuristicBaselineController`, saves `(images, actions)` pairs to `.npz`, trains `BehaviorCloningPolicy` (CNNEncoder + linear head) with MSE loss.
- **RL pipeline**: Uses SB3's `SAC` with `CnnPolicy` (10k timesteps, buffer=1k, batch=32, `optimize_memory_usage=True`). Trains in blocks of 1k steps with periodic hard env resets to mitigate MuJoCo memory leaks. The system has ~8GB RAM so buffer size and thread count are kept low.

### MuJoCo Memory Leak Pattern

MuJoCo leaks memory on `env.reset()` due to off-screen rendering buffers. Any long-running loop (training, demo collection, evaluation) must periodically: (1) `env.close()`, (2) `gc.collect()`, (3) `ctypes.CDLL(None).malloc_trim(0)` (Linux only), then recreate the env. See `clean_memory()` in `train_rl.py` for the canonical pattern.

## Code Conventions

- **Type hints**: Required on all function signatures (including `Optional`, `Union`, `numpy.typing`).
- **Docstrings**: Google-style on all classes and public methods.
- **Formatting**: PEP 8 via `black`; lint via `flake8`.
- **Logging**: Use `src/utils/logger/logger.py` (`TerminalLogger`). No raw `print` in production modules.
- **Framework isolation**: Core logic stays framework-agnostic; torch/SB3 specifics live in `src/models/` and wrappers.
- **Filename quirk**: `src/environments/__innit__.py` and `src/models/__innit__.py` use a double-n typo. Do not rename unless explicitly authorized.

## Reference Docs

- `README.md`: assignment spec (task, stress-test eval criteria, deliverables).
- `instructions.txt`: install + per-milestone run instructions (in Greek).
- `development.md` / `progress.md`: engineering changelog and milestone progress.
- Note: `README.md`/`instructions.txt` reference a stale RL artifact name (`sac_nut_assembly.zip`); the code saves `SAC_10k.zip`.
