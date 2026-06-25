#!/usr/bin/env python3
"""End-to-end orchestration for the Vision-Based NutAssembly project.

Core task pipeline (run by default), in dependency order:
    1. wrapper   : sanity-check the Gymnasium wrapper (offscreen render)
    2. oracle    : evaluate the scripted oracle expert -> proves the task is solved
    3. collect   : oracle generates SUCCESSFUL demos -> expert_demos.npz
    4. train_bc  : behaviour cloning on the demos    -> bc_model.pth
    5. eval_bc   : roll out the vision BC policy (success rate)

Opt-in RL stages (only with --all or by naming them):
    6. train_rl  : SAC + dense reaching reward       -> SAC_50k.zip
    7. test_rl   : live render + perf plot (needs a display)

The pipeline is RESUMABLE (stages whose output exists are skipped unless --force),
dependency-aware (a stage is skipped if its input is missing), and CPU-friendly
(uses the headless OSMesa GL backend for offscreen stages).

Usage:
    python run_pipeline.py                  # core task pipeline (default)
    python run_pipeline.py --all            # also run train_rl + test_rl
    python run_pipeline.py --force          # rerun, ignoring existing outputs
    python run_pipeline.py --only oracle    # run a single stage
    python run_pipeline.py --skip collect   # skip a stage
    python run_pipeline.py --stages collect train_bc   # explicit subset
"""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RL_MODEL = "SAC_50k.zip"  # must match MODEL_SAVE_PATH in src/models/train_rl.py
REQUIRED = ("robosuite", "mujoco", "torch", "stable_baselines3", "gymnasium")

# (name, requires_artifact_or_None, produces_artifact_or_None, command-as-module-args)
# `requires` is checked at runtime: if the input file is absent (e.g. the upstream
# stage produced nothing), the stage is skipped instead of crashing.
STAGES = [
    ("wrapper",  None,               None,               ["-m", "src.environments.test_wrapper"]),
    ("oracle",   None,               None,               ["-m", "src.models.test_oracle"]),
    ("collect",  None,               "expert_demos.npz", ["-m", "src.demos.collect_demos"]),
    ("train_bc", "expert_demos.npz", "bc_model.pth",     ["-m", "src.models.train_bc"]),
    ("eval_bc",  "bc_model.pth",     None,               ["-m", "src.models.evaluate_bc"]),
    ("train_rl", None,               RL_MODEL,           ["-m", "src.models.train_rl"]),
    # test_rl is handled specially (GUI + correct model path).
]
STAGE_NAMES = [s[0] for s in STAGES] + ["test_rl"]

# The task-performing pipeline (oracle expert -> demos -> vision BC policy -> eval).
# RL stages are opt-in: SAC from pixels won't solve the task on CPU and is slow.
CORE_STAGES = {"wrapper", "oracle", "collect", "train_bc", "eval_bc"}


# --- pretty logging ---
def log(msg):
    print(f"\n\033[1;36m==> {msg}\033[0m", flush=True)


def warn(msg):
    print(f"\033[1;33m[warn] {msg}\033[0m", file=sys.stderr, flush=True)


def die(msg):
    print(f"\033[1;31m[error] {msg}\033[0m", file=sys.stderr, flush=True)
    sys.exit(1)


def find_project_python():
    """Prefer a venv interpreter that actually has the project deps."""
    for venv in ("ic_env", ".venv", "venv", "env"):
        cand = ROOT / venv / "bin" / "python"
        if cand.exists():
            return str(cand)
    return None


def deps_present(python_exe):
    """Return list of missing packages for the given interpreter."""
    if python_exe == sys.executable:
        return [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    code = (
        "import importlib.util,sys;"
        f"miss=[m for m in {REQUIRED!r} if importlib.util.find_spec(m) is None];"
        "print('\\n'.join(miss))"
    )
    out = subprocess.run([python_exe, "-c", code], capture_output=True, text=True)
    return [m for m in out.stdout.split() if m]


def run(python_exe, args, env):
    """Run `python_exe args...` from ROOT, streaming output; raise on failure."""
    subprocess.run([python_exe, *args], cwd=ROOT, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="rerun stages even if their output exists")
    ap.add_argument("--only", metavar="STAGE", choices=STAGE_NAMES,
                    help="run only this stage")
    ap.add_argument("--skip", metavar="STAGE", action="append", default=[],
                    choices=STAGE_NAMES, help="skip this stage (repeatable)")
    ap.add_argument("--stages", nargs="+", metavar="STAGE", choices=STAGE_NAMES,
                    help="explicit subset of stages to run")
    ap.add_argument("--all", action="store_true",
                    help="run every stage including the opt-in RL stages "
                         "(train_rl, test_rl); default runs only the core task pipeline")
    args = ap.parse_args()

    if args.only:
        selected = {args.only}
    elif args.stages:
        selected = set(args.stages)
    elif args.all:
        selected = set(STAGE_NAMES)
    else:
        selected = set(CORE_STAGES)  # default: the task-performing pipeline
    selected -= set(args.skip)

    # --- pick the interpreter that has the deps ---
    python_exe = sys.executable
    missing = deps_present(python_exe)
    if missing:
        venv_py = find_project_python()
        if venv_py and not deps_present(venv_py):
            log(f"Using project interpreter: {venv_py}")
            python_exe = venv_py
            missing = []
    if missing:
        die("Required packages are missing: " + ", ".join(missing) + "\n"
            "  Create the env and install deps first:\n"
            "    python3 -m venv ic_env && source ic_env/bin/activate\n"
            "    pip install --upgrade pip && pip install -r requirements.txt\n"
            "  (Linux also needs: sudo apt-get install -y "
            "libgl1-mesa-dev libosmesa6-dev patchelf)")
    else:
        log("All core dependencies present.")

    # Headless software GL backend for the offscreen-render stages.
    offscreen_env = dict(os.environ)
    offscreen_env.setdefault("MUJOCO_GL", "osmesa")

    results = {}  # stage -> "ok" | "skipped" | "failed"

    # --- run the offscreen / no-GUI stages ---
    for name, requires, output, cmd in STAGES:
        if name not in selected:
            results[name] = "skipped"
            log(f"Skipping stage '{name}' (deselected)")
            continue
        if requires and not (ROOT / requires).exists():
            results[name] = "skipped"
            warn(f"Skipping stage '{name}': prerequisite '{requires}' not found "
                 "(did the upstream stage produce it?).")
            continue
        if not args.force and output and (ROOT / output).exists():
            results[name] = "skipped"
            log(f"Skipping stage '{name}' (output '{output}' exists; --force to rerun)")
            continue
        log(f"Stage '{name}': python {' '.join(cmd)}")
        try:
            run(python_exe, cmd, offscreen_env)
        except subprocess.CalledProcessError as e:
            results[name] = "failed"
            warn(f"Stage '{name}' FAILED (exit {e.returncode}); continuing with the rest.")
            continue
        # A stage can "succeed" yet produce nothing (e.g. collect found no demos).
        if output and not (ROOT / output).exists():
            results[name] = "failed"
            warn(f"Stage '{name}' finished but did not produce '{output}'.")
        else:
            results[name] = "ok"
            log(f"Stage '{name}' done.")

    # --- test_rl: on-screen MuJoCo window + Qt matplotlib backend ---
    if "test_rl" in selected:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            results["test_rl"] = "skipped"
            warn("Skipping 'test_rl': no DISPLAY/WAYLAND_DISPLAY (it requires a GUI).")
        elif not (ROOT / RL_MODEL).exists():
            results["test_rl"] = "skipped"
            warn(f"Skipping 'test_rl': model '{RL_MODEL}' not found (run train_rl first).")
        else:
            log(f"Stage 'test_rl': live render of {RL_MODEL}")
            # Use the system GL backend (not OSMesa) for the on-screen window.
            gui_env = dict(os.environ)
            gui_env.pop("MUJOCO_GL", None)
            try:
                run(python_exe,
                    ["-c", "from src.models.test_rl import test_rl_agent;"
                           f"test_rl_agent(model_path={RL_MODEL!r})"],
                    gui_env)
                results["test_rl"] = "ok"
                log("Stage 'test_rl' done. Plot -> rl_logs/evaluation_live_plot.png")
            except subprocess.CalledProcessError as e:
                results["test_rl"] = "failed"
                warn(f"Stage 'test_rl' FAILED (exit {e.returncode}).")
    else:
        results.setdefault("test_rl", "skipped")

    # --- summary ---
    log("Pipeline summary:")
    icon = {"ok": "\033[1;32mok     \033[0m",
            "skipped": "\033[1;33mskipped\033[0m",
            "failed": "\033[1;31mfailed \033[0m"}
    for name in STAGE_NAMES:
        print(f"   {icon.get(results.get(name, 'skipped'))}  {name}")
    failed = [n for n, s in results.items() if s == "failed"]
    if failed:
        die("Some stages failed: " + ", ".join(failed))
    log("Pipeline complete.")


if __name__ == "__main__":
    main()
