"""One-time robosuite 1.4.1 fix-ups for Windows.

robosuite 1.4.1 does not run out-of-the-box on Windows. Two patches are needed
inside the installed package (so they must be re-applied after every fresh
`pip install -r requirements.txt`):

  1. Copy `mujoco.dll` from the `mujoco` package into `robosuite/utils/`, which
     robosuite hard-loads but does not ship.
  2. Disable GPU rendering via `robosuite/macros_private.py`, because robosuite
     forces `MUJOCO_GL=egl` (Linux-only) when GPU rendering is on, which raises
     on Windows. Offscreen rendering then uses the GLFW context.

The robosuite<->1.5 controller-API difference is handled in code
(`src/environments/make_env.py`), not here.

Run once after installing requirements:

    python scripts/setup_robosuite_windows.py
"""

import importlib.util
import os
import shutil


def _package_dir(name: str) -> str | None:
    spec = importlib.util.find_spec(name)  # does not execute the package
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    return os.path.dirname(spec.origin) if spec.origin else None


def fix_mujoco_dll() -> None:
    robosuite_dir = _package_dir("robosuite")
    mujoco_dir = _package_dir("mujoco")
    if not robosuite_dir or not mujoco_dir:
        print("  ! robosuite or mujoco not installed; skipping DLL copy")
        return

    dest = os.path.join(robosuite_dir, "utils", "mujoco.dll")
    src = os.path.join(mujoco_dir, "mujoco.dll")
    if os.path.exists(dest):
        print(f"  = mujoco.dll already present: {dest}")
        return
    if not os.path.exists(src):
        print(f"  ! source mujoco.dll not found at {src}")
        return
    shutil.copyfile(src, dest)
    print(f"  + copied mujoco.dll -> {dest}")


def fix_gpu_rendering_macro() -> None:
    robosuite_dir = _package_dir("robosuite")
    if not robosuite_dir:
        print("  ! robosuite not installed; skipping macro override")
        return

    path = os.path.join(robosuite_dir, "macros_private.py")
    if os.path.exists(path):
        print(f"  = macros_private.py already present: {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Disable GPU (egl) rendering, unsupported on Windows in robosuite 1.4.1.\n"
            "MUJOCO_GPU_RENDERING = False\n"
        )
    print(f"  + wrote {path}")


if __name__ == "__main__":
    print("Applying robosuite Windows fix-ups...")
    fix_mujoco_dll()
    fix_gpu_rendering_macro()
    print("Done. Verify with: python -m src.environments.test_wrapper")
