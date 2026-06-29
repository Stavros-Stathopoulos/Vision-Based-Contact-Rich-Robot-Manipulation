"""Single factory for the NutAssembly environment.

Previously every entry script re-declared the same `suite.make(...)` block, so a
change to one knob (horizon, camera, task) had to be made in six places. Build
environments through here instead.
"""

import os
import platform

# MuJoCo needs a GL backend chosen before robosuite is imported.
if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import robosuite as suite
from stable_baselines3.common.monitor import Monitor

from .gym_wrapper import RobosuiteGymWrapper
from ..config import (
    TASK_NAME,
    ROBOT,
    GRIPPER,
    CONTROLLER,
    CAMERA_NAME,
    IMG_SIZE,
    CONTROL_FREQ,
    HORIZON,
    REWARD_SHAPING,
    NUM_STACK,
    RANDOMIZATION_SCALE,
)

# robosuite NutAssembly geometry (from robosuite/environments/manipulation/nut_assembly.py).
# The default per-nut spawn boxes, expressed relative to TABLE_OFFSET.
_TABLE_OFFSET = (0.0, 0.0, 0.82)
_DEFAULT_NUT_RANGES = {
    "SquareNutSampler": dict(x_range=[-0.115, -0.11], y_range=[0.11, 0.225]),
    "RoundNutSampler": dict(x_range=[-0.115, -0.11], y_range=[-0.225, -0.11]),
}


def _scaled(rng, scale):
    """Shrink an [a, b] interval around its center by `scale`."""
    center = (rng[0] + rng[1]) / 2.0
    half = (rng[1] - rng[0]) / 2.0 * scale
    return [center - half, center + half]


def make_placement_initializer(scale: float):
    """Curriculum sampler: replicates robosuite's default NutAssembly placement
    but with spawn boxes and yaw scaled down by `scale`. Sub-sampler names match
    what NutAssembly expects (`{NutName}Sampler`); objects are attached by the env."""
    from robosuite.utils.placement_samplers import (
        SequentialCompositeSampler,
        UniformRandomSampler,
    )

    sampler = SequentialCompositeSampler(name="ObjectSampler")
    for name, rng in _DEFAULT_NUT_RANGES.items():
        sampler.append_sampler(
            UniformRandomSampler(
                name=name,
                x_range=_scaled(rng["x_range"], scale),
                y_range=_scaled(rng["y_range"], scale),
                rotation=(-np.pi * scale, np.pi * scale),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=np.array(_TABLE_OFFSET),
                z_offset=0.02,
            )
        )
    return sampler


def make_controller_config():
    """Works across robosuite versions: 1.5+ uses composite controllers
    (`controller="BASIC"`), 1.4.x uses `default_controller="OSC_POSE"`. Both give
    the Panda a 7-D action (XYZ + RPY + gripper)."""
    try:  # robosuite >= 1.5
        from robosuite.controllers import load_composite_controller_config
        return load_composite_controller_config(controller=CONTROLLER)
    except ImportError:  # robosuite 1.4.x
        from robosuite.controllers import load_controller_config
        return load_controller_config(default_controller="OSC_POSE")


def make_raw_env(
    has_renderer: bool = False,
    horizon: int = HORIZON,
    task_name: str = TASK_NAME,
    randomization_scale: float = RANDOMIZATION_SCALE,
):
    """The bare robosuite env (vision-only: use_object_obs=False)."""
    controller_config = make_controller_config()
    # scale >= 1.0 -> use robosuite's own (full) randomization; smaller -> curriculum
    placement_initializer = (
        make_placement_initializer(randomization_scale) if randomization_scale < 1.0 else None
    )
    return suite.make(
        env_name=task_name,
        robots=ROBOT,
        gripper_types=GRIPPER,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=CAMERA_NAME,
        camera_heights=IMG_SIZE,
        camera_widths=IMG_SIZE,
        control_freq=CONTROL_FREQ,
        horizon=horizon,
        reward_shaping=REWARD_SHAPING,
        placement_initializer=placement_initializer,
    )


def make_env(
    has_renderer: bool = False,
    horizon: int = HORIZON,
    num_stack: int = NUM_STACK,
    monitor_dir: str | None = None,
    task_name: str = TASK_NAME,
    randomization_scale: float = RANDOMIZATION_SCALE,
):
    """Gymnasium-wrapped env ready for SB3 (optionally Monitor-wrapped)."""
    env = RobosuiteGymWrapper(
        make_raw_env(
            has_renderer=has_renderer,
            horizon=horizon,
            task_name=task_name,
            randomization_scale=randomization_scale,
        ),
        num_stack=num_stack,
    )
    if monitor_dir is not None:
        env = Monitor(env, monitor_dir, info_keywords=("is_success",))
    return env
