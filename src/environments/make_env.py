"""Single source of truth for building the NutAssembly environment.

Every script (demo collection, BC/RL training, evaluation) builds the env through
`make_nut_env(...)` so the configuration stays consistent. Vision-based setting is
enforced: `use_object_obs=False` (the *policy* only ever sees pixels). Privileged
state is still readable from `env.sim` for the oracle expert and reward shaping —
that is training-time only and does not enter the policy observation.
"""
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# Default task: the round nut (single_object_mode=2). It is rotation-symmetric, so
# the oracle solves it reliably (no square-hole/peg yaw alignment needed).
DEFAULT_SINGLE_OBJECT_MODE = 2
DEFAULT_NUT_TYPE = "round"
IMG_SIZE = 84


def make_nut_env(
    *,
    use_camera_obs=True,
    has_renderer=False,
    has_offscreen_renderer=True,
    horizon=700,
    reward_shaping=True,
    single_object_mode=DEFAULT_SINGLE_OBJECT_MODE,
    nut_type=DEFAULT_NUT_TYPE,
    control_freq=20,
):
    """Build a raw robosuite NutAssembly env.

    Args:
        use_camera_obs: expose the agentview image (needed for vision policies).
        has_renderer: open an on-screen MuJoCo window (needs a display).
        has_offscreen_renderer: enable offscreen rendering (needed for camera obs).
        horizon: max steps per episode. The scripted oracle needs a long horizon
            (~600+ steps per nut); learned policies can use a shorter one.
        single_object_mode: 0=both nuts, 1=either nut, 2=the `nut_type` nut only.
        nut_type: "round" or "square" (only used when single_object_mode==2).
    """
    controller_config = load_composite_controller_config(controller="BASIC")
    return suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        use_camera_obs=use_camera_obs,
        use_object_obs=False,            # vision-only policy observation
        camera_names="agentview",
        camera_heights=IMG_SIZE,
        camera_widths=IMG_SIZE,
        control_freq=control_freq,
        horizon=horizon,
        reward_shaping=reward_shaping,
        single_object_mode=single_object_mode,
        nut_type=nut_type,
    )
