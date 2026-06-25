import os
import platform
if platform.system() == "Windows":
    os.environ.setdefault("MUJOCO_GL", "glfw")

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from src.controllers import SimpleController
from src.utils.logger import TerminalLogger as logger


controller_config = load_composite_controller_config(controller='BASIC')
env = suite.make(
    env_name="NutAssembly",
    robots="Panda",
    gripper_types="PandaGripper",
    controller_configs=controller_config,

    # Vision-based setting
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    use_object_obs=False,

    # Camera configuration
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,

    # Task / Control Settings
    control_freq=20,
    horizon=500,
    reward_shaping=True,
)

controller = SimpleController(env.action_spec)
obs = env.reset()
done = False
episode_reward = 0.0

while not done:
    action, _ = controller.act(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward

logger.info(f"Episode reward: {episode_reward}")
env.close()
