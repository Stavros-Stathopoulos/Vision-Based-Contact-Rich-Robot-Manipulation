import os, torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gym_wrapper import RobosuiteGymWrapper

LOG_DIR = "./rl_logs"
MODEL_SAVE_PATH = "SAC_state_based_50k"

def create_env():
    controller_config = load_composite_controller_config(controller="BASIC")
    raw_env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,         # Απενεργοποίηση καμερών
        use_object_obs=True,          # ΕΝΕΡΓΟΠΟΙΗΣΗ POSITION STATE
        control_freq=20,
        horizon=200,
        reward_shaping=True,
        placement_initializer=None
    )
    env = RobosuiteGymWrapper(raw_env)
    env = Monitor(env, LOG_DIR)
    return env

def train_rl_agent():
    os.makedirs(LOG_DIR, exist_ok=True)
    env = create_env()
    
    print("RL Train: Configuring SAC Agent with State-Based MLP Policy\n")
    model = SAC(
        "MlpPolicy",                  # <--- Χρήση MLP αντί για CNN
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,            # Χωρίς εικόνες, ο buffer καταλαμβάνει ελάχιστη RAM
        batch_size=256,               # Μεγαλύτερο batch για ταχύτερη μάθηση
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Starting SAC Training for 50,000 timesteps...")
    model.learn(total_timesteps=50000, progress_bar=True)
    
    model.save(MODEL_SAVE_PATH)
    print(f"Success! State-Based RL Policy saved to {MODEL_SAVE_PATH}.zip")
    env.close()

if __name__ == "__main__":
    train_rl_agent()