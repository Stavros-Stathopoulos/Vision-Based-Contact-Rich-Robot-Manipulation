import os
import gc
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from src.environments.gym_wrapper import RobosuiteGymWrapper
from src.encoders.cnn_encoder import CNNEncoder

# Υπερπαράμετροι
TOTAL_TIMESTEPS = 50000
LOG_DIR = "./rl_logs"  # Κατάλογος για τα logs του Stable Baselines3 (προαιρετικό, αλλά χρήσιμο για παρακολούθηση)
MODEL_SAVE_PATH = "SAC_50k"  # Αποθήκευση του εκπαιδευμένου μοντέλου

def create_env():
    controller_config = load_composite_controller_config(controller="BASIC")
    raw_env = suite.make(
        env_name="NutAssembly",
        robots="Panda",
        gripper_types="PandaGripper",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=500,
        reward_shaping=True,
        placement_initializer=None
    )
    env = RobosuiteGymWrapper(raw_env)
    env = Monitor(env, LOG_DIR)
    return env

def train_rl_agent():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Αρχικοποίηση του πρώτου environment
    env = create_env()
    
    policy_kwargs = dict(
        features_extractor_class=CNNEncoder,
        features_extractor_kwargs=dict(embedding_dim=256),
    )

    print("RL Train: Configuring SAC Agent with YOUR Spatial-Aware CNN Encoder\n")
    model = SAC(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=50000,   
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Starting SAC Training for {TOTAL_TIMESTEPS} timesteps\n")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save(MODEL_SAVE_PATH)
    print(f"Training completed. Model saved to {MODEL_SAVE_PATH}\n")
    env.close()

if __name__ == "__main__":
    train_rl_agent()