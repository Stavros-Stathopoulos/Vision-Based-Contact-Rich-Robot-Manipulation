import os
import gc
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from src.environments.make_env import make_nut_env
from src.environments.gym_wrapper import RobosuiteGymWrapper
from src.encoders.cnn_encoder import CNNEncoder

# Υπερπαράμετροι
TOTAL_TIMESTEPS = 50000
LOG_DIR = "./rl_logs"  # Κατάλογος για τα logs του Stable Baselines3 (προαιρετικό, αλλά χρήσιμο για παρακολούθηση)
MODEL_SAVE_PATH = "SAC_50k"  # Αποθήκευση του εκπαιδευμένου μοντέλου

# --- Ρυθμίσεις χαμηλών πόρων (για απλούς υπολογιστές χωρίς GPU) ---
BUFFER_SIZE = 30000  # Ο replay buffer είναι ο κύριος καταναλωτής RAM (uint8 εικόνες 3x84x84)
BATCH_SIZE = 64
CPU_THREADS = 4      # Περιορισμός νημάτων ώστε το μηχάνημα να παραμένει αποκρίσιμο

def create_env():
    # Same task as the oracle/BC pipeline (round nut) via the shared factory.
    raw_env = make_nut_env(use_camera_obs=True, horizon=200, reward_shaping=True)
    # reach_reward_scale > 0: πυκνό reward που αυξάνεται όσο ο gripper πλησιάζει το παξιμάδι.
    env = RobosuiteGymWrapper(raw_env, reach_reward_scale=0.5, reach_tanh_scale=10.0)
    env = Monitor(env, LOG_DIR)
    return env

def train_rl_agent():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Σε CPU-only μηχανές, ο περιορισμός των threads αποτρέπει το oversubscription.
    if not torch.cuda.is_available():
        torch.set_num_threads(CPU_THREADS)

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
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        # optimize_memory_usage: αποθηκεύει κάθε observation μία φορά αντί για
        # (obs, next_obs) -> περίπου υποδιπλασιάζει τη RAM του replay buffer.
        # Απαιτεί handle_timeout_termination=False (ασφαλές εδώ: ο wrapper θέτει
        # πάντα truncated=False, οπότε δεν υπάρχει timeout bootstrap να διατηρηθεί).
        optimize_memory_usage=True,
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Starting SAC Training for {TOTAL_TIMESTEPS} timesteps\n")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model.save(MODEL_SAVE_PATH)
    print(f"Training completed. Model saved to {MODEL_SAVE_PATH}\n")
    env.close()

if __name__ == "__main__":
    train_rl_agent()
