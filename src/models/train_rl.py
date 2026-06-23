import os
import gc
import torch
import ctypes
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src.environments.gym_wrapper import RobosuiteGymWrapper

# Υπερπαράμετροι
TOTAL_TIMESTEPS = 50000  # Συνολικός αριθμός βημάτων εκπαίδευσης
STEPS_PER_BLOCK = 2000  # Κλείσιμο και άνοιγμα του env ανά 2.000 steps για μηδενισμό του leak
LOG_DIR = "./rl_logs"  # Κατάλογος για τα logs του Stable Baselines3 (προαιρετικό, αλλά χρήσιμο για παρακολούθηση)
MODEL_SAVE_PATH = "SAC_50k"  # Αποθήκευση του εκπαιδευμένου μοντέλου

torch.set_num_threads(os.cpu_count())

def clean_memory():
    """Καθολικός καθαρισμός μνήμης Python και C-libraries."""
    gc.collect()
    # Εξαναγκασμός της C βιβλιοθήκης (malloc) να επιστρέψει τη μνήμη στο OS
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass  # Το malloc_trim υποστηρίζεται κυρίως σε Linux, το προσπερνάμε σε Windows

def create_env():
    """Βοηθητική συνάρτηση για τη δημιουργία καθαρού περιβάλλοντος."""
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
        horizon=100,  # Το μέγιστο όριο βημάτων ανά episode. Μπορεί να αυξηθεί αν χρειάζεται.
        reward_shaping=True,
    )
    env = RobosuiteGymWrapper(raw_env)
    env = Monitor(env, LOG_DIR)
    return env

def train_rl_agent():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. Αρχικοποίηση του πρώτου environment
    env = create_env()
    
    print("RL Train: Configuring SAC Agent with CNN Policy\n")
    model = SAC(
        "CnnPolicy",  # CNN policy για επεξεργασία εικόνων
        env,   # Το περιβάλλον που δημιουργήθηκε με το Gym Wrapper
        verbose=1,  # Εμφάνιση λεπτομερειών εκπαίδευσης
        learning_rate=3e-4,  # Learning rate για τον Adam optimizer. Μικρότερο learning rate μπορεί να οδηγήσει σε πιο σταθερή αλλά αργή εκπαίδευση.
        buffer_size=2000,  # Διατήρηση λογικού ορίου για αποφυγή RAM overflow
        batch_size=32,  # Μικρό batch size για πιο συχνές ενημερώσεις του μοντέλου
        device="cuda" if torch.cuda.is_available() else "cpu"  # Χρήση GPU αν είναι διαθέσιμη
    )
    
    trained_steps = 0
    block_count = 1
    
    print(f"Starting Iterative SAC Training for {TOTAL_TIMESTEPS} timesteps\n")
    
    # 2. Το Loop των Blocks (ανά 2.000 βήματα)
    while trained_steps < TOTAL_TIMESTEPS:
        print(f"Starting Block {block_count} ({trained_steps} -> {trained_steps + STEPS_PER_BLOCK} timesteps)\n")
        
        # Εκπαίδευση για το τρέχον block. Το reset_num_timesteps=False κρατάει το global step σταθερό.
        model.learn(total_timesteps=STEPS_PER_BLOCK, reset_num_timesteps=False, progress_bar=True)
        trained_steps += STEPS_PER_BLOCK
        block_count += 1
        
#       3. Σκληρό reset της MuJoCo μνήμης 
        print("Hard resetting environment and purging MuJoCo memory buffers\n")
        env.close()
        clean_memory()
        
        # 4. Δημιουργία νέου περιβάλλοντος και σύνδεση με το υπάρχον μοντέλο
        env = create_env()
        model.set_env(env)  # Σύνδεση του Agent με το νέο καθαρό περιβάλλον
        
    # 5. Αποθήκευση μοντέλου
    model.save(MODEL_SAVE_PATH)
    print(f"Success! RL Policy saved to {MODEL_SAVE_PATH}.zip")
    env.close()

if __name__ == "__main__":
    train_rl_agent()