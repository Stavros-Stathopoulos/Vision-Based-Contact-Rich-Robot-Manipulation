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
MODEL_SAVE_PATH = "SAC"  # Αποθήκευση του εκπαιδευμένου μοντέλου

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
    
    # 1. Δημιουργία του πρώτου περιβάλλοντος
    env = create_env()
    
    print("\nRL Train: Configuring SAC Agent with CNN Policy\n")
    model = SAC(
        "CnnPolicy",  # CNN policy για επεξεργασία εικόνων
        env,   # Το περιβάλλον που δημιουργήθηκε με το Gym Wrapper
        verbose=1,  # Εμφάνιση λεπτομερειών εκπαίδευσης
        learning_rate=3e-4,  # Learning rate για τον Adam optimizer. Μικρότερο learning rate μπορεί να οδηγήσει σε πιο σταθερή αλλά αργή εκπαίδευση.
        buffer_size=2000,  # Διατήρηση λογικού ορίου για αποφυγή RAM overflow
        batch_size=32,  # Μικρό batch size για πιο συχνές ενημερώσεις του μοντέλου
        device="cuda" if torch.cuda.is_available() else "cpu"  # Χρήση GPU αν είναι διαθέσιμη
    )
    
    # 2. Τμηματική εκπαίδευση (Iterative Training Loop)
    trained_steps = 0
    block_count = 1
    
    print(f"Starting Iterative SAC Training for {TOTAL_TIMESTEPS} timesteps")
    
    while trained_steps < TOTAL_TIMESTEPS:
        print(f"\nStarting Block {block_count} ({trained_steps} / {TOTAL_TIMESTEPS} timesteps)")
        
        # Εκπαίδευση για ένα μικρό block βημάτων
        model.learn(total_timesteps=STEPS_PER_BLOCK, reset_num_timesteps=False, progress_bar=True)
        trained_steps += STEPS_PER_BLOCK
        block_count += 1
        
        # Σημαντικό: Καταστροφή του τρέχοντος περιβάλλοντος για πλήρη απελευθέρωση της MuJoCo
        print("Hard resetting environment and purging MuJoCo memory buffers...")
        env.close()
        
        # Καθαρισμός μνήμης
        clean_memory()
        
        # Δημιουργία ολοκαίνουργιου instance περιβάλλοντος
        env = create_env()
        model.set_env(env)  # Σύνδεση του Agent με το νέο καθαρό περιβάλλον
        
    # 3. Αποθήκευση μοντέλου
    model.save(MODEL_SAVE_PATH)
    print(f"\nSuccess! RL Policy saved to {MODEL_SAVE_PATH}.zip")
    env.close()

if __name__ == "__main__":
    train_rl_agent()