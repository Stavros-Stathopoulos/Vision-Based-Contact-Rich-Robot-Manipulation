import os
import numpy as np
import gc
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from ..src.controllers.baseline_controller.controller import HeuristicBaselineController
from gym_wrapper import RobosuiteGymWrapper # <--- Προσθήκη του Wrapper

NUM_SUCCESSFUL_EPISODES = 10 
MAX_STEPS_PER_EPISODE = 50  
MAX_TOTAL_EPISODES = 100 
OUTPUT_FILE = "expert_demos.npz"  

print("Initializing environment for data collection (State-Based)")
controller_config = load_composite_controller_config(controller="BASIC")

# 1. Αλλαγή των παραμέτρων για use_object_obs=True
raw_env = suite.make(
    env_name="NutAssembly",
    robots="Panda",
    gripper_types="PandaGripper",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False, # <--- False αφού δεν θέλουμε κάμερες
    use_camera_obs=False,         # <--- False
    use_object_obs=True,          # <--- True (Ενεργοποίηση συντεταγμένων)
    control_freq=20,
    horizon=MAX_STEPS_PER_EPISODE,
)

# 2. Περνάμε το περιβάλλον από τον wrapper
env = RobosuiteGymWrapper(raw_env)
controller = HeuristicBaselineController(action_dim=env.action_space.shape[0])

# Λίστες για flat states (όχι images)
states = []
actions = []

successful_episodes_counter = 0
episode_idx = 0

print(f"Starting data collection. Target: {NUM_SUCCESSFUL_EPISODES} successful episodes.")

while successful_episodes_counter < NUM_SUCCESSFUL_EPISODES and episode_idx < MAX_TOTAL_EPISODES:
    episode_idx += 1
    obs, _ = env.reset() # <--- Ο wrapper επιστρέφει ήδη το έτοιμο 1D Vector
    controller.reset()
    
    ep_states = []
    ep_actions = []
    ep_reward = 0.0
    
    # Χρειαζόμαστε το raw_obs μόνο για τον HeuristicController που διαβάζει τα keys απευθείας
    raw_obs = env.env._get_observations() 
    
    for step in range(MAX_STEPS_PER_EPISODE):
        ep_states.append(obs) # <--- Αποθήκευση του 1D State Vector
        
        action = controller.act(raw_obs)
        ep_actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        raw_obs = env.env._get_observations() 
        ep_reward += reward
        
        if terminated or truncated:
            break
            
    if ep_reward > 0.0:
        successful_episodes_counter += 1
        states.extend(ep_states)
        actions.extend(ep_actions)
        print(f"Episode {episode_idx}: SUCCESS! Collected: {successful_episodes_counter}/{NUM_SUCCESSFUL_EPISODES}")
    else:
        print(f"Episode {episode_idx}: Failed. Discarding trajectory.")
    
    del ep_states
    del ep_actions
    gc.collect()

if len(states) > 0:
    states = np.array(states, dtype=np.float32) # <--- float32 αντί για uint8
    actions = np.array(actions, dtype=np.float32)

    np.savez_compressed(OUTPUT_FILE, states=states, actions=actions)
    print(f"\nDataset saved! States shape: {states.shape}, Actions shape: {actions.shape}")
else:
    print("\nNo successful episodes collected.")
env.close()