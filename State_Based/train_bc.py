import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Εισάγουμε το νέο State-Based Policy που φτιάξαμε στο προηγούμενο βήμα
from bc_model import BehaviorCloningStatePolicy 

BATCH_SIZE = 32  
EPOCHS = 20  
LEARNING_RATE = 1e-3  
DATASET_PATH = "expert_demos.npz"
MODEL_SAVE_PATH = "bc_state_model.pth" # <--- Νέο όνομα αρχείου βάρους

def train_bc_model():
    print("Loading Expert Demonstration Dataset (State-Based)\n")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    dataset = np.load(DATASET_PATH)
    states = dataset["states"]    # Σχήμα: (N, input_dim) - Flat vectors
    actions = dataset["actions"]  # Σχήμα: (N, 7)

    # ΑΦΑΙΡΕΘΗΚΕ ΤΟ TRANSPOSE ΚΑΙ Η ΔΙΑΙΡΕΣΗ ΜΕ ΤΟ 255.
    states_tensor = torch.tensor(states, dtype=torch.float32) 
    actions_tensor = torch.tensor(actions, dtype=torch.float32) 

    torch_dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on: {device}\n")
    
    # Διάβασε δυναμικά το μέγεθος εισόδου του state vector
    input_dim = states.shape[1] 
    
    # Αρχικοποίηση του νέου MLP μοντέλου
    model = BehaviorCloningStatePolicy(input_dim=input_dim, action_dim=7).to(device)
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    
    print("Starting State-Based Behavior Cloning Training...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            predictions = model(batch_states)  
            loss = criterion(predictions, batch_actions)  

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

            epoch_loss += loss.item() * batch_states.size(0)

        total_epoch_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch}/{EPOCHS}] - Mean MSE Loss: {total_epoch_loss:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Success! Model weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_bc_model()