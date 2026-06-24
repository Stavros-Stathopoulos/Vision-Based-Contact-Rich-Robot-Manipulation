import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.bc_model import BehaviorCloningPolicy

BATCH_SIZE = 32  # Setting a batch size of 32 for training. This means that the model will process 32 samples at a time before updating the weights. Balance between memory efficiency and gradient estimation stability.
EPOCHS = 20  # Number of complete passes through the entire training dataset. More epochs can lead to better learning but also risk overfitting if too high.
LEARNING_RATE = 1e-3  # Learning rate for the Adam optimizer. Controls the step size during gradient updates.
DATASET_PATH = "expert_demos.npz"
MODEL_SAVE_PATH = "bc_model.pth"

def train_bc_model():
    print("Loading Expert Demonstration Dataset\n")
    if not torch.os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found. Run collect_demos.py first!")
        return

    # 1. Φόρτωση δεδομένων
    dataset = np.load(DATASET_PATH)
    states = dataset["states"]    # Σχήμα: (N, 84, 84, 3) - RGB images
    actions = dataset["actions"]  # Σχήμα: (N, 7) - continuous actions (3 for position, 3 for orientation, 1 for gripper)

    # 2. Κανονικοποίηση σε PyTorch format. Μετατροπή από (N, H, W, C) σε (N, C, H, W) από MuJoco format σε PyTorch format & normalization to [0, 1]
    states = np.transpose(states, (0, 3, 1, 2))  # Μετατροπή σε (N, 3, 84, 84) - PyTorch format
    states_tensor = torch.tensor(states, dtype=torch.float32) / 255.  # Κανονικοποίηση στο [0, 1] 
    actions_tensor = torch.tensor(actions, dtype=torch.float32) 

    # 3. Δημιουργία DataLoader για batch training to avoid memory issues
    torch_dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True)  # batch size of 32 is used and shuffling the data to ensure that the model does not learn any order-based patterns from the dataset.

    # 4. Αρχικοποίηση Network, MSELoss και Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will run on: {device}\n")
    
    model = BehaviorCloningPolicy(action_dim=7).to(device)
    criterion = nn.MSELoss()  # L = Σ||π(s) - a||². It measures the average squared difference between the predicted actions and the expert actions in order to minimize this loss.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer is used to update the model parameters based on the computed gradients from the loss function via Automatic Differentiation / Backpropagation to minize the next step's error.
    
    print("Starting Behavior Cloning Training")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        '''
        The training loop iterates over the dataset for a specified number of epochs. 
        In each epoch, it processes the data in batches, computes the loss, performs backpropagation, and updates the model weights. 
        The mean loss for each epoch is printed to monitor training progress.
        '''
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            # Forward Pass
            predictions = model(batch_states)  # Compute the predicted actions from the model given the current batch of state images.
            loss = criterion(predictions, batch_actions)  # Compute the Mean Squared Error loss between the predicted actions and the expert actions for the current batch.

            # Backward Pass
            optimizer.zero_grad()  # Clear the gradients of all optimized tensors in order to prevent accumulation from previous batches.
            loss.backward()  # Compute the gradient of the loss with respect to the model parameters using backpropagation.It calculates how much each parameter contributed to the loss and stores this information in the .grad attribute of each parameter.

            # Optimization
            optimizer.step()  # Update the model parameters using the computed gradients.

            epoch_loss += loss.item() * batch_states.size(0)

        total_epoch_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch}/{EPOCHS}] - Mean MSE Loss: {total_epoch_loss:.5f}")

    # 5. Save trained model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Success! Behavior Cloning Policy weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_bc_model()