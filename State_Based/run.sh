#!/bin/bash
# 1. Δίνεις δικαιώματα εκτέλεσης στο script
#chmod +x State_Based/run.sh
# 2. Τρέχεις το αυτοματοποιημένο pipeline
#./State_Based/run.sh
# Clear terminal screen
#clear

echo "====================================================================="
echo "   STARTING AUTOMATED PIPELINE: STATE-BASED ROBOT MANIPULATION"
echo "====================================================================="
echo ""

# 1. Καθορισμός του Python Interpreter από το Virtual Environment
VENV_PYTHON="./ic_env/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_PYTHON"
    echo "Please ensure your virtual environment is correctly configured."
    exit 1
fi

# 2. Βήμα 1: Συλλογή Expert Demonstrations
echo "[STEP 1/4] Collecting Expert Demonstrations (State-Based)..."
$VENV_PYTHON -I -m State_Based.collect_demos
if [ $? -ne 0 ]; then
    echo "[ERROR] Data collection failed. Exiting..."
    exit 1
fi
echo "[SUCCESS] Expert demos collected successfully."
echo ""

# 3. Βήμα 2: Εκπαίδευση Behavior Cloning (BC)
echo "[STEP 2/4] Training Behavior Cloning Policy (MLP)..."
$VENV_PYTHON -I -m State_Based.train_bc
if [ $? -ne 0 ]; then
    echo "[ERROR] Behavior Cloning training failed. Exiting..."
    exit 1
fi
echo "[SUCCESS] Behavior Cloning model weights saved."
echo ""

# 4. Βήμα 3: Εκπαίδευση Reinforcement Learning (SAC)
echo "[STEP 3/4] Training Reinforcement Learning Agent (SAC - 50k Steps)..."
$VENV_PYTHON -I -m State_Based.train_rl
if [ $? -ne 0 ]; then
    echo "[ERROR] Reinforcement Learning training failed. Exiting..."
    exit 1
fi
echo "[SUCCESS] SAC Agent policy saved."
echo ""

# 5. Βήμα 4: Live Προσομοίωση και Αξιολόγηση
echo "[STEP 4/4] Launching Live MuJoCo Simulation..."
echo "Running 5 evaluation episodes using the trained SAC Agent..."
$VENV_PYTHON -I -m State_Based.simulate_agents --mode sac --episodes 5

echo ""
echo "====================================================================="
echo "   PIPELINE COMPLETE! All agents trained and evaluated successfully."
echo "====================================================================="
