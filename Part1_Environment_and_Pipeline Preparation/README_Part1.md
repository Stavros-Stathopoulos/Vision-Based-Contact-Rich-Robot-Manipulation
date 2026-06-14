# Phase 1: Environment & Pipeline Setup — Instructions

This document contains step-by-step instructions to set up the clean Python Virtual Environment and run the structural files for **Phase 1** of the Intelligent Control Project on Windows.

---

## 1. Virtual Environment Setup

To isolate the project dependencies and avoid library conflicts, we use Python's built-in `venv`.

### Step 1: Create the Environment
Open PowerShell, navigate to your project root folder (`C:\GitHub\Vision-Based-Contact-Rich-Robot-Manipulation`), and run:
```powershell
python -m venv ic_env
```

### Step 2: Enable Script Execution (Windows Only)

If you get a security/unauthorized access error when trying to activate the environment, allow script execution for your current PowerShell session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Step 3: Activate the Environment

Run the activation script:

```powershell
.\ic_env\Scripts\Activate
```

## 2. Installing Project Dependencies

```powershell
# 1. Clear pip cache and upgrade core packaging tools
pip cache purge
python -m pip install --upgrade pip setuptools wheel

# 2. Install PyTorch & Vision libraries
pip install torch torchvision

# 3. Install robosuite core without downloading its dependencies automatically
pip install robosuite --no-deps

# 4. Install standard wheels required by robosuite & gymnasium
pip install gymnasium termcolor mujoco imageio litellm llvmlite

# 5. Install complex dependencies using the bypass flag to prevent NumPy compilation loops
pip install mink==0.0.5 numba opencv-python pynput pytest scipy qpsolvers quadprog --no-deps
```

### Setup robosuite Macros

Initialize the required macro configuration file by running:

```powershell
python C:\ic_env\Lib\site-packages\robosuite\scripts\setup_macros.py
```

## 3. Project Structure & Execution

Phase 1 consists of three Python scripts.

Execute the scripts in the following order inside your active virtual environment:

### Task 1.1 Verification of Environment 

Verifies that robosuite and MuJoCo initialize correctly, step functions work, and image tensors are received downsampled to $84 \times 84 \times 3$.

```powershell
python Part1_1_Robosuite_Testing.py
```

### Task 1.2 Base Controller Interface 

Defines the BaseController abstract class layout enforcing the mandatory reset() and act(obs) methods via Python's Abstract Base Classes (abc).

```powershell
python Part1_2_Test_Base_Controller_Interface.py
```

### Task 1.3 Vision Pipeline CNN Encoder

Implements a Nature-CNN architecture using PyTorch to encode raw agentview_image matrices into high-quality $256$-dimensional state embeddings.

```powershell
python Part1_3_CNN_Vision_Encoder.py
```

All tasks for Phase 1 are verified and functional. Ready to proceed to Phase 2.