# .\run.bat 

@echo off
chcp 65001 > nul
echo =====================================================================
echo    STARTING AUTOMATED PIPELINE: STATE-BASED ROBOT MANIPULATION
echo =====================================================================
echo.

:: 1. Καθορισμός των paths της Python του Virtual Environment
set VENV_PYTHON=.\ic_env\Scripts\python.exe

if not exist %VENV_PYTHON% (
    echo [ERROR] Virtual environment not found at .\ic_env\Scripts\python.exe
    echo Please make sure your virtual environment is correctly configured.
    pause
    exit /b
)

:: 2. Βήμα 1: Συλλογή Expert Demonstrations
echo [STEP 1/4] Collecting Expert Demonstrations (State-Based)...
%VENV_PYTHON% -I -m State_Based.collect_demos
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data collection failed. Exiting...
    pause
    exit /b
)
echo [SUCCESS] Expert demos collected successfully.
echo.

:: 3. Βήμα 2: Εκπαίδευση Behavior Cloning (BC)
echo [STEP 2/4] Training Behavior Cloning Policy (MLP)...
%VENV_PYTHON% -I -m State_Based.train_bc
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Behavior Cloning training failed. Exiting...
    pause
    exit /b
)
echo [SUCCESS] Behavior Cloning model weights saved.
echo.

:: 4. Βήμα 3: Εκπαίδευση Reinforcement Learning (SAC)
echo [STEP 3/4] Training Reinforcement Learning Agent (SAC - 50k Steps)...
%VENV_PYTHON% -I -m State_Based.train_rl
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Reinforcement Learning training failed. Exiting...
    pause
    exit /b
)
echo [SUCCESS] SAC Agent policy saved.
echo.

:: 5. Βήμα 4: Live Προσομοίωση και Αξιολόγηση
echo [STEP 4/4] Launching Live MuJoCo Simulation...
echo Running 5 evaluation episodes using the trained SAC Agent...
%VENV_PYTHON% -I -m State_Based.simulate_agents --mode sac --episodes 5

echo.
echo =====================================================================
echo    PIPELINE COMPLETE! All agents trained and evaluated successfully.
echo =====================================================================
pause