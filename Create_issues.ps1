$REPO = "Stavros-Stathopoulos/Vision-Based-Contact-Rich-Robot-Manipulation"

Write-Host "Starting creation of Issues in $REPO..." -ForegroundColor Cyan

# Phase 1: Environment & Pipeline Setup
gh issue create -R $REPO -t "Task 1.1: Environment Setup and robosuite Testing" -b "Install the robosuite framework and MuJoCo physics engine. Verify the installation by running the minimal environment loop, ensuring image tensors are received correctly without errors."
gh issue create -R $REPO -t "Task 1.2: Implement Common Controller Interface" -b "Create the core structure of the Controller class implementing the mandatory reset and act methods according to the project specifications."
gh issue create -R $REPO -t "Task 1.3: Build CNN Encoder for Vision Pipeline" -b "Design and implement a Convolutional Neural Network (CNN) encoder to process the agentview_image (84x84) and extract low-dimensional feature embeddings, since privileged object states are disabled."

# Phase 2: Baselines & Imitation Learning
gh issue create -R $REPO -t "Task 2.1: Develop Simple Baseline Controller" -b "Implement a simple baseline controller (such as a random policy or a heuristic scripted controller) to serve as the minimum performance benchmark for comparisons."
gh issue create -R $REPO -t "Task 2.2: Collect Demonstration Dataset" -b "Record successful trajectories (state-action pairs) using either the scripted controller or manual guidance to build a dataset for imitation learning."
gh issue create -R $REPO -t "Task 2.3: Implement Behavior Cloning (BC)" -b "Train a policy using Supervised Learning / Behavior Cloning that maps the visual features from Task 1.3 to the expert actions recorded in Task 2.2."

# Phase 3: Advanced Intelligent Control (Core RL)
gh issue create -R $REPO -t "Task 3.1: Setup Core RL Algorithm (Continuous Action Space)" -b "Select and configure a reinforcement learning algorithm suitable for continuous action spaces (such as SAC or TD3) and integrate it with the CNN visual encoder."
gh issue create -R $REPO -t "Task 3.2: Design Recovery Logic for Contact-Rich Failures" -b "Develop a mechanism (via reward shaping or a state-machine) that detects slips, misalignments, or poor grasps, triggering the robot to recover and retry."
gh issue create -R $REPO -t "Task 3.3: Apply Domain Randomization for Robustness" -b "Enable and tune environment randomizations (variations in object mass, friction, positions, lighting, and camera angles) during RL training to handle hidden stress tests."

# Phase 4: Evaluation & Deliverables
gh issue create -R $REPO -t "Task 4.1: Create Evaluation and Logging Scripts" -b "Develop scripts to evaluate the final controller over multiple episodes, automatically tracking metrics such as Success Rate, Cumulative Reward, and failure types."
gh issue create -R $REPO -t "Task 4.2: Record and Edit Demo Video" -b "Capture and compile a short demonstration video highlighting representative successful assembly runs as well as interesting failure modes."
gh issue create -R $REPO -t "Task 4.3: Write the Final Technical Report" -b "Draft the final project report document detailing the methodology, algorithms, experimental setups, ablation studies, and failure analysis."

Write-Host "All Issues have been successfully created on GitHub!" -ForegroundColor Green