# Project 2: Vision-Based Contact-Rich Robot Manipulation

This repository contains the complete implementation for Project 2 of the **Intelligent Control** course (Academic Year 2025-2026).
It is implemented from team 2:

| Name                  | Id    |
|-----------------------|-------|
|Mpantekas Nikolaos     |1092562|
|Papoutsopoulos Nikolaos|1101000|
|Stathopoulos Stavros   |1101069|

---

## Objective (The Task)

The goal is to control a 7-DOF **Franka Emika Panda** robotic arm equipped with a **PandaGripper** in a contact-rich assembly environment (**NutAssembly**). The robot must successfully execute the following sequential policy:

1. **Perceive & Grasp:** Locate and securely grasp a nut.
2. **Transport & Align:** Transfer the nut above the corresponding peg and align it with high precision.
3. **Insertion:** Successfully insert the nut onto the peg.

---

## Primary Challenges

### Vision-Based Control

In typical simulation environments, policies enjoy "privileged information" (the exact $x, y, z$ coordinates of the object extracted directly from the physics engine). For this project, **privileged state access is strictly prohibited** (`use_object_obs=False`). Your policy must map raw image tensors (e.g., `agentview_image` at an $84 \times 84$ resolution) directly to actions. The agent must infer geometry and spatial relationships purely from visual features.

### Contact-Rich Dynamics

The moment the nut interfaces with the peg, high friction and contact forces dominate the system dynamics. A micro-scale positional error (on the millimeter level) will cause jam states or wedging. The controller must actively manage these contact forces and, crucially, demonstrate **error recovery** (e.g., adapting to finger slippage, initial alignment failures, or unstable grasps).

---

## Simulation Environment

The project is built on **robosuite**, a simulation framework powered by the **MuJoCo** physics engine. You are provided with a baseline Operational Space Controller (OSC) for Cartesian end-effector control. Your objective is to replace the random action sampling interface (`env.action_spec[0].sample()`) with an intelligent, closed-loop control policy.

---

## Evaluation Criteria (Stress Tests & Robustness)

Evaluation will not occur under static, nominal conditions. Solutions will be subjected to unannounced **stress tests featuring randomized initial conditions**:

* Randomized initial position and orientation (yaw) of the nut.
* Perturbations in peg geometry and placement.
* Variations in physical properties (friction coefficients, object mass).
* Visual domain shifts (lighting variations, camera angle perturbations).

Performance metrics are strictly bound to **Success Rate**, **Cumulative Reward**, and **Robustness/Recovery Capabilities** under stress.

---

## Permissible Methodologies

The architecture design is completely open-ended. Viable paradigms include:

* **Reinforcement Learning (RL):** Model-free algorithms (e.g., PPO, SAC, TD3) coupled with Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs) for visual encoding.
* **Imitation Learning / Behavior Cloning:** Training policies on expert demonstrations.
* **Hybrid Architectures:** Combining classical control theory (e.g., Model Predictive Control / Operational Space Control) for free-space trajectory tracking, and switching to RL/Impedance control for the contact-rich insertion phase.

---

## Deliverables

Each team must submit:

1. **Controller Module:** A production-ready controller implementing the required `Controller` interface.
2. **Source Code:** Fully reproducible training and evaluation pipelines.
3. **Model Artifacts:** Trained weights, network checkpoints, or learned parameter configurations.
4. **Demonstration Video:** A short video showcasing nominal success cases along with critical failure modes and recoveries.
5. **Technical Report:** A comprehensive engineering report detailing methodology, experimental results, ablation studies, and failure case analyses.
