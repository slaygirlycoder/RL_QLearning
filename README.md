# RL_QLearning
**Implementation of the Q learning RL algorithm**
FrozenLake 8x8 – Q‑Learning Agent
This repository contains a Q‑Learning implementation that solves the FrozenLake-v1 environment (8x8 map) using OpenAI Gymnasium.

**Features**
Training mode – Learns an optimal policy via epsilon‑greedy exploration.

Rendering mode – Visualize the agent moving on the frozen lake.

Model persistence – Saves and loads the Q‑table using pickle.

Reward plotting – Displays the moving average of successes per episode.

**How to Run**
bash
python frozen_lake_qlearning.py
Output
frozen_lake8x8.pkl – Saved Q‑table after training.
frozen_lack8x8.png – Plot of training progress (average reward over last 100 episodes).




