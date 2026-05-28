# AutoPRA
RL-based autonomous VLSI placement &amp; routing agent | PPO 38.9% HPWL improvement | 100% routing success
# AutoPRA — Autonomous VLSI Placement & Routing Agent

A multi-agent Reinforcement Learning framework for automated VLSI cell 
placement and routing.

## Results
- PPO achieved **38.9% HPWL improvement** on 100-cell layout environment
- A* routing agent achieved **100% routing success** across 147 wire connections 
  with ~39% wire length reduction
- Simulated Annealing baseline achieved **70.8% HPWL improvement**
- Full comparison across DQN, A2C, PPO, and SA algorithms

## Tech Stack
- Python, PyTorch, Stable Baselines3, Gymnasium
- Google Colab T4 GPU
- Custom placement environment built from scratch

## About
Built as part of final year major project at DSATM Bangalore.
Explores RL-based approaches to automate the placement & routing 
stage of VLSI physical design.
