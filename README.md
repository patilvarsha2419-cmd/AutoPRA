# \# AutoPRA

# 

# \*\*A Reinforcement Learning Framework for Placement and Routing Automation in VLSI\*\*

# 

# AutoPRA applies reinforcement learning (DQN, A2C, PPO) to the VLSI cell placement and routing problem, benchmarking against traditional methods like Simulated Annealing and real EDA tool outputs.

# 

# \## Overview

# 

# This repo contains two versions:

# 

# \- \*\*v1\*\* — Single-agent RL on a 100-cell synthetic netlist (12x12 grid). Includes DQN/A2C/PPO training, A\* routing, and comparison against Simulated Annealing.

# \- \*\*v2\*\* — Extended framework on real benchmarks: MCNC `cm138a` and CircuitNet RISC-V (RISCY-a-1-c2, 250-cell subset, 28x28 grid). Adds congestion-aware rewards and signal-integrity evaluation.

# 

# \## Results Summary

# 

# | Version | Best Method | HPWL Improvement |

# |---------|-------------|-------------------|

# | v1      | PPO         | 38.9% (vs random baseline) |

# | v2 (MCNC)      | PPO  | 39.7% |

# | v2 (CircuitNet) | PPO | up to 62.1% |

# 

# v1 also includes A\* routing with 100% success on 147 wire connections at 4.7% overhead.

# 

# \## Repository Structure

