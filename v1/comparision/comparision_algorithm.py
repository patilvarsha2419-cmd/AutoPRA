"""
compare_algorithms.py

Runs a fair, equal-budget comparison across DQN, A2C, and PPO (loaded
from their saved checkpoints), plus the Simulated Annealing baseline
and a pure random baseline, and prints/saves a summary table.

Note: This script expects trained model checkpoints to already exist
(from running train_dqn.py, train_a2c.py, train_ppo.py). It does NOT
retrain anything — it only evaluates and compares.
"""

import os
import numpy as np
import pandas as pd

from stable_baselines3 import DQN, A2C, PPO

from v1.env.placement_env import PlacementEnv
from v1.baselines.simulated_annealing import SimulatedAnnealingPlacer


MODEL_PATHS = {
    "DQN": "models/dqn_placement/final_model",
    "A2C": "models/a2c_placement/final_model",
    "PPO": "models/ppo_placement/final_model",
}

MODEL_CLASSES = {
    "DQN": DQN,
    "A2C": A2C,
    "PPO": PPO,
}

RESULTS_CSV_PATH = "results/comparison_table.csv"


def evaluate_random_baseline(env, num_episodes=5):
    """Places cells uniformly at random and returns mean HPWL."""
    hpwl_results = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            valid_actions = np.where(env.unwrapped.grid_occupancy == -1)[0]
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
        hpwl_results.append(env.unwrapped.final_hpwl())

    return float(np.mean(hpwl_results))


def evaluate_rl_model(model, env, num_episodes=5):
    """Evaluates a trained RL model and returns mean HPWL."""
    hpwl_results = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        hpwl_results.append(env.unwrapped.final_hpwl())

    return float(np.mean(hpwl_results))


def evaluate_sa_baseline(adjacency, num_runs=3):
    """Runs Simulated Annealing multiple times and returns mean best HPWL."""
    hpwl_results = []

    for run_idx in range(num_runs):
        placer = SimulatedAnnealingPlacer(adjacency=adjacency, seed=run_idx)
        result = placer.run(verbose=False)
        hpwl_results.append(result["final_hpwl"])

    return float(np.mean(hpwl_results))


def main():
    os.makedirs("results", exist_ok=True)

    env = PlacementEnv()

    print("Evaluating random baseline...")
    random_hpwl = evaluate_random_baseline(env)

    print("Evaluating Simulated Annealing baseline...")
    sa_hpwl = evaluate_sa_baseline(env.adjacency)

    results = {
        "Random": random_hpwl,
        "Simulated Annealing": sa_hpwl,
    }

    for name, path in MODEL_PATHS.items():
        model_file = f"{path}.zip"
        if not os.path.exists(model_file):
            print(f"Skipping {name}: no checkpoint found at {model_file}")
            continue

        print(f"Evaluating {name}...")
        model_class = MODEL_CLASSES[name]
        model = model_class.load(path)
        results[name] = evaluate_rl_model(model, env)

    # build comparison table
    rows = []
    baseline_hpwl = results["Random"]
    for name, hpwl in results.items():
        improvement_pct = 100.0 * (baseline_hpwl - hpwl) / baseline_hpwl
        rows.append({
            "Method": name,
            "Mean HPWL": round(hpwl, 1),
            "Improvement vs Random (%)": round(improvement_pct, 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Mean HPWL").reset_index(drop=True)

    print("\n=== Comparison Table ===")
    print(df.to_string(index=False))

    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()