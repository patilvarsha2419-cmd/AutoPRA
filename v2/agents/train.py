"""
AutoPRA v2 — PPO Training Script
=================================
Trains a PPO agent for VLSI cell placement on either:
  - MCNC cm138a benchmark (138 cells, synthetic but realistic netlist)
  - CircuitNet RISC-V benchmark (250 cells, real chip data)

Usage:
    python train.py --benchmark mcnc       --steps 1000000
    python train.py --benchmark circuitnet --steps 500000

PPO Hyperparameters (final working values):
    learning_rate : 1e-4  — lower than default for stable convergence
    n_steps       : 1024  — longer rollouts for better credit assignment
    batch_size    : 256   — larger batch for stable gradient
    n_epochs      : 10    — reuse experience efficiently
    gamma         : 0.995 — high discount for long-horizon placement
    gae_lambda    : 0.95  — standard advantage estimation
    clip_range    : 0.2   — PPO clipping threshold
    ent_coef      : 0.005 — mild exploration bonus
"""

import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from v2.benchmarks.mcnc_cm138a import (
    load_mcnc_cm138a, build_adjacency, get_grid_size
)
from v2.benchmarks.circuitnet_loader import load_circuitnet_benchmark
from v2.env.placement_env import AutoPRAEnv


# ── Callback ──────────────────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """
    Evaluation callback for tracking HPWL and congestion during training.

    Runs deterministic eval episodes at regular intervals and logs:
      - HPWL (Half Perimeter Wire Length) — primary placement metric
      - Congestion overflow — fraction of illegally overlapping slots
    """

    def __init__(self, eval_env, eval_freq=20000, verbose=1):
        """
        Args:
            eval_env  (AutoPRAEnv): environment for evaluation
            eval_freq (int)       : evaluate every N training steps
            verbose   (int)       : 1 = print progress
        """
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.hpwl_log  = []
        self.cong_log  = []
        self.step_log  = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Run one deterministic eval episode
            obs, _ = self.eval_env.reset()
            done   = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.eval_env.step(int(action))

            hpwl = self.eval_env.final_hpwl()
            cong = self.eval_env.final_congestion()

            self.hpwl_log.append(hpwl)
            self.cong_log.append(cong)
            self.step_log.append(self.n_calls)

            if self.verbose:
                print(f"  Step {self.n_calls:>7,} | "
                      f"HPWL: {hpwl:.1f} | "
                      f"Congestion: {cong*100:.2f}%")

        return True


# ── Benchmark loading ─────────────────────────────────────────────────────────

def load_mcnc_benchmark():
    """
    Load MCNC cm138a benchmark and build all required data structures.

    Returns:
        dict: benchmark data for AutoPRAEnv
    """
    from collections import defaultdict

    benchmark = load_mcnc_cm138a(seed=42)
    nets      = benchmark['nets']
    num_cells = benchmark['num_cells']
    cell_w    = benchmark['cell_w']
    cell_h    = benchmark['cell_h']
    grid_w, grid_h = get_grid_size(benchmark)

    adjacency = build_adjacency(nets, num_cells)

    cell_to_nets = defaultdict(list)
    for net_idx, net in enumerate(nets):
        for c in net:
            cell_to_nets[c].append(net_idx)

    return {
        'num_cells'   : num_cells,
        'grid_w'      : grid_w,
        'grid_h'      : grid_h,
        'nets'        : nets,
        'cell_w'      : cell_w,
        'cell_h'      : cell_h,
        'adjacency'   : adjacency,
        'cell_to_nets': cell_to_nets,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(benchmark_name='mcnc', total_steps=1_000_000,
          eval_freq=20000, save_path=None):
    """
    Train PPO agent on specified benchmark.

    Args:
        benchmark_name (str) : 'mcnc' or 'circuitnet'
        total_steps    (int) : total training timesteps
        eval_freq      (int) : evaluation frequency
        save_path      (str) : path to save trained model

    Returns:
        model    (PPO)            : trained model
        callback (MetricsCallback): training logs
    """
    # Load benchmark
    print(f"\n📦 Loading benchmark: {benchmark_name.upper()}")
    if benchmark_name == 'mcnc':
        bm = load_mcnc_benchmark()
    elif benchmark_name == 'circuitnet':
        bm = load_circuitnet_benchmark()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. "
                         f"Choose 'mcnc' or 'circuitnet'.")

    print(f"   Cells : {bm['num_cells']} | "
          f"Nets: {len(bm['nets'])} | "
          f"Grid: {bm['grid_w']}x{bm['grid_h']}")

    # Random baseline
    rand_hpwls = []
    for _ in range(5):
        env_tmp = AutoPRAEnv(
            bm['nets'], bm['adjacency'], bm['cell_to_nets'],
            bm['cell_w'], bm['cell_h'],
            bm['num_cells'], bm['grid_w'], bm['grid_h']
        )
        obs, _ = env_tmp.reset()
        done   = False
        while not done:
            obs, _, done, _, _ = env_tmp.step(
                env_tmp.action_space.sample()
            )
        rand_hpwls.append(env_tmp.final_hpwl())
    rand_hpwl = np.mean(rand_hpwls)
    print(f"   Random baseline HPWL: {rand_hpwl:.1f}")

    if benchmark_name == 'circuitnet' and 'real_hpwl' in bm:
        print(f"   Real EDA HPWL (gold): {bm['real_hpwl']:.1f}")

    # Build environments
    def make_env():
        return AutoPRAEnv(
            bm['nets'], bm['adjacency'], bm['cell_to_nets'],
            bm['cell_w'], bm['cell_h'],
            bm['num_cells'], bm['grid_w'], bm['grid_h']
        )

    vec_env  = make_vec_env(make_env, n_envs=4)
    eval_env = make_env()

    # PPO model — final working hyperparameters
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate = 1e-4,
        n_steps       = 1024,
        batch_size    = 256,
        n_epochs      = 10,
        gamma         = 0.995,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.005,
        verbose       = 0,
    )

    callback = MetricsCallback(eval_env, eval_freq=eval_freq, verbose=1)

    print(f"\n🚀 Training PPO — {total_steps:,} steps")
    print(f"   4 parallel envs | eval every {eval_freq:,} steps\n")

    start = time.time()
    model.learn(total_timesteps=total_steps, callback=callback)
    elapsed = time.time() - start

    print(f"\n✅ Training done in {elapsed/60:.1f} minutes")

    # Final evaluation (5 runs for stability)
    hpwl_runs, cong_runs = [], []
    for _ in range(5):
        obs, _ = eval_env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = eval_env.step(int(action))
        hpwl_runs.append(eval_env.final_hpwl())
        cong_runs.append(eval_env.final_congestion())

    final_hpwl = np.mean(hpwl_runs)
    final_cong = np.mean(cong_runs)

    print(f"\n📊 Final Results:")
    print(f"{'─'*55}")
    print(f"  {'Method':<20} {'HPWL':>10} {'Improvement':>12} {'Congestion':>10}")
    print(f"{'─'*55}")
    print(f"  {'Random':<20} {rand_hpwl:>10.1f} {'0.0%':>12} {'0.0%':>10}")

    if benchmark_name == 'circuitnet' and 'real_hpwl' in bm:
        rh = bm['real_hpwl']
        print(f"  {'Real EDA (gold)':<20} {rh:>10.1f} {'—':>12} {'0.0%':>10}")

    improv = (rand_hpwl - final_hpwl) / rand_hpwl * 100
    print(f"  {'AutoPRA PPO':<20} {final_hpwl:>10.1f} {improv:>11.1f}% {final_cong*100:>9.2f}%")
    print(f"{'─'*55}")

    # Save model
    if save_path is None:
        save_path = f"autopra_v2_{benchmark_name}_ppo"
    model.save(save_path)
    print(f"\n💾 Model saved: {save_path}.zip")

    return model, callback, {
        'rand_hpwl' : rand_hpwl,
        'final_hpwl': final_hpwl,
        'final_cong': final_cong,
        'improvement': improv,
        'step_log'  : callback.step_log,
        'hpwl_log'  : callback.hpwl_log,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AutoPRA v2 PPO placement agent"
    )
    parser.add_argument(
        '--benchmark', type=str, default='mcnc',
        choices=['mcnc', 'circuitnet'],
        help='Benchmark to train on (default: mcnc)'
    )
    parser.add_argument(
        '--steps', type=int, default=1_000_000,
        help='Total training steps (default: 1000000)'
    )
    parser.add_argument(
        '--eval_freq', type=int, default=20000,
        help='Evaluation frequency in steps (default: 20000)'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Path to save model (default: autopra_v2_<benchmark>_ppo)'
    )
    args = parser.parse_args()

    train(
        benchmark_name=args.benchmark,
        total_steps=args.steps,
        eval_freq=args.eval_freq,
        save_path=args.save,
    )