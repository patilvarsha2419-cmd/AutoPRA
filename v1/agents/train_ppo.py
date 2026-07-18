"""
train_ppo.py

Trains a PPO (Proximal Policy Optimization) agent on the AutoPRA v1
placement environment using Stable Baselines3.

This was the best-performing algorithm in v1: 38.9% HPWL improvement
over the random baseline at 2M steps (best HPWL 719 vs random baseline
1177), outperforming both A2C (37.2%) and DQN (failed to converge).
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from v1.env.placement_env import PlacementEnv


TOTAL_TIMESTEPS = 2_000_000
LOG_DIR = "logs/ppo"
MODEL_SAVE_PATH = "models/ppo_placement"


def make_env():
    env = PlacementEnv()
    env = Monitor(env)
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(f"{MODEL_SAVE_PATH}/final_model")

    print("Training complete. Final HPWL on eval env:")
    obs, _ = eval_env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
    print(eval_env.unwrapped.final_hpwl())


if __name__ == "__main__":
    main()