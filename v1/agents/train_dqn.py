"""
train_dqn.py

Trains a DQN (Deep Q-Network) agent on the AutoPRA v1 placement
environment using Stable Baselines3.

Note (from experiments): DQN struggled to converge on this task within
2M steps — it stayed in an "Exploring" phase the whole run and never
learned a strong placement policy. Kept in the repo as a baseline
comparison against A2C and PPO.
"""

import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from v1.env.placement_env import PlacementEnv


TOTAL_TIMESTEPS = 2_000_000
LOG_DIR = "logs/dqn"
MODEL_SAVE_PATH = "models/dqn_placement"


def make_env():
    env = PlacementEnv()
    env = Monitor(env)
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
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