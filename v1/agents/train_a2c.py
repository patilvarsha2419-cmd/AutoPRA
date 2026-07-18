"""
train_a2c.py

Trains an A2C (Advantage Actor-Critic) agent on the AutoPRA v1
placement environment using Stable Baselines3.

Note (from experiments): A2C started learning around ~60K steps and
reached a "Good" placement policy by ~1.12M steps, achieving 37.2%
HPWL improvement at 2M steps (vs 38.9% for PPO). Solid mid-tier
baseline between DQN (failed) and PPO (best).
"""

import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from v1.env.placement_env import PlacementEnv


TOTAL_TIMESTEPS = 2_000_000
LOG_DIR = "logs/a2c"
MODEL_SAVE_PATH = "models/a2c_placement"


def make_env():
    env = PlacementEnv()
    env = Monitor(env)
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
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