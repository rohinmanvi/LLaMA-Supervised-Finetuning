import os
import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import highway_env
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 20000,
    "env_name": "highway-fast-v0",
}

run = wandb.init(
    project="PPO_Highway",
    config=config,
    sync_tensorboard=True,
)

models_dir = "models/highway_ppo"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env(config["env_name"], n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(config["policy_type"],
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log=f"models/highway_ppo/{run.id}")

        eval_callback = EvalCallback(env, best_model_save_path=models_dir, eval_freq=4000, n_eval_episodes=100, deterministic=True, render=False)
        wandb_callback = WandbCallback(verbose=2)

        print("Training ...")

        model.learn(total_timesteps=config["total_timesteps"], callback=[eval_callback, wandb_callback])

        print("Done Training")

        run.finish()
