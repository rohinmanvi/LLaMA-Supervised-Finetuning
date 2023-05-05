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

n_cpu = 6
batch_size = 64
env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
model = PPO("MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ppo/")

config = {"total_timesteps": int(2e4)}
run = wandb.init(project="highway_ppo", config=config, sync_tensorboard=True)
models_dir = "models/highway_ppo"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

eval_callback = EvalCallback(env, best_model_save_path=models_dir, eval_freq=4000, n_eval_episodes=100, deterministic=True, render=False)
wandb_callback = WandbCallback(verbose=2)

model.learn(total_timesteps=config["total_timesteps"], callback=[eval_callback, wandb_callback])

run.finish()
