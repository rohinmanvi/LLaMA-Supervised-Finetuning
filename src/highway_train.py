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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import highway_env
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "intersection-v0",
}

run = wandb.init(
    # project="PPO_Highway",
    # project="PPO_Roundabout",
    project="PPO_Intersection",
    config=config,
    sync_tensorboard=True,
)

models_dir = "models/intersection_ppo_2"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if __name__ == "__main__":
    train = True
    if train:
        env = DummyVecEnv([Monitor(gym.make(config["env_name"]))])
        model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"{models_dir}/{run.id}")

        eval_callback = EvalCallback(env, best_model_save_path=models_dir, eval_freq=10000, n_eval_episodes=100, deterministic=True, render=False)
        wandb_callback = WandbCallback(verbose=2)

        print("Training ...")

        model.learn(total_timesteps=config["total_timesteps"], callback=[eval_callback, wandb_callback])

        print("Done Training")

        run.finish()
