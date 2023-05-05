import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# import wandb
# from wandb.integration.sb3 import WandbCallback

import highway_env

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "HighwayEnv",
}

# run = wandb.init(
#     # project="PPO_Waypoint",
#     # project="PPO",
#     project="HighwayEnv",
#     config=config,
#     sync_tensorboard=True,
# )

models_dir = "models/PPO_Highway"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def make_env():
    env = gym.make('highway-fast-v0')
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

model = PPO(config["policy_type"], env, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path=models_dir, eval_freq=10000, n_eval_episodes=100, deterministic=True, render=False)

# wandb_callback = WandbCallback(verbose=2)

print("Training ...")

model.learn(total_timesteps=config["total_timesteps"], callback=[eval_callback])

print("Done Training")

run.finish()

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
import highway_env


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
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
        # Train the agent
        model.learn(total_timesteps=int(2e4))
        # Save the agent
        model.save("highway_ppo/model")

    model = PPO.load("highway_ppo/model")
    env = gym.make("highway-fast-v0")
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
