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

np.set_printoptions(suppress=True)

model = PPO.load("highway_ppo/model")
env = gym.make("highway-fast-v0")
for _ in range(1):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs)
        str(action)
        print(f"Action: {str(action)}")
        obs, reward, done, truncated, info = env.step(action)
        print(f"Observation:\n{str(np.round(obs, 3))}")
