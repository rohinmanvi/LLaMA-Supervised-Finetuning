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

data = []

for i in range(10000):
    print(i)

    obs, info = env.reset()
    done = truncated = False

    sequence = ""

    while not (done or truncated):
        observation_string = f"Observation:\n{str(np.round(obs, 3))}"
        sequence += observation_string + "\n"

        action, _ = model.predict(obs)
        
        action_string = f"Action: {str(action)}"
        sequence += action_string + "\n"

        obs, reward, done, truncated, info = env.step(action)

    data.append({"text": sequence})

with open("data/highway_sequence_data.jsonl", "w") as f:
    for datum in data:
        json.dump(datum, f)
        f.write("\n")
