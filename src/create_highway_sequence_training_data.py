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
import json

np.set_printoptions(suppress=True)

model = PPO.load("roundabout_ppo/model")
env = gym.make("roundabout-v0")

# Open the file in append mode
with open("data/roundabout_sequence_data_incremental.jsonl", "a") as f:
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

        # Write the data to the file
        json.dump({"text": sequence}, f)
        f.write("\n")
        f.flush()
