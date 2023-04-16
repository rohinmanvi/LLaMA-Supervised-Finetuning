import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback

from waypoint_driver_env import DriverEnv

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000000,
    "env_name": "WaypointDriverEnv",
}

run = wandb.init(
    project="PPO_Waypoint",
    config=config,
    sync_tensorboard=True,
)

models_dir = "models/PPO_Waypoint3"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def make_env():
    env = DriverEnv()
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

# model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model = PPO.load("models/PPO_Waypoint2/best_model.zip", env=env, tensorboard_log=f"runs/{run.id}")

eval_callback = EvalCallback(env, best_model_save_path=models_dir, eval_freq=10000, deterministic=True, render=False)

wandb_callback = WandbCallback(verbose=2)

print("Training ...")

model.learn(total_timesteps=config["total_timesteps"], callback=[eval_callback, wandb_callback])

print("Done Training")

run.finish()
