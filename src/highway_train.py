import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

import highway_env

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "HighwayEnv",
}

run = wandb.init(
    # project="PPO_Waypoint",
    # project="PPO",
    project="HighwayEnv",
    config=config,
    sync_tensorboard=True,
)

models_dir = "models/PPO_Highway"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make("highway-v0")

model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

eval_callback = EvalCallback(env, best_model_save_path=models_dir, eval_freq=10000, n_eval_episodes=100, deterministic=True, render=False)

wandb_callback = WandbCallback(verbose=2)

print("Training ...")

model.learn(total_timesteps=config["total_timesteps"], callback=[eval_callback, wandb_callback])

print("Done Training")

run.finish()
