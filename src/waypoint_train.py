import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

from waypoint_driver_env import DriverEnv

wandb.init(project="PPO_Waypoint")

models_dir = "PPO_Waypoint"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = DriverEnv()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

eval_callback = EvalCallback(env, best_model_save_path=models_dir, eval_freq=10000,
                             deterministic=True, render=False)

wandb_callback = WandbCallback()

print("Training ...")

model.learn(total_timesteps=10000000, tb_log_name=models_dir, callback=[eval_callback, wandb_callback])

print("Done Training")