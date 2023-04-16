import numpy as np
import math

from stable_baselines3 import PPO

from waypoint_driver_env import DriverEnv

np.set_printoptions(suppress=True)

env = DriverEnv()

# model = PPO.load("models/PPO_Waypoint/best_model")

average_reward = 0
episodes = 5

for i in range(episodes):

    observation = env.reset()
    done = False

    print("Reset")
    print(f"Observation: {np.round(observation, 3)}")

    while not done:
        # action, _ = model.predict(observation)
        action = np.array([6.0, 0.0])

        observation, reward, done, _ = env.step(action)
        average_reward += reward

        print(f"Observation: {np.round(observation, 3)}")

average_reward /= episodes
print(f"average reward: {average_reward}")