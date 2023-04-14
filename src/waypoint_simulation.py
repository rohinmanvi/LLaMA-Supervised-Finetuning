import numpy as np
import math

from stable_baselines3 import PPO

from waypoint_driver_env import DriverEnv

np.set_printoptions(suppress=True)

env = DriverEnv()

model = PPO.load("models/PPO_Waypoint/best_model.zip")

def random_policy(observation):
    acceleration = np.random.uniform(low=-10.0, high=6.0)
    steering_rate = np.random.uniform(low=-0.874, high=0.874)
    
    return np.array([acceleration, steering_rate])

average_reward = 0
episodes = 3

for i in range(episodes):

    observation = env.reset()
    done = False

    while not done:
        print(f"Observation: {np.round(observation, 3)}")

        action, _ = model.predict(observation)
        # action = random_policy(observation)
        # action = np.array([2.0, 0.01])

        observation, reward, done, _ = env.step(action)

        average_reward += reward

average_reward /= episodes
print(f"average reward: {average_reward}")