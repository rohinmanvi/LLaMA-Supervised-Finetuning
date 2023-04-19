import numpy as np
import math
import random
import json

from stable_baselines3 import PPO

from driver_env import DriverEnv
from prompting import get_waypoint_sequence

np.set_printoptions(suppress=True)

env = DriverEnv()

model = PPO.load("models/PPO/best_model")

episodes = 10000

data = []

for i in range(episodes):

    observation = env.reset()
    done = False

    print("Reset")

    sequence = ""

    step = 0

    observation_for_prompt = observation.copy()

    x = 0
    y = 0
    theta = 0

    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, _ = env.step(action)

        step += 1

        if step % 10 == 0:
            ego_state = env.state[0]
            x_prime = ego_state.x
            y_prime = ego_state.y

            x_diff = x_prime - x
            y_diff = y_prime - y

            angle = np.arctan2(y_diff, x_diff) - theta
            angle = env._clamp_angle(angle)

            distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

            action = (distance, angle)

            sequence += get_waypoint_sequence(observation_for_prompt, action) + "\n"

            observation_for_prompt = observation.copy()

            x = x_prime
            y = y_prime
            theta = ego_state.theta

    sequence = sequence.strip()
    
    data.append({"text": sequence})

random.shuffle(data)

with open("data/waypoint_sequence_data.jsonl", "w") as f:
    for datum in data:
        json.dump(datum, f)
        f.write("\n")