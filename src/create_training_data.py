import numpy as np
import math
import random
import json

from stable_baselines3 import PPO

from driver_env import DriverEnv
from prompting import get_short_prompt, get_waypoint_completion

np.set_printoptions(suppress=True)

env = DriverEnv()

model = PPO.load("models/PPO/best_model")

episodes = 1000

data = []

for i in range(episodes):

    observation = env.reset()
    done = False

    print("Reset")

    step = 0

    prompt = get_short_prompt(observation)

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

            completion = get_waypoint_completion(action)

            print("============================================================================")
            print(prompt + completion)
            print("============================================================================")

            data.append({"prompt": prompt, "completion": completion})

            prompt = get_short_prompt(observation)

            x = x_prime
            y = y_prime
            theta = ego_state.theta

random.shuffle(data)

with open("data/waypoint_data.jsonl", "w") as f:
    for datum in data:
        json.dump(datum, f)
        f.write("\n")