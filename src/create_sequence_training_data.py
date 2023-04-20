import numpy as np
import math
import random
import json

from stable_baselines3 import PPO

from driver_env import DriverEnv
from prompting import get_waypoint_sequence_prompt, get_waypoint_completion

np.set_printoptions(suppress=True)

env = DriverEnv()

model = PPO.load("models/PPO/best_model")

def swap_left_right(s):
    temp_string = s.replace("left", "TEMP_PLACEHOLDER")
    temp_string = temp_string.replace("right", "left")
    final_string = temp_string.replace("TEMP_PLACEHOLDER", "right")
    return final_string

def get_action(env, x_prime, y_prime, x, y, theta):
    x_diff = x_prime - x
    y_diff = y_prime - y

    angle = np.arctan2(y_diff, x_diff) - theta
    angle = env._clamp_angle(angle)

    distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

    return distance, angle

episodes = 100

data = []
individual_data = []

for i in range(episodes):

    observation = env.reset()
    done = False

    print("Reset")

    sequence = ""

    step = 0

    prompt = get_waypoint_sequence_prompt(observation)
    # hacky fix
    prompt = swap_left_right(prompt)

    completion = ""

    x = 0
    y = 0
    theta = 0

    prev_x = 0
    prev_y = 0
    prev_theta = 0

    while not done:
        action, _ = model.predict(observation)
        observation, reward, done, _ = env.step(action)

        step += 1

        if step % 10 == 0:
            ego_state = env.state[0]
            x_prime = ego_state.x
            y_prime = ego_state.y

            if step > 10:
                action = get_action(env, x_prime, y_prime, prev_x, prev_y, prev_theta)

                individual_data[-1]["completion"] += get_waypoint_completion(action)
                
                sequence += get_waypoint_completion(action) + "\n"

            action = get_action(env, x_prime, y_prime, x, y, theta)

            completion = get_waypoint_completion(action)

            individual_data.append({"prompt": prompt, "completion": completion})

            sequence += prompt + completion

            prompt = get_waypoint_sequence_prompt(observation)

            prev_x = x
            prev_y = y
            prev_theta = theta

            x = x_prime
            y = y_prime
            theta = ego_state.theta

    sequence = sequence.strip()
    
    data.append({"text": sequence})

random.shuffle(data)
random.shuffle(individual_data)

with open("data/new_waypoint_sequence_data.jsonl", "w") as f:
    for datum in data:
        json.dump(datum, f)
        f.write("\n")

with open("data/new_waypoint_data.jsonl", "w") as f:
    for datum in individual_data:
        json.dump(datum, f)
        f.write("\n")