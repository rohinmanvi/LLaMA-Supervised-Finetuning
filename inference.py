from model_handler import ModelHandler

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

# prompt = "\"Our vehicle is going 5.8 m/s with a steering angle of 7.9° to the left. The other vehicle is 14.9 m away and is 1.0° to the left. It is going 7.0 m/s with a direction of 34.4° to the left.\" ->"

# response = model_handler.generate_text(
#     peft_model='llama-driver3',
#     text=prompt,
#     temperature=0.1,
#     top_p=0.75,
#     top_k=50,
#     max_new_tokens=32
# )

# print(response)

import numpy as np
import math
import re

from driver_env import DriverEnv

np.set_printoptions(suppress=True)

import time

start_time = time.time()

def distance_string(distance):
        return f"{distance:.1f} m"


def speed_string(velocity):
    return f"{velocity:.1f} m/s"


def angle_string(angle):
    degrees = abs(np.rad2deg(angle))
    direction = "" if degrees == 0 else f" to the {'left' if angle > 0 else 'right'}"

    return f"{degrees:.1f}°{direction}"


def get_short_prompt(observation):
    ego_velocity, steering, angle, distance, direction, agent_velocity = observation

    return f""""Our vehicle is going {speed_string(ego_velocity)} with a steering angle of {angle_string(steering)}. The other vehicle is {distance_string(distance)} away and is {angle_string(angle)}. It is going {speed_string(agent_velocity)} with a direction of {angle_string(direction)}." ->"""


env = DriverEnv()

average_reward = 0
episodes = 1

for i in range(episodes):

    observation = env.reset()
    done = False

    while not done:
        # print(f"Observation: {np.round(observation, 3)}")

        prompt = get_short_prompt(observation)

        response = model_handler.generate_text(
            peft_model='llama-driver3',
            text=prompt,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.1,
            top_p=0.75,
            top_k=50,
            num_beams=1
        )

        response = response[len(prompt):]

        print("============================================================================")
        print(prompt + response)
        print("============================================================================")

        result = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        acceleration = float(result[0])
        steering_rate = np.deg2rad(float(result[-1]))
        action = (acceleration, steering_rate)

        observation, reward, done, _ = env.step(action)

        average_reward += reward

average_reward /= episodes
print(f"average reward: {average_reward}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")