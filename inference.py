import numpy as np
import math
import re
import time

from driver_env import DriverEnv
from model_handler import ModelHandler

np.set_printoptions(suppress=True)

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
model_handler = ModelHandler("decapoda-research/llama-7b-hf")

start_time = time.time()

average_reward = 0
episodes = 1

for i in range(episodes):

    observation = env.reset()
    done = False

    while not done:
        # print(f"Observation: {np.round(observation, 3)}")

        prompt = get_short_prompt(observation)

        generation_config = transformers.GenerationConfig(
            max_new_tokens=32,
            do_sample=False
        )

        response = model_handler.generate_text(
            peft_model='llama-driver3',
            text=prompt,
            generation_config=generation_config
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