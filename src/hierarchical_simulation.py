import numpy as np
import math

from stable_baselines3 import PPO

from transformers import GenerationConfig

from driver_env import DriverEnv
from prompting import get_waypoint_prompt, extract_action
from model_handler import ModelHandler

np.set_printoptions(suppress=True)

env = DriverEnv()

model = PPO.load("models/PPO_Waypoint/best_model")

# model_handler = ModelHandler("decapoda-research/llama-7b-hf")

average_reward = 0
episodes = 1

for i in range(episodes):

    observation = env.reset()
    done = False

    print("Reset")
    print(f"Observation: {np.round(observation, 3)}")

    steps = 0

    prev_acceleration = 0.0
    prev_steering_rate = 0.0

    waypoint_x = 0.0
    waypoint_y = 0.0

    while not done:

        ego_state = env.state[0]
        x = ego_state.x
        y = ego_state.y
        theta = ego_state.theta

        print((x, y, theta))

        if steps % 10 == 0:
            # prompt = get_waypoint_prompt(observation)

            # generation_config = GenerationConfig(
            #     max_new_tokens=32,
            #     do_sample=False
            # )

            # response = model_handler.generate_text(
            #     peft_model='models/llama-waypoint-driver2',
            #     text=prompt,
            #     generation_config=generation_config
            # )

            # response = response[len(prompt):]

            # print("============================================================================")
            # print(prompt + response)
            # print("============================================================================")

            # distance, angle = extract_action(response)

            # true_angle = angle + theta

            # delta_x = distance * np.cos(true_angle)
            # delta_y = distance * np.sin(true_angle)

            # waypoint_x = x + delta_x
            # waypoint_y = y + delta_y

            waypoint_x += 10.0
            waypoint_y += 0.0

            print((x, y))
            print((waypoint_x, waypoint_y))

        velocity, steering, _, _, _, _ = observation

        x_diff = waypoint_x - x
        y_diff = waypoint_y - y

        angle = np.arctan2(y_diff, x_diff) - theta
        angle = env._clamp_angle(angle)

        distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

        waypoint_time = (steps % 10) * 0.1

        waypoint_observation = np.array([velocity, steering, prev_acceleration, prev_steering_rate, angle, distance, waypoint_time])

        print(f"Waypoint Observation: {np.round(waypoint_observation, 3)}")

        action, _ = model.predict(waypoint_observation)
        observation, reward, done, _ = env.step(action)
        average_reward += reward

        prev_acceleration, prev_steering_rate = action

        steps += 1

        print(f"Observation: {np.round(observation, 3)}")

average_reward /= episodes
print(f"average reward: {average_reward}")