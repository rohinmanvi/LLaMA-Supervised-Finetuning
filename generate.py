from model_handler import ModelHandler

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

prompt = "\"Our vehicle is going 5.8 m/s with a steering angle of 7.9° to the left. The other vehicle is 14.9 m away and is 1.0° to the left. It is going 7.0 m/s with a direction of 34.4° to the left.\" ->"

response = model_handler.generate_text(
    peft_model='llama-driver2',
    text=prompt,
    temperature=0.1,
    top_p=0.75,
    top_k=50,
    max_new_tokens=32
)

print(response)

# import numpy as np
# import math
# import re

# from driver_env import DriverEnv

# np.set_printoptions(suppress=True)

# env = DriverEnv()

# average_reward = 0
# episodes = 1

# for i in range(episodes):

#     observation = env.reset()
#     done = False

#     while not done:
#         # print(f"Observation: {np.round(observation, 3)}")

#         def get_shortest_prompt(observation):
#             ego_velocity, steering, angle, distance, direction, agent_velocity = observation
#             return f"[{ego_velocity:.1f}, {np.rad2deg(steering):.1f}, {np.rad2deg(angle):.1f}, {distance:.1f}, {np.rad2deg(direction):.1f}, {agent_velocity:.1f}] ->"

#         prompt = get_shortest_prompt(observation)

#         response = model_handler.generate_text(
#             peft_model='llama-driver',
#             text=prompt,
#             temperature=0.1,
#             top_p=0.75,
#             top_k=50,
#             max_new_tokens=32
#         )

#         response = response[len(prompt):]

#         print("============================================================================")
#         print(prompt + response)
#         print("============================================================================")

#         result = re.findall(r"[-+]?\d*\.\d+|\d+", response)
#         acceleration = float(result[0])
#         steering_rate = np.deg2rad(float(result[2]))
#         action = (acceleration, steering_rate)

#         observation, reward, done, _ = env.step(action)

#         average_reward += reward

# average_reward /= episodes
# print(f"average reward: {average_reward}")