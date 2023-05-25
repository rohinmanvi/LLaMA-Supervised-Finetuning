import numpy as np
from stable_baselines3 import PPO
from transformers import GenerationConfig

from driver_env import DriverEnv
from prompting import get_waypoint_sequence_shortest_prompt, extract_two_action
from model_handler import ModelHandler

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

pastelBlue = "#0072B2"
pastelRed = "#F5615C"
pastelGreen = "#009E73"
pastelPurple = "#8770FE"

env = DriverEnv()

model = PPO.load("models/PPO/best_model")

observation = env.reset(2)

done = False

while not done:
    action, _ = model.predict(observation)
    observation, reward, done, _ = env.step(action)

expert_ego_positions = env.ego_positions
agent_positions = env.agent_positions

env.close()

expert_x, expert_y = zip(*expert_ego_positions)
agent_x, agent_y = zip(*agent_positions)

# Plot agent vehicle positions with smaller dots and no connecting line
plt.scatter(expert_x, expert_y, s=0.25, c=pastelGreen, marker='o', label='Ego Vehicle')
# Plot agent vehicle positions with smaller dots and no connecting line
plt.scatter(agent_x, agent_y, s=0.25, c=pastelRed, marker='o', label='Agent Vehicle')
plt.xlabel('X-axis (meters)')
plt.ylabel('Y-axis (meters)')

# Add larger dots every 10 steps (1 second)
for i in range(0, len(expert_ego_positions), 10):
    plt.scatter(expert_ego_positions[i][0], expert_ego_positions[i][1], s=10, c=pastelGreen, marker='o')
    plt.scatter(agent_positions[i][0], agent_positions[i][1], s=10, c=pastelRed, marker='o')

# Calculate the necessary x and y limits to achieve a 3:2 aspect ratio
min_x, max_x = plt.xlim()
min_y, max_y = plt.ylim()
width = max_x - min_x
height = max_y - min_y
desired_aspect_ratio = 3 / 2

if width / height > desired_aspect_ratio:
    new_height = width / desired_aspect_ratio
    plt.ylim(min_y - (new_height - height) / 2, max_y + (new_height - height) / 2)
else:
    new_width = height * desired_aspect_ratio
    plt.xlim(min_x - (new_width - width) / 2, max_x + (new_width - width) / 2)

# Set the aspect ratio to be equal for both axes
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('positions_plot.png', dpi=300)  # Increase the resolution by setting the DPI
plt.close()