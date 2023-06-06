import numpy as np
import gym
import highway_env
import json
import re
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import time
import pandas as pd

def load_data(file):
    observations = []
    actions = []
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            obs_text = data['text'].split('Action:')[0].strip()
            obs_text = re.findall(r'\[.*?\]', obs_text)
            obs = [list(map(float, re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line))) for line in obs_text]
            obs = np.array(obs).flatten().tolist()
            action = int(data['text'].split('Action:')[1].strip())
            observations.append(obs)
            actions.append(action)
    return observations, actions

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out

# Load and preprocess the data
observations, actions = load_data('data/highway_planner_data_incremental.jsonl')

scaler = StandardScaler()
observations = scaler.fit_transform(observations)

INPUT_SIZE = len(observations[0])
HIDDEN_SIZE = 64
NUM_CLASSES = 5

# Load the trained MLP model
model = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
model.load_state_dict(torch.load("models/mlp_model.pth"))
model.eval()

env = gym.make("highway-fast-v0", render_mode='rgb_array')

total_rewards = []
episode_lengths = []
min_inference_times = []
max_inference_times = []
avg_inference_times = []
truncated_episodes = 0

for episode in range(100):
    obs, info = env.reset()
    done = truncated = False
    inference_times = []
    total_reward = 0

    while not done and not truncated:
        obs_text = re.findall(r'\[.*?\]', str(obs))
        obs_values = [list(map(float, re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line))) for line in obs_text]
        obs_values = np.array(obs_values).flatten().tolist()
        obs_values = scaler.transform([obs_values])  # Scale the observation values

        start_time = time.time()
        action = torch.argmax(model(torch.tensor(obs_values, dtype=torch.float32))).item()
        end_time = time.time()

        obs, reward, done, info = env.step(action)
        total_reward += reward

        inference_time = end_time - start_time
        inference_times.append(inference_time)

    total_rewards.append(total_reward)
    episode_lengths.append(len(inference_times))

    min_inference_times.append(min(inference_times))
    max_inference_times.append(max(inference_times))
    avg_inference_times.append(sum(inference_times) / len(inference_times))

for i, t in enumerate(inference_times):
    print(f"Step {i + 1}: {t:.6f} seconds")
print(f"Total inference time: {sum(inference_times):.6f} seconds")
print(f"Total reward: {total_reward}")
print(f"Episode length: {len(inference_times)} steps")
print(f"============================================================================")

max_episode_length = max(episode_lengths)

for length in episode_lengths:
    if length < max_episode_length:
        truncated_episodes += 1

average_reward = np.mean(total_rewards)
average_length = np.mean(episode_lengths)
collision_rate = truncated_episodes / len(total_rewards)
average_inference_time = np.mean(avg_inference_times)

print(f"Average reward per episode: {average_reward}")
print(f"Average episode length: {average_length} steps")
print(f"Collision rate: {collision_rate}")
print(f"Average inference time: {average_inference_time} seconds")

data = pd.DataFrame({
    "total_rewards": total_rewards,
    "episode_lengths": episode_lengths,
    "min_inference_times": min_inference_times,
    "max_inference_times": max_inference_times,
    "avg_inference_times": avg_inference_times
})
data.to_csv('mlp_highway_data.csv', index=False)

env.close()
