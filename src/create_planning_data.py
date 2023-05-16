import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory
import json
import os

np.set_printoptions(suppress=True)

env = gym.make('highway-fast-v0')

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

# Function to find the next available file number
def get_next_available_file_number(base_filename):
    counter = 0
    while True:
        filename = f"{base_filename}_{counter}.jsonl"
        if not os.path.exists(filename):
            return counter
        counter += 1

# Use the function to get the next available file number
base_filename = "data/highway_planner_data_detailed"
next_file_number = get_next_available_file_number(base_filename)

# Create the full filename with the next available number
filename = f"{base_filename}_{next_file_number}.jsonl"

# Use the new filename in your code
with open(filename, "a") as f:
    for i in range(10000):
        print(i)

        obs, info = env.reset()
        done = truncated = False

        sequence = ""

        while not (done or truncated):
            observation_string = f"Observation:\n{np.round(obs, 3)}"
            sequence += observation_string + "\n"

            action = agent.act(obs)

            action_string = f"Action: {action}"
            sequence += action_string + "\n"

            obs, reward, done, truncated, info = env.step(action)

        # Write the data to the file
        json.dump({"text": sequence}, f)
        f.write("\n")
        f.flush()

env.close()