import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory
import json
from gym import Wrapper

class RewardNormalizationWrapper(Wrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        return obs, reward, done, info

np.set_printoptions(suppress=True)

env = gym.make('intersection-v0', render_mode='rgb_array')

min_reward = -1  # Set the minimum possible reward for your environment
max_reward = 1  # Set the maximum possible reward for your environment
env = RewardNormalizationWrapper(env, min_reward, max_reward)

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

with open("data/intersection_planner_sequence_data_incremental.jsonl", "a") as f:
    for i in range(10000):
        print(i)

        obs, info = env.reset()
        done = truncated = False

        sequence = ""

        while not (done or truncated):
            observation_string = f"Observation:\n{str(np.round(obs, 3))}"
            sequence += observation_string + "\n"

            action = agent.act(obs)

            action_string = f"Action: {str(action)}"
            sequence += action_string + "\n"

            obs, reward, done, truncated, info = env.step(action)

        # Write the data to the file
        json.dump({"text": sequence}, f)
        f.write("\n")
        f.flush()

env.close()
