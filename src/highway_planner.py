import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory

np.set_printoptions(suppress=True)

env = gym.make("roundabout-v0", render_mode='rgb_array')

class RewardClampingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        clamped_reward = np.clip(reward, 0, 1)
        return obs, clamped_reward, done, info

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

def record_videos(env, video_folder="videos_roundabout_expert"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

env = RewardClampingWrapper(env)

# env = record_videos(env)

for _ in range(5):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        print(f"Observation:\n{np.round(obs, 3)}")
        action = agent.act(obs)
        print(f"Action: {action}")
        obs, reward, done, truncated, info = env.step(action)

env.close()
