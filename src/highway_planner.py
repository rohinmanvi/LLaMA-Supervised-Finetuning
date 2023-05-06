import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory

np.set_printoptions(suppress=True)

env = gym.make("highway-fast-v0", render_mode='rgb_array')

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

def record_videos(env, video_folder="videos_planning"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

env = record_videos(env)

for _ in range(1):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        print(f"Observation:\n{str(np.round(obs, 3))}")
        action = agent.act(obs)
        print(f"Action: {str(action)}")
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

env.close()
