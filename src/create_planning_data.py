import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory
import json

np.set_printoptions(suppress=True)

env = gym.make("roundabout-v0", render_mode='rgb_array')

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

with open("data/roundabout_planner_sequence_data_incremental.jsonl", "a") as f:
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
