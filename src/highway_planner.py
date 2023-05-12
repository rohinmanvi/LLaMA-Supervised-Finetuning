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

def record_videos(env, video_folder="highway_opd_videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

env = record_videos(env)

total_rewards = []  # List to store total rewards per episode
episode_lengths = []  # List to store episode lengths
truncated_episodes = 0  # Counter for truncated episodes

for episode in range(100):
    obs, info = env.reset()
    done = truncated = False
    total_reward = 0  # Reset total reward for the new episode
    steps = 0  # Reset steps counter for the new episode

    while not (done or truncated):
        print(f"Observation:\n{np.round(obs, 3)}")
        action = agent.act(obs)
        print(f"Action: {action}")

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward  # Add the reward to the total reward
        steps += 1  # Increment steps counter

    total_rewards.append(total_reward)  # Store total reward for this episode
    episode_lengths.append(steps)  # Store episode length

max_episode_length = max(episode_lengths)  # Get the maximum episode length

for length in episode_lengths:
    if length < max_episode_length:
        truncated_episodes += 1  # Increment counter if episode was less than max length

print(f"Total reward: {total_reward}")
print(f"Episode length: {steps} steps")

average_reward = np.mean(total_rewards)
average_length = np.mean(episode_lengths)
collision_rate = truncated_episodes / len(total_rewards)

print(f"Average reward per episode: {average_reward}")
print(f"Average episode length: {average_length} steps")
print(f"Collision rate: {collision_rate}")

env.close()
