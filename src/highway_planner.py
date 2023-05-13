import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory
import time
import pandas as pd

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

total_rewards = []
episode_lengths = []
min_inference_times = []
max_inference_times = []
avg_inference_times = []
truncated_episodes = 0

for episode in range(100):
    obs, info = env.reset()
    done = truncated = False
    total_reward = 0
    steps = 0

    inference_times = []

    while not (done or truncated):
        print(f"Observation:\n{np.round(obs, 3)}")
        
        start_time = time.time()
        action = agent.act(obs)
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        print(f"Action: {action}")

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward  
        steps += 1

    total_rewards.append(total_reward)
    episode_lengths.append(steps)
    
    min_inference_times.append(min(inference_times))
    max_inference_times.append(max(inference_times))
    avg_inference_times.append(sum(inference_times) / len(inference_times))

    print(f"Total reward: {total_reward}")
    print(f"Episode length: {steps} steps")
    print(f"Average episode inference time: {np.mean(inference_times)} seconds") 

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
data.to_csv('planner_highway_data.csv', index=False)

env.close()
