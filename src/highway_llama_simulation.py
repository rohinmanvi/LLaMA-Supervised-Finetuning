import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory

from transformers import GenerationConfig
from model_handler import ModelHandler
import time  # Import time module

np.set_printoptions(suppress=True)

env = gym.make("highway-fast-v0", render_mode='rgb_array')

def record_videos(env, video_folder="highway_videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

env = record_videos(env)

model_handler = ModelHandler("decapoda-research/llama-7b-hf")
generation_config = GenerationConfig(max_new_tokens=1, do_sample=False)

total_rewards = []  # List to store total rewards per episode
episode_lengths = []  # List to store episode lengths

for episode in range(25):
    np.random.seed(episode)  # optional, for numpy random number generator
    env.seed(episode)  # set the seed for the environment

    obs, info = env.reset()
    done = truncated = False

    prompt_so_far = ""
    inference_times = []  # List to store inference times

    total_reward = 0  # Reset total reward for the new episode

    while not (done or truncated):
        start_time = time.time()  # Start timer

        prompt_so_far += f"Observation:\n{np.round(obs, 3)}\nAction: "

        response = model_handler.generate_text(
            peft_model='models/highway-driver-final-2',
            text=prompt_so_far,
            generation_config=generation_config
        )

        response = response[len(prompt_so_far):]
        action = int(response.strip())

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward  # Add the reward to the total reward

        prompt_so_far += response + "\n"

        end_time = time.time()  # End timer
        inference_time = end_time - start_time  # Calculate inference time
        inference_times.append(inference_time)  # Store inference time

    total_rewards.append(total_reward)  # Store total reward for this episode
    episode_lengths.append(len(inference_times))  # Store episode length

    print(f"============================================================================")
    print(f"Episode {episode + 1}")
    print(prompt_so_far)
    print("Inference times:")
    for i, t in enumerate(inference_times):
        print(f"Step {i + 1}: {t:.6f} seconds")
    print(f"Total inference time: {sum(inference_times):.6f} seconds")
    print(f"Total reward: {total_reward}")
    print(f"Episode length: {len(inference_times)} steps")
    print(f"============================================================================")

average_reward = np.mean(total_rewards)
average_length = np.mean(episode_lengths)

print(f"Average reward per episode: {average_reward}")
print(f"Average episode length: {average_length} steps")

env.close()
