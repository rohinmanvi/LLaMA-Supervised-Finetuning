import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
import time  # Import time module
import pandas as pd  # Import pandas module

np.set_printoptions(suppress=True)

env = gym.make("highway-fast-v0", render_mode='rgb_array')

def record_videos(env, video_folder="highway_roberta_videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

env = record_videos(env)

model_dir = "models/roberta_driver"

# Load the trained RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=5)  # Assuming 5 different actions
model.eval()

# Use the pipeline API for easier inference
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=False)

total_rewards = []
episode_lengths = []
min_inference_times = []
max_inference_times = []
avg_inference_times = []
truncated_episodes = 0

for episode in range(10):
    obs, info = env.reset()
    done = truncated = False

    prompt_so_far = ""
    inference_times = []

    total_reward = 0

    while not (done or truncated):
        prompt = f"Observation:\n{np.round(obs, 5)}"

        prompt_so_far += prompt

        start_time = time.time()

        # Generate an action using the trained BERT model
        response = classifier(prompt)[0]
        action = int(response['label'].split('_')[1])

        end_time = time.time()

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        prompt_so_far += f"\nAction: {action}\n"

        inference_time = end_time - start_time
        inference_times.append(inference_time)

    total_rewards.append(total_reward)
    episode_lengths.append(len(inference_times))

    min_inference_times.append(min(inference_times))
    max_inference_times.append(max(inference_times))
    avg_inference_times.append(sum(inference_times) / len(inference_times))

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
data.to_csv('roberta_highway_data.csv', index=False)

env.close()
