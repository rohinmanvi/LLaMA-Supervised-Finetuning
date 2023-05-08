import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from rl_agents.agents.common.factory import agent_factory

from transformers import GenerationConfig
from model_handler import ModelHandler

np.set_printoptions(suppress=True)

env = gym.make("highway-fast-v0", render_mode='rgb_array')

agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

def record_videos(env, video_folder="videos_llama"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

env = record_videos(env)

model_handler = ModelHandler("decapoda-research/llama-7b-hf")
generation_config = GenerationConfig(max_new_tokens=1, do_sample=False)

for _ in range(5):
    obs, info = env.reset()
    done = truncated = False

    prompt_so_far = ""

    while not (done or truncated):

        prompt_so_far += f"Observation:\n{str(np.round(obs, 3))}\nAction: "

        response = model_handler.generate_text(
            peft_model='models/highway-driver-final',
            text=prompt_so_far,
            generation_config=generation_config
        )

        response = response[len(prompt_so_far):]
        action = int(response.strip())

        print("============================================================================")
        print(prompt_so_far + "|" + response)
        print("============================================================================")

        prompt_so_far += response + "\n"

        obs, reward, done, truncated, info = env.step(action)

env.close()
