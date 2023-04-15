import math
import time

from transformers import GenerationConfig

from driver_env import DriverEnv
from prompting import get_short_prompt, extract_action
from model_handler import ModelHandler

env = DriverEnv()

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

start_time = time.time()

average_reward = 0
episodes = 1

for i in range(episodes):

    observation = env.reset()
    done = False

    while not done:

        prompt = get_short_prompt(observation)

        generation_config = GenerationConfig(
            max_new_tokens=32,
            do_sample=False
        )

        response = model_handler.generate_text(
            peft_model='models/low-level-llama',
            text=prompt,
            generation_config=generation_config
        )

        response = response[len(prompt):]

        print("============================================================================")
        print(prompt + response)
        print("============================================================================")

        action = extract_action(response)

        observation, reward, done, _ = env.step(action)

        average_reward += reward

average_reward /= episodes
print(f"average reward: {average_reward}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")