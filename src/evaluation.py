import numpy as np
import json

from transformers import GenerationConfig

from prompting import get_waypoint_prompt, extract_action
from model_handler import ModelHandler

examples = []

with open('data/waypoint_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        examples.append(data)

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

generation_config = GenerationConfig(
    max_new_tokens=32,
    do_sample=False
)

limit = 10

for example in examples:

    prompt = example['prompt']
    completion = example['completion']

    response = model_handler.generate_text(
        peft_model='models/llama-waypoint-driver2',
        text=prompt,
        generation_config=generation_config
    )

    response = response[len(prompt):]

    print("============================================================================")
    print(prompt + response)
    print(f"Answer: {completion}")
    print("============================================================================")

    limit -= 1

    if limit == 0:
        break