import numpy as np
import json

from transformers import GenerationConfig

from prompting import get_waypoint_prompt, extract_action
from model_handler import ModelHandler

examples = []

limit = 3

with open('data/new_waypoint_sequence_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        examples.append(data['text'])

        limit -= 1

        if limit == 0:
            break

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

generation_config = GenerationConfig(
    max_new_tokens=93,
    do_sample=False,
)

def find_nth_occurrence_and_get_substring_before(main_string, substring, n):
    count = 0
    index = 0

    while count < n:
        index = main_string.find(substring, index)
        if index == -1:
            return None
        count += 1
        index += len(substring)

    return main_string[:index]

for example in examples:

    sequence = example

    print("============================================================================")
    print(f"\nAnswer:\n{sequence}")
    print("============================================================================")

    for i in range(1, 10, 4):
        prompt = find_nth_occurrence_and_get_substring_before(sequence, "Action:", i)

        response = model_handler.generate_text(
            peft_model='models/sequence-driver',
            text=prompt,
            generation_config=generation_config
        )

        response = response[len(prompt):]

        print("============================================================================")
        print(prompt + "|" + response)
        print("============================================================================")