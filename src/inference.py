import numpy as np

from transformers import GenerationConfig

from prompting import get_waypoint_prompt, extract_action
from model_handler import ModelHandler

prompt = """Observation: "Our vehicle is going 7.0 m/s with a steering angle of 0.0°. The other vehicle is 44.4 m away and is 12.9° to the right. It is going 7.0 m/s with a direction of 14.6° to the right."
Action: """

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

generation_config = GenerationConfig(
    max_new_tokens=20,
    do_sample=False,
)

response = model_handler.generate_text(
    peft_model='models/finetune-llama-waypoint-driver',
    text=prompt,
    generation_config=generation_config
)

response = response[len(prompt):]

print("============================================================================")
print(prompt + "|" + response)
print("============================================================================")
