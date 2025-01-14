import numpy as np

from transformers import GenerationConfig

from prompting import get_waypoint_prompt, extract_action
from model_handler import ModelHandler

# prompt = """Observation: "Our vehicle is going 7.0 m/s with a steering angle of 0.0°. The other vehicle is 44.4 m away and is 12.9° to the right. It is going 7.0 m/s with a direction of 14.6° to the right."
# Action: """

# prompt = """Observation: "Our vehicle is going 7.0 m/s with a steering angle of 0.0°. The other vehicle is 42.5 m away and is 39.0° to the left. It is going 7.0 m/s with a direction of 35.8° to the left."
# Action: """

# Observation: "Our vehicle is going 7.0 m/s with a steering angle of 0.0°. The other vehicle is 42.5 m away and is 39.0° to the left. It is going 7.0 m/s with a direction of 35.8° to the left."
# Action:  (9.5 m, -16.9°)

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

generation_config = GenerationConfig(
    max_new_tokens=20,
    do_sample=False,
)

prompt = """Observation: "Our vehicle is going 7.0 m/s with a steering angle of 0.0°. The other vehicle is 44.4 m away and is 12.9° to the right. It is going 7.0 m/s with a direction of 14.6° to the right."
Action: """

response = model_handler.generate_text(
    peft_model='models/finetune-llama-waypoint-driver',
    text=prompt,
    generation_config=generation_config
)

response = response[len(prompt):]

print("============================================================================")
print(prompt + "|" + response)
print("============================================================================")

prompt = "Observation: \"Our vehicle is going 7.0 m/s with a steering angle of 0.0\u00b0. The other vehicle is 44.4 m away and is 12.9\u00b0 to the right. It is going 7.0 m/s with a direction of 14.6\u00b0 to the right.\"\nAction: "

response = model_handler.generate_text(
    peft_model='models/finetune-llama-waypoint-driver',
    text=prompt,
    generation_config=generation_config
)

response = response[len(prompt):]

print("============================================================================")
print(prompt + "|" + response)
print("============================================================================")

prompt = """Observation: "Our vehicle is going 7.0 m/s with a steering angle of 0.0°. The other vehicle is 42.5 m away and is 39.0° to the left. It is going 7.0 m/s with a direction of 35.8° to the left."
Action: """

response = model_handler.generate_text(
    peft_model='models/finetune-llama-waypoint-driver',
    text=prompt,
    generation_config=generation_config
)

response = response[len(prompt):]

print("============================================================================")
print(prompt + "|" + response)
print("============================================================================")

prompt = "Observation: \"Our vehicle is going 7.0 m/s with a steering angle of 0.0\u00b0. The other vehicle is 42.5 m away and is 35.6\u00b0 to the left. It is going 7.0 m/s with a direction of 38.5\u00b0 to the left.\"\nAction: "

response = model_handler.generate_text(
    peft_model='models/finetune-llama-waypoint-driver',
    text=prompt,
    generation_config=generation_config
)

response = response[len(prompt):]

print("============================================================================")
print(prompt + "|" + response)
print("============================================================================")
