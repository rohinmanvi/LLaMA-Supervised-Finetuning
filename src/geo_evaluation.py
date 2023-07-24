import numpy as np
import json

from transformers import GenerationConfig

from model_handler import ModelHandler

examples = []

limit = 5

with open('data/data_large_cheap_test.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        examples.append(data)

        limit -= 1

        if limit == 0:
            break

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

generation_config = GenerationConfig(
    max_new_tokens=4,
    do_sample=False
)

for example in examples:

    prompt = example['prompt']
    completion = example['completion']

    response = model_handler.generate_text(
        peft_model='models/asset-index-address-places',
        text=prompt,
        generation_config=generation_config
    )

    response = response[len(prompt):]

    print("============================================================================")
    print(prompt + response)
    print(f"Answer: {completion}")
    print("============================================================================")

predictions = predictions_output.predictions.squeeze()
labels = test_df['target'].to_numpy()

mse = mean_squared_error(labels, predictions)
mae = mean_absolute_error(labels, predictions)
R2 = r2_score(labels, predictions)
r2 = np.corrcoef(labels, predictions)[0, 1] ** 2

print(f'r^2: {r2}')
print(f'R^2: {R2}')
print(f'MSE: {mse}')