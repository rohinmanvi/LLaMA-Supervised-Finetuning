from transformers import BertTokenizer, BertForSequenceClassification
import torch

output_dir = 'models/bert_driver'

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(output_dir)  # use the directory where the model was saved
tokenizer = BertTokenizer.from_pretrained(output_dir)  # use the directory where the tokenizer was saved

# Assuming 'observation' is your new data
observation = "Observation:\n[[ 1. 0.781 0. 0.312 0. ]\n [ 1. 0.109 0.333 -0.045 0. ]\n [ 1. 0.243 0. -0.013 0. ]\n [ 1. 0.377 0. -0.019 0. ]\n [ 1. 0.503 0.333 -0.027 0. ]]"

# Preprocess the observation
inputs = tokenizer(observation, padding=True, truncation=True, return_tensors='pt')

# Predict action
model.eval()  # Set the model to eval mode
with torch.no_grad():  # No need to calculate gradients for inference, so wrap with no_grad to save memory
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_action = torch.argmax(logits, dim=-1)  # Get the most likely action

print(f'The predicted action is {predicted_action.item()}.')
