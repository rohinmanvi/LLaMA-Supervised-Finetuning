import json
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

class HighwayPlannerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(file):
    observations = []
    actions = []
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            observations.append(data['text'].split('Action:')[0].strip())
            actions.append(int(data['text'].split('Action:')[1].strip()))
    return observations, actions

# Load the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('xlm-roberta-base')
model = RobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=5)  # Assuming 5 different actions

# Load and preprocess the data
observations, actions = load_data('data/highway_planner_data_incremental.jsonl')

# Split the data into training and testing datasets
train_observations, test_observations, train_labels, test_labels = train_test_split(observations, actions, test_size=0.1, stratify=actions)

# Tokenize and convert to datasets
train_inputs = tokenizer(train_observations, padding=True, truncation=True, return_tensors='pt')
test_inputs = tokenizer(test_observations, padding=True, truncation=True, return_tensors='pt')

train_dataset = HighwayPlannerDataset(train_inputs, train_labels)
eval_dataset = HighwayPlannerDataset(test_inputs, test_labels)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='models/roberta_driver',   # output directory
    num_train_epochs=4,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_steps=10,
    report_to='wandb'
)

# Define the trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

# Train the model
trainer.train()

trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
