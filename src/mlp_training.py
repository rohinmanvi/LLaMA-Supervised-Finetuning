import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out

def load_data(file):
    observations = []
    actions = []
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            obs_text = data['text'].split('Action:')[0].strip()
            obs_text = re.findall(r'\[.*?\]', obs_text)
            obs = [list(map(float, re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line))) for line in obs_text]
            obs = np.array(obs).flatten().tolist()
            action = int(data['text'].split('Action:')[1].strip())
            observations.append(obs)
            actions.append(action)
    return observations, actions

# Load and preprocess the data
observations, actions = load_data('data/highway_planner_data_incremental.jsonl')

scaler = StandardScaler()
observations = scaler.fit_transform(observations)

INPUT_SIZE = len(observations[0])
HIDDEN_SIZE = 64
NUM_CLASSES = 5
NUM_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.01

# Initialize the model, loss function, and optimizer
model = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Split the data into training and testing datasets
train_observations, test_observations, train_labels, test_labels = train_test_split(observations, actions, test_size=0.1, stratify=actions)

# Create Tensor datasets
train_data = TensorDataset(torch.tensor(train_observations, dtype=torch.float32), torch.tensor(train_labels))
test_data = TensorDataset(torch.tensor(test_observations, dtype=torch.float32), torch.tensor(test_labels))

# Create dataloaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (observations, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(observations)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
    scheduler.step()

# Save the model
torch.save(model.state_dict(), 'models/mlp_model.pth')

# Set the model to evaluation mode
model.eval()

# Initialize a list to store the model's predictions
predictions = []

# Iterate over the test data and generate predictions
with torch.no_grad():
    correct = 0
    total = 0
    for observations, labels in test_loader:
        outputs = model(observations)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate the accuracy of the model on the test data
    accuracy = correct / total

print(f'Test Accuracy of the model on the test observations: {accuracy * 100}%')
