'''
multilayer perceptron to predict the next event
'''


import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).type(torch.float64)
        self.relu = nn.ReLU().type(torch.float64)
        self.fc2 = nn.Linear(hidden_size, output_size).type(torch.float64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    



import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from data_handler import MedicalDataset

# Hyperparameters
input_size = 88000 # Specify the input size
hidden_size = 40000
output_size = 88000 # Specify the output size
batch_size = 5137
learning_rate = 0.001
epochs = 10

# Create the dataset and data loader
train_dataset = MedicalDataset('train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = MedicalDataset('val')
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Create the model
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(total_loss)

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {average_loss:.4f}")

    # Validation
    # model.eval()
    # with torch.no_grad():
    #     valid_predictions = []
    #     valid_targets = []

    #     for inputs, targets in valid_loader:
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         valid_predictions.extend(predicted.cpu().numpy())
    #         valid_targets.extend(targets.squeeze().cpu().numpy())

    #     accuracy = accuracy_score(valid_targets, valid_predictions)
    #     print(f"Epoch [{epoch + 1}/{epochs}] Validation Accuracy: {accuracy:.4f}")

print("Training finished!")

# Evaluation
test_dataset = MedicalDataset('test')
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
with torch.no_grad():
    test_predictions = []
    test_targets = []

    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_targets.extend(targets.squeeze().cpu().numpy())

    test_accuracy = accuracy_score(test_targets, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")