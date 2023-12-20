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
hidden_size = 1000
output_size = 10 # Specify the output size
batch_size = 100
learning_rate = 0.001
epochs = 10

# Create the dataset and data loader
train_dataset = MedicalDataset('train', num_labels='c10')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# valid_dataset = MedicalDataset('val', num_labels='c10')
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Create the model
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    k=0
    for inputs, targets in train_loader:
        print(f"batch={k}")
        k+=1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {average_loss:.4f}")
print("Training finished!")

# Evaluation
test_dataset = MedicalDataset('test', num_labels='c10')
test_loader = DataLoader(test_dataset, batch_size=batch_size)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model.eval()
with torch.no_grad():
    test_predictions = []
    test_targets = []
    num=0
    acc=0
    prec=0
    rec=0
    f_one=0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        #print(inputs)
        #print(outputs)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().numpy())
        #print(test_predictions.extend(predicted.cpu().numpy()))
        test_targets.extend(targets.squeeze().cpu().numpy())
        #print(test_targets.extend(targets.squeeze().cpu().numpy()))
        # Define the threshold
        threshold = 0.5  # You can adjust this threshold as needed

        # Apply the threshold to your predicted probabilities
        predicted_labels = (outputs > threshold).float()

        # Calculate the accuracy
        accuracy = (predicted_labels == targets).float().mean()
        print(predicted_labels)
        print(targets)

        predicted_labels = predicted_labels.cpu().numpy()
        true_labels = targets.cpu().numpy()

        # Reshape the arrays if needed (e.g., flatten them)
        predicted_labels = predicted_labels.reshape(-1)
        true_labels = true_labels.reshape(-1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        num+=1
        acc+=accuracy
        prec+=precision
        rec+=recall
        f_one+=f1
    print("-------------------------------------------")
    print(f"Total Accuracy: {acc/num:.4f}")
    print(f"Total Precision: {prec/num:.4f}")
    print(f"Total Recall: {rec/num}")
    print(f"Total F1-score: {f_one/num}")