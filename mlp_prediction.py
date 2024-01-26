'''
multilayer perceptron to predict the next event
'''
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from data_handler import MedicalDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time






use_multiple_gpu=False
input_size = 87899
hidden_size = 1000
output_size = 10000
batch_size = 512
learning_rate = 0.001
epochs = 3
num_samples=100000
filename = 'c10000-5k_layer-first_second_day-entire_dataset.sav'
device = torch.device('cpu')   # cuda:0
include_intensities=False
three_layer = True











def loading_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    percent = round(progress * 100, 2)
    if progress ==1:
        print(f'[{arrow + spaces}] {percent}% Complete', end='\n')
    else:
        print(f'[{arrow + spaces}] {percent}% Complete', end='\r')


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.fc1(x) 
#         x = self.relu(x)
#         x = self.bn1(x)
#         x = self.fc2(x)
#         x = self.sig(x)
#         return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        if three_layer:
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu2 = nn.ReLU()
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        if three_layer:
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.bn2(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x
        


def train_model():
    start = time.time()
    train_dataset = MedicalDataset(split_type='1day', version='train', num_labels='c10000', num_samples=num_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=128)
    model = MLP(input_size, hidden_size, output_size).float()
    if use_multiple_gpu:
        model = model.to(device)
        #model = nn.DataParallel(model, device_ids=[0,1,2,3]) #, output_device=0)
    #criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting Training.")
    for epoch in range(epochs):
        now = time.time()
        model.train()
        total_loss = 0.0
        k=1
        total_items=len(train_loader)
        for inputs, targets in train_loader:
            loading_bar(k, total_items)
            k+=1           
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        end = time.time()
        print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {average_loss:.4f}, elapsed time: {round((end-now)/60, 2)} minutes")
    print(f"Model: output size={output_size}, filename={filename}")
    print(f"Training finished! Elapsed time: {round((end-start)/60, 2)} minutes")
    torch.save(model.state_dict(), filename)


def test_model():
    model = MLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename))
    model.to(device)
    test_dataset = MedicalDataset(split_type='1day', version='test', num_labels='c10000', num_samples=num_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=128)
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_targets = []
        num=0
        acc=0
        prec=0
        rec=0
        f_one=0
        k=0
        for inputs, targets in test_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            outputs = model(inputs)
            #print(inputs)
            # print(outputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(targets.squeeze().cpu().numpy())
            threshold = 0.5
            predicted_labels = (outputs > threshold).float()
            accuracy = (predicted_labels == targets).float().mean()
            # print(predicted_labels)
            # print(targets)
            predicted_labels = predicted_labels.cpu().numpy()
            true_labels = targets.cpu().numpy()
            # print("vorher:")
            # print(true_labels.shape)
            predicted_labels = predicted_labels.reshape(-1)
            true_labels = true_labels.reshape(-1)
            # print("nachher:")
            # print(true_labels.shape)
            # print(predicted_labels)
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            print(f"batch={k}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            k+=1
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
        print(f"Model: output size={output_size}, filename={filename}")


train_model()
test_model()