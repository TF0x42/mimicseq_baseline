'''
multilayer perceptron to predict the next event
'''
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from data_handler import MedicalDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
import time


def loading_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    percent = round(progress * 100, 2)
    if progress ==1:
        print(f'[{arrow + spaces}] {percent}% Complete', end='\n')
    else:
        print(f'[{arrow + spaces}] {percent}% Complete', end='\r')


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        if num_layers==3:
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.relu2 = nn.ReLU()
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        if self.num_layers==3:
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.bn2(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x
        
def train_model(args):
    start = time.time()
    train_dataset = MedicalDataset(split_type='1day', version='train', num_labels=args.clustering, include_intensities=args.include_intensities, skip=args.skip)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=128)
    label_mapping = {
        'event_id': 87899,
        'c10': 10,
        'c100': 100,  
        'c1000': 1000,
        'c10000': 10000,
    }
    output_size = label_mapping.get(args.clustering, 0)
    model = MLP(87899, args.hidden_layer_size, output_size, args.num_layers).float()
    model = model.to(args.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Starting Training.")
    for epoch in range(args.num_epochs):
        now = time.time()
        model.train()
        total_loss = 0.0
        k=1
        total_items=len(train_loader)
        for inputs, targets in train_loader:
            loading_bar(k, total_items)
            k+=1           
            inputs = inputs.float().to(args.device)
            targets = targets.float().to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        end = time.time()
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] Training Loss: {average_loss:.4f}, elapsed time: {round((end-now)/60, 2)} minutes")
    print(f"Model: output size={output_size}, model_name={args.model_name}")
    print(f"Training finished! Elapsed time: {round((end-start)/60, 2)} minutes")
    torch.save(model.state_dict(), args.model_name)


def test_model(args):
    label_mapping = {
        'event_id': 87899,
        'c10': 10,
        'c100': 100,  
        'c1000': 1000,
        'c10000': 10000,
    }
    output_size = label_mapping.get(args.clustering, 0)
    model = MLP(87899, args.hidden_layer_size, output_size, args.num_layers)
    model.load_state_dict(torch.load(args.model_name))
    model.to(args.device)
    test_dataset = MedicalDataset(split_type='1day', version='test', num_labels=args.clustering, include_intensities=args.include_intensities, skip=args.skip)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=128)
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
            inputs = inputs.float().to(args.device)
            targets = targets.float().to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(targets.squeeze().cpu().numpy())
            threshold = 0.5
            predicted_labels = (outputs > threshold).float()
            accuracy = (predicted_labels == targets).float().mean()
            predicted_labels = predicted_labels.cpu().numpy()
            true_labels = targets.cpu().numpy()
            predicted_labels = predicted_labels.reshape(-1)
            true_labels = true_labels.reshape(-1)
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            print(f"batch={k}")
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
        print(f"Model: output size={output_size}, model_name={args.model_name}")



