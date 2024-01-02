'''
multilayer perceptron to predict the next event
'''
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from sklearn.metrics import accuracy_score
from data_handler import MedicalDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from torch.nn.parallel import DistributedDataParallel as DDP

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)#.type(torch.float64)
        self.relu = nn.ReLU()#.type(torch.float64)
        self.bn1 = nn.BatchNorm1d(hidden_size)#.type(torch.float64)
        self.fc2 = nn.Linear(hidden_size, output_size)#.type(torch.float64)
        self.sig = nn.Sigmoid()#.type(torch.float64)

    def forward(self, x):
        print("hi x")
        print(x)
        print("working 1")
        x = self.fc1(x)
        print("working 2")
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.sig(x)
        print("working end")
        return x
    
use_multiple_gpu=True
input_size = 88000
hidden_size = 400
output_size = 10
batch_size = 160
learning_rate = 0.001
epochs = 3
num_samples=1000
filename = 'test_model.sav'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12344'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size):
    setup(rank, world_size)

    model = MLP(input_size, hidden_size, output_size).to(rank)
    model = DDP(model, device_ids=[rank])
    # if use_multiple_gpu:
    #     model = model.to('cuda')
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[0,1], output_device=0)

    train_dataset = MedicalDataset(split_type='1day', version='train', num_labels='c10', num_samples=num_samples)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, sampler=train_sampler)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        k=0
        for inputs, targets in train_loader:
            inputs = inputs.float().to(rank)
            targets = targets.float().to(rank)
            print(f"batch={k}")
            print(inputs)
            k+=1
            optimizer.zero_grad()
            outputs = model(inputs)
            print("working 3")
            loss = criterion(outputs, targets)
            print("working")
            loss.backward()
            print("working")
            optimizer.step()
            print("working")
            total_loss += loss.item()
            print("working")
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {average_loss:.4f}")
    print("Training finished!")
    torch.save(model.state_dict(), filename)
    cleanup()


def test_model():
    model = MLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename))
    model.to('cuda:0')
    test_dataset = MedicalDataset(split_type='1day', version='test', num_labels='c10', num_samples=num_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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
            inputs = inputs.float().to('cuda')
            targets = targets.float().to('cuda')
            outputs = model(inputs)
            #print(inputs)
            print(outputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(targets.squeeze().cpu().numpy())
            threshold = 0.5
            predicted_labels = (outputs > threshold).float()
            accuracy = (predicted_labels == targets).float().mean()
            print(predicted_labels)
            print(targets)
            predicted_labels = predicted_labels.cpu().numpy()
            true_labels = targets.cpu().numpy()
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


# train_model()
# test_model()
        
def main():
    world_size = torch.cuda.device_count()
    print(world_size)
    torch.multiprocessing.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()