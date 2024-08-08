import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms

# download and load dataset
def data_module(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training_data = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    valid_size = 0.2
    train_size = int((1 - valid_size) * len(training_data))
    val_size = len(training_data) - train_size
    training_data, val_data = data.random_split(
        training_data, [train_size, val_size])

    train_dataloader = data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dataloader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader, test_dataloader

# define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    return val_loss, val_accuracy

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    print('Training complete')
    return model

def main():
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader = data_module(
        batch_size=BATCH_SIZE)

    model = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    trained_model = train(model, train_dataloader,
                          val_dataloader, criterion, optimizer, num_epochs, device)

    torch.save(trained_model.state_dict(), 'model.pth')
    print("Model saved to model.pth")


if __name__ == '__main__':
    main()
