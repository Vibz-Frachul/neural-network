# MLP untuk MNIST dengan PyTorch
# Klasifikasi digit tulisan tangan pake multi-layer perceptron.
# Dataset: MNIST — 70.000 gambar grayscale 28×28 (0-9).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Arsitektur: 784 -> 256(ReLU) -> 128(ReLU) -> 10(LogSoftmax)
# Layer yang lebih dalam dengan node lebih sedikit di akhir bikin network belajar fitur
# yang makin abstrak. 256 neuron di layer pertama cukup buat nangkep pola dasar MNIST.

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 784)  # flatten 28×28 -> 784
        return self.net(x)

def train(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
    acc = 100.0 * correct / len(loader.dataset)
    print(f"epoch {epoch:2d} | loss: {total_loss/len(loader):.4f} | acc: {acc:.2f}%")

if __name__ == "__main__":
    # download + transform dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=1000, shuffle=False)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    print("training MLP on MNIST...")
    for epoch in range(1, 6):
        train(model, train_loader, optimizer, criterion, epoch)

    # evaluasi final
    model.eval()
    correct = sum((model(images).argmax(dim=1) == labels).sum().item()
                  for images, labels in test_loader)
    print(f"\ntest accuracy: {100.0 * correct / len(test_loader.dataset):.2f}%")
