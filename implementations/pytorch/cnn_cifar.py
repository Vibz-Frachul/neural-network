# CNN untuk CIFAR-10 dengan PyTorch
# Klasifikasi 10 kelas object: pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, truk.
# Dataset: 60.000 gambar RGB 32×32.
# Arsitektur: Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 lebih kompleks dari MNIST — butuh convolution buat nangkep pola spasial.
# Dua layer convolution dengan channel makin dalem makin gede: pola dasar (edge) -> pola kompleks (object parts).

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))       # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))       # 16 -> 8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_loader = DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])),
        batch_size=1000, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"training CNN on CIFAR-10 ({device})...")
    for epoch in range(1, 16):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = sum((model(images.to(device)).argmax(dim=1) == labels.to(device)).sum().item()
                      for images, labels in test_loader)
        acc = 100.0 * correct / len(test_loader.dataset)
        print(f"epoch {epoch:2d} | lr: {scheduler.get_last_lr()[0]:.6f} | test acc: {acc:.2f}%")

    print(f"\nfinal test accuracy: {acc:.2f}%")
