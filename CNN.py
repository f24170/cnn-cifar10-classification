import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用設備：", device)
if device.type == 'cuda':
    print("GPU 名稱：", torch.cuda.get_device_name(0))
    print("CUDA 可用：", torch.cuda.is_available())
else:
    print("未使用 GPU，改用 CPU 執行。")

# ================== 1. 資料預處理與載入 ==================
# 轉換 CIFAR-10 圖像為 Tensor，並進行標準化
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 標準化至 [-1, 1]
])

# 下載 CIFAR-10 數據集
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CIFAR-10 類別名稱
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# ================== 2. 建立 CNN 模型 ==================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.batch1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch3(self.conv3(x))))
        x = self.pool(torch.relu(self.batch4(self.conv4(x))))
        x = self.pool(torch.relu(self.batch5(self.conv5(x))))
        x = x.view(-1, 512 * 1 * 1)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.fc2(x)
        return x


# ================== 3. 訓練與測試函數 ==================
# 初始化 CNN 模型
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


# 訓練函數
def train(model, train_loader, criterion, optimizer, scheduler, epochs=6):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step(running_loss)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")


# 測試函數
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# ================== 4. 執行訓練與測試 ==================
train(model, train_loader, criterion, optimizer, scheduler, epochs=6)
test(model, test_loader)

def visualize_predictions(model, test_loader):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = (img + 1) / 2
        ax.imshow(img)
        ax.set_title(f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}")
        ax.axis("off")
    plt.show()

visualize_predictions(model, test_loader)