# -*- coding: utf-8 -*-
# @Time    : 2024/12/17 22:30
# @Author  : thu-huangzc
# @Email   : riccardohhhhzz@gmail.com
# @File    : train_phone_detector.py
# @Software: Vscode
# @Brief   : 使用公开的打电话图像分类数据集对预训练的EfficientNet-b4进行训练
# @Command : CUDA_VISIBLE_DEVICES=0 python train_phone_detector.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.efficientnet import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.model_selection import train_test_split
from utils.general import ResizeWithPadding

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = "datasets/smoke_phone"

# Parameters
batch_size = 64
epochs = 10
learning_rate = 0.001
weight_decay = 1e-4
image_size = 224

# 使用自定义的 ResizeWithPadding 作为 transforms 的一部分
transform = transforms.Compose([
    ResizeWithPadding(target_size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Dataset Loader for specific tasks
class BinaryDataset(Dataset):
    def __init__(self, data_dir, class1, class2, transform=None):
        self.class1_dir = os.path.join(data_dir, class1)
        self.class2_dir = os.path.join(data_dir, class2)
        self.class1_label = 0
        self.class2_label = 1
        self.transform = transform

        self.data = []

        for img in os.listdir(self.class1_dir):
            self.data.append((os.path.join(self.class1_dir, img), self.class1_label))

        for img in os.listdir(self.class2_dir):
            self.data.append((os.path.join(self.class2_dir, img), self.class2_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load datasets
def load_datasets(task):
    if task == "smoke":
        dataset = BinaryDataset(data_dir, "normal", "smoke", transform)
    elif task == "phone":
        dataset = BinaryDataset(data_dir, "normal", "phone", transform)
    else:
        raise ValueError("Unsupported task")

    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size, shuffle=False),
    )

# Build model
def build_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.classifier[1].in_features, 2),
    )
    return model

# Training loop
def train_model(task):
    print(f"Training model for task: {task}")

    train_loader, val_loader = load_datasets(task)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Save model
    torch.save(model.state_dict(), f"{task}_detection.pth")
    print(f"Model for task {task} saved.")

if __name__ == "__main__":
    from PIL import Image

    # train_model("smoke")
    train_model("phone")
