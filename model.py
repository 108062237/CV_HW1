import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights

from tqdm import tqdm

import json
from PIL import Image

model_name = "resnet18"

if model_name == "resnet18":
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
elif model_name == "resnet34":
    model = resnet34(weights=ResNet34_Weights.DEFAULT) 

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"model: {model_name}, device: {device}")
print(f"parameters: {sum([param.numel() for param in model.parameters()])}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100
    print(f"訓練損失: {epoch_loss:.4f}, 訓練準確率: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()  
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total * 100
    print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")
    return val_loss, val_acc

EPOCHS = 5

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}:")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

