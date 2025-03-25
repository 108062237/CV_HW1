import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import build_model
import yaml
from tqdm import tqdm
from data_loader import get_dataloaders

# 設定路徑
CONFIG_PATH = "configs/config.yaml"
MODEL_PATH = "checkpoints/test_1.pth"
VAL_DIR = "../data/val"
BATCH_SIZE = 32

# 讀取 config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
model = build_model(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=config['device']))
model.to(config['device'])
model.eval()

# 定義 transform（需與訓練一致）
val_transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 建立 validation dataset & dataloader
train_loader, val_loader, test_loader = get_dataloaders(config)

# 預測
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating"):
        images, labels = images.to(config['device']), labels.to(config['device'])
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"✅ Validation Accuracy: {accuracy:.2f}%")
