from torchvision import transforms
from PIL import Image
import os

from torch.utils.data import Dataset, DataLoader
import torch

IMAGE_SIZE = (224, 224)
transform_resize = transforms.Resize(IMAGE_SIZE)

img_path = "./data/train/0/0f0aed51-5899-4336-98cd-03fc0516e2cd.jpg"  
img = Image.open(img_path)
img_resized = transform_resize(img)

print(f"原始尺寸: {img.size}, 調整後尺寸: {img_resized.size}")

train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),   # 統一尺寸
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomRotation(15),   # 隨機旋轉 ±15 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 隨機調整亮度 & 對比
    transforms.ToTensor(),  # 轉換為 PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
])

# 測試數據增強
img_transformed = train_transforms(img)
print(f"轉換後的 Tensor 形狀: {img_transformed.shape}")  # 應該是 [3, 224, 224]


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 讀取所有圖片路徑與標籤
        for class_id in sorted(os.listdir(root_dir)):  # 0~99
            class_path = os.path.join(root_dir, class_id)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(int(class_id))  # 以資料夾名稱作為類別標籤

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # 確保是 RGB 圖像
        if self.transform:
            image = self.transform(image)

        return image, label

# 建立 DataLoader
BATCH_SIZE = 32

train_dataset = CustomDataset(root_dir="./data/train", transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 測試 DataLoader
for images, labels in train_loader:
    print(f"批次圖像大小: {images.shape}")  # 應該是 [32, 3, 224, 224]
    print(f"批次標籤: {labels}")
    break  # 只看第一個批次


