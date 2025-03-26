from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)  
    
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        self.class_names = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))],
            key=lambda x: int(x)
        )
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(config):
    # Transform for validation & test
    base_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Train transform with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(config['train_dir'], transform=train_transform)
    val_dataset   = CustomDataset(config['val_dir'], transform=base_transform)
    test_dataset  = TestDataset(config['test_dir'], transform=base_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

def show_image_with_label(image_tensor, label, class_name=None):
    import matplotlib.pyplot as plt
    image = F.to_pil_image(image_tensor.cpu())  # 還原成 PIL image
    plt.imshow(image)
    title = f"Label: {label}" if class_name is None else f"Label: {label} (Class: {class_name})"
    plt.title(title)
    plt.axis("off")
    plt.show()

# testing
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders(config)

    print("🔍 檢查 train_loader 資料是否正確")
    dataset = train_loader.dataset
    class_names = dataset.class_names if hasattr(dataset, 'class_names') else None

    for i, (image, label) in enumerate(train_loader):
        for j in range(min(4, image.size(0))):  # 每個 batch 顯示最多4張
            class_name = class_names[label[j]] if class_names else None
            print(f"第 {i+1} 個 batch, 第 {j+1} 張圖 - Label: {label[j].item()}, Class: {class_name}")
            show_image_with_label(image[j], label[j].item(), class_name)

        user = input("按 Enter 顯示下一批，輸入 q 離開：")
        if user.lower() == 'q':
            break

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for images, labels in train_loader:
        print(f"Train batch image shape: {images.shape}, labels: {labels}")
        break

    for images, filenames in test_loader:
        print(f"Test image shape: {images.shape}, filename: {filenames}")
        break
 
