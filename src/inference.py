import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
from model.model import build_model  
import yaml
from torchvision import transforms

CONFIG_PATH = "configs/config.yaml"
MODEL_PATH = "checkpoints/test_1.pth"
OUTPUT_CSV = "prediction.csv"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=config['device']))
model.eval()
model.to(config['device'])

def run_inference(test_loader):
    predictions = []

    for images, filenames in tqdm(test_loader, desc="Predicting"):
        images = images.to(config['device'])
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        for filename, pred in zip(filenames, preds):
            img_id = os.path.splitext(os.path.basename(filename))[0]  # 去掉副檔名
            predictions.append({"image_name": img_id, "pred_label": pred.item()})

    df = pd.DataFrame(predictions)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\u2705 預測完成，結果已儲存在 {OUTPUT_CSV}")

transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """對單張圖片進行預測，返回類別"""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(config['device'])

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    
    return pred.item()

def show_prediction(image_path):
    """顯示單張圖片並預測類別"""
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(config['device'])

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    
    class_label = pred.item()
    
    # 顯示圖片
    plt.imshow(image)
    plt.title(f"Predicted Class: {class_label}")
    plt.axis("off")
    plt.show()

    print(f"🖼 圖片: {image_path}, 預測類別: {class_label}")

def predict_val_folder(val_dir):
    """遍歷 val 資料夾中的圖片，依序預測並印出結果"""
    for class_name in sorted(os.listdir(val_dir)):  # 遍歷每個子資料夾
        class_path = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 跳過非資料夾的內容
        
        print(f"📂 資料夾: {class_name}")

        for img_name in sorted(os.listdir(class_path)):  # 遍歷該類別資料夾內的圖片
            img_path = os.path.join(class_path, img_name)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):  # 確保是圖片檔案
                pred_class = predict_image(img_path)
                print(f"  🖼 {img_name} -> 預測類別: {pred_class}")

if __name__ == "__main__":
    # from data_loader import get_dataloaders
    # _, _, test_loader = get_dataloaders(config)
    # run_inference(test_loader)

    # 測試單張圖片預測
    # image_path = "../data/val/8/57fb8db2-ac5d-4989-a43d-cef84f3aea4e.jpg"  # 替換成你的圖片路徑
    # show_prediction(image_path)
    predict_val_folder("../data/val")
