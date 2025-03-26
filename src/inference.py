import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
from model.model import build_model  
import yaml
from torchvision import transforms
from data_loader import get_dataloaders

CONFIG_PATH = "configs/config.yaml"
MODEL_PATH = "checkpoints/resnext101_SE_epoch24.pth"
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


if __name__ == "__main__":
    # from data_loader import get_dataloaders
    _, _, test_loader = get_dataloaders(config)
    run_inference(test_loader)

    
