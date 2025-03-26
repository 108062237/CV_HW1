import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import yaml
from torch.utils.data import Dataset, DataLoader
from model.model import build_model
from data_loader import get_dataloaders

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Config ----------
CONFIG_PATH = "configs/config.yaml"
MODEL_PATHS = [
    "checkpoints/resnet101_epoch55.pth",
    "checkpoints/resnext50_epoch30.pth",
    "checkpoints/resnext101_epoch38.pth",
    "checkpoints/resnext101_SE_epoch24.pth"
]
MODEL_TYPES = ["resnet101", "resnext50_32x4d", "resnext101_32x8d", "resnext101_32x8d"]
SE_TYPE = [False, False, False, True]   
OUTPUT_CSV = "prediction.csv"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Build Models ----------
def load_model(config, model_type, weight_path, use_se=False):
    config = config.copy()
    config['model_name'] = model_type
    config['use_se'] = use_se
    model = build_model(config)
    model.load_state_dict(torch.load(weight_path, map_location=config['device']))
    model.to(config['device'])
    model.eval()
    return model

models = [load_model(config, model_type, weight_path, use_se)
          for model_type, weight_path, use_se in zip(MODEL_TYPES, MODEL_PATHS, SE_TYPE)]

# ---------- Inference Function ----------
def run_ensemble_inference(test_loader):
    predictions = []

    for images, filenames in tqdm(test_loader, desc="Ensembling Predict"):
        images = images.to(config['device'])
        with torch.no_grad():
            logits_sum = None
            for model in models:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                logits_sum = probs if logits_sum is None else logits_sum + probs

            avg_probs = logits_sum / len(models)
            preds = torch.argmax(avg_probs, dim=1)

        for filename, pred in zip(filenames, preds):
            img_id = os.path.splitext(os.path.basename(filename))[0]
            predictions.append({"image_name": img_id, "pred_label": pred.item()})

    df = pd.DataFrame(predictions)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\u2705 預測完成，Ensemble 結果已儲存在 {OUTPUT_CSV}")


def evaluate_ensemble(models, val_loader, device):
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="Evaluating Ensemble"):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits_sum = None
            for model in models:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                logits_sum = probs if logits_sum is None else logits_sum + probs

            avg_probs = logits_sum / len(models)
            preds = torch.argmax(avg_probs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"\n✅ Ensemble Validation Accuracy: {acc:.4f}")
    return acc

def plot_confusion_matrix(cm, title="Confusion Matrix" , save_path="ensemble_confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)

# ---------- Main ----------
if __name__ == "__main__":
    _, val_loader, test_loader = get_dataloaders(config)
    evaluate_ensemble(models, val_loader, config['device'])

    y_true, y_pred = [], []
    for images, labels in val_loader:
        images = images.to(config["device"])
        labels = labels.to(config["device"])

        with torch.no_grad():
            logits_sum = None
            for model in models:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                logits_sum = probs if logits_sum is None else logits_sum + probs

            avg_probs = logits_sum / len(models)
            preds = torch.argmax(avg_probs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, title="Ensemble Confusion Matrix")

   # run_ensemble_inference(test_loader)
