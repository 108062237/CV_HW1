from data_loader import get_dataloaders
from model.model import build_model
from train import train
from evaluate import evaluate
import yaml

if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = build_model(config)

    train(model, train_loader, val_loader, config)
    evaluate(model, test_loader, config)
