import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import save_checkpoint
from evaluate import evaluate
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(model, train_loader, val_loader, config):
    if config.get("loss_function", "focal") == "focal":
        criterion = FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=config.get("focal_alpha", 1.0)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get("weight_decay", 1e-4)
    )

    if config.get("scheduler", "reduce_on_plateau") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 3),
            verbose=True
        )
    else:
        scheduler = None

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = config.get("early_stop_patience", 7)

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config['num_epochs']}]", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(config['device']), labels.to(config['device'])

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_acc = evaluate(model, val_loader, config)
        print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")

        scheduler.step(val_acc)


        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, epoch, filename=f"checkpoints/best_model{val_acc:.4f}.pth")
            print(f"Best model saved at epoch {epoch+1} with val acc {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. EarlyStop counter: {patience_counter}/{early_stop_patience}")


        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}. Best val acc: {best_acc:.4f} (epoch {best_epoch+1})")
            break


if __name__ == "__main__":
    import yaml
    from model import build_model
    from data_loader import get_dataloaders

    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, _ = get_dataloaders(config)

    model = build_model(config)

    train(model, train_loader, val_loader, config)