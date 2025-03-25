import torch
import os

def save_checkpoint(model, epoch, filename=None):
    os.makedirs("checkpoints", exist_ok=True)
    if filename is None:
        filename = f"checkpoints/model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), filename)
    print(f"[Checkpoint] Saved model at {filename}")
