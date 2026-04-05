import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.lung_dataset import LungDataset
from models.unet import UNet
from config.config import *

# ========================
# LOSS
# ========================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

def combined_loss(preds, targets):
    return bce_loss(preds, targets) + dice_loss(preds, targets)


# ========================
# METRICS
# ========================

def dice_score(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2 * intersection) / (union + 1e-8)


def iou_score(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + 1e-8)


def precision_recall(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return precision, recall


# ========================
# DATA (STABLE)
# ========================

train_loader = DataLoader(
    LungDataset(DATASET_PATH, split="train"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 if DEVICE.type == "cuda" else 0,
    pin_memory=True if DEVICE.type == "cuda" else False
)

val_loader = DataLoader(
    LungDataset(DATASET_PATH, split="val"),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2 if DEVICE.type == "cuda" else 0,
    pin_memory=True if DEVICE.type == "cuda" else False
)


# ========================
# MODEL
# ========================

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

# ========================
# RESUME (SAFE VERSION 🔥)
# ========================

checkpoint_path = os.path.join(CHECKPOINT_DIR, "lung_model.pth")

start_epoch = 0

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"✅ Resuming training from epoch {start_epoch}")

    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")
        print("Training from scratch...")

# ========================
# TRACKING
# ========================

train_losses, val_losses = [], []
dice_list, iou_list = [], []
precision_list, recall_list = [], []

plt.ion()

# ========================
# TRAIN FUNCTIONS
# ========================

def train_one_epoch(loader):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        preds = model(imgs)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(loader):
    model.eval()

    total_loss = total_dice = total_iou = 0
    total_precision = total_recall = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = combined_loss(preds, masks)

            total_loss += loss.item()
            total_dice += dice_score(preds, masks).item()
            total_iou += iou_score(preds, masks).item()

            p, r = precision_recall(preds, masks)
            total_precision += p.item()
            total_recall += r.item()

    n = len(loader)

    return (
        total_loss / n,
        total_dice / n,
        total_iou / n,
        total_precision / n,
        total_recall / n
    )


# ========================
# TRAIN LOOP
# ========================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_file = "logs/lung_training_log.csv"

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "ValLoss", "Dice", "IoU", "Precision", "Recall"])

best_val = float("inf")
patience = 5
counter = 0

for epoch in range(start_epoch, EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} (Resume from {start_epoch+1})")

    train_loss = train_one_epoch(train_loader)
    val_loss, dice, iou, precision, recall = validate(val_loader)

    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Dice:       {dice:.4f}")
    print(f"IoU:        {iou:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    dice_list.append(dice)
    iou_list.append(iou)
    precision_list.append(precision)
    recall_list.append(recall)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, val_loss, dice, iou, precision, recall])

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, checkpoint_path)
        print("✅ Lung model saved!")
    else:
        counter += 1

    if counter >= patience:
        print("⛔ Early stopping triggered")
        break

    plt.clf()
    plt.subplot(2,2,1); plt.plot(train_losses); plt.plot(val_losses); plt.title("Loss")
    plt.subplot(2,2,2); plt.plot(dice_list); plt.plot(iou_list); plt.title("Dice/IoU")
    plt.subplot(2,2,3); plt.plot(precision_list); plt.plot(recall_list); plt.title("Prec/Rec")
    plt.pause(0.1)

plt.ioff()
plt.figure(figsize=(10,5))
plt.plot(dice_list); plt.plot(iou_list)
plt.plot(precision_list); plt.plot(recall_list)
plt.legend(["Dice","IoU","Precision","Recall"])
plt.title("Lung Model Metrics")
plt.savefig("logs/lung_metrics.png")
plt.show()

print("\n🎉 Lung training complete!")