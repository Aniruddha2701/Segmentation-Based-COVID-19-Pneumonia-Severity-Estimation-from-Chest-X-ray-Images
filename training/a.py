import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.infection_dataset import InfectionDataset
from models.unet import UNet
from config.config import *

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
# DATA (ONLY VAL NEEDED)
# ========================

val_loader = DataLoader(
    InfectionDataset(DATASET_PATH, split="test"),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2 if DEVICE.type == "cuda" else 0,
    pin_memory=True if DEVICE.type == "cuda" else False
)

# ========================
# MODEL
# ========================

model = UNet().to(DEVICE)

# ========================
# LOAD CHECKPOINT (MANDATORY)
# ========================

checkpoint_path = os.path.join(CHECKPOINT_DIR, "infection_model.pth")

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("❌ No trained model found. Training required.")

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

print("✅ Model loaded successfully. No training will be performed.")

# ========================
# VALIDATION ONLY
# ========================

model.eval()

total_dice = total_iou = 0
total_precision = total_recall = 0

with torch.no_grad():
    for imgs, masks in tqdm(val_loader):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)

        total_dice += dice_score(preds, masks).item()
        total_iou += iou_score(preds, masks).item()

        p, r = precision_recall(preds, masks)
        total_precision += p.item()
        total_recall += r.item()

n = len(val_loader)

print("\n📊 Evaluation Results:")
print(f"Dice:      {total_dice / n:.4f}")
print(f"IoU:       {total_iou / n:.4f}")
print(f"Precision: {total_precision / n:.4f}")
print(f"Recall:    {total_recall / n:.4f}")

print("\n🎯 Done. Model was NOT retrained.")