import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.lung_dataset import LungDataset
from models.unet import UNet


# ========================
# CONFIG
# ========================
DATASET_PATH = r"D:\Re-creation\Dataset"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# ========================
# LOSS FUNCTIONS
# ========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


bce_loss = nn.BCELoss()
dice_loss = DiceLoss()


def combined_loss(preds, targets):
    return bce_loss(preds, targets) + dice_loss(preds, targets)


# ========================
# DATA
# ========================
train_dataset = LungDataset(DATASET_PATH, split="train")
val_dataset = LungDataset(DATASET_PATH, split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ========================
# MODEL
# ========================
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ========================
# TRAIN FUNCTION
# ========================
def train_one_epoch(loader):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(loader):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ========================
# VALIDATION
# ========================
def validate(loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = combined_loss(preds, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


# ========================
# TRAIN LOOP
# ========================
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch(train_loader)
    val_loss = validate(val_loader)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/lung_model.pth")

        print("✅ Model saved!")

print("\n🎉 Training complete!")