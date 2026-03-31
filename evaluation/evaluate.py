import torch
from tqdm import tqdm

from models.unet import UNet
from datasets.lung_dataset import LungDataset
from config.config import *


def dice_score(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


model = UNet().to(DEVICE)
model.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/{MODEL_NAME}", map_location=DEVICE)
)
model.eval()

dataset = LungDataset(DATASET_PATH, split="test")

loader = torch.utils.data.DataLoader(dataset, batch_size=4)

total_dice = 0

with torch.no_grad():
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = torch.sigmoid(model(imgs))   # 🔥 FIX
        preds = (preds > 0.5).float()

        total_dice += dice_score(preds, masks).item()

print("Average Dice Score:", total_dice / len(loader))