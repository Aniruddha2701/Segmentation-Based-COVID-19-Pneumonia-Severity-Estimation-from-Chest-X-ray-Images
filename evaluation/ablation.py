import torch
from tqdm import tqdm

from models.unet import UNet
from datasets.infection_dataset import InfectionDataset
from datasets.lung_dataset import LungDataset
from utils.severity import compute_severity
from config.config import *


def evaluate_without_severity():
    model = UNet().to(DEVICE)
    model.load_state_dict(
        torch.load(f"{CHECKPOINT_DIR}/infection_model.pth", map_location=DEVICE)
    )
    model.eval()

    dataset = InfectionDataset(DATASET_PATH, split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    total_dice = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = torch.sigmoid(model(imgs))
            preds = (preds > 0.5).float()

            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()

            dice = (2 * intersection) / (union + 1e-8)
            total_dice += dice.item()

    return total_dice / len(loader)


def evaluate_with_severity():
    lung_model = UNet().to(DEVICE)
    infection_model = UNet().to(DEVICE)

    lung_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/lung_model.pth", map_location=DEVICE))
    infection_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/infection_model.pth", map_location=DEVICE))

    lung_model.eval()
    infection_model.eval()

    dataset = InfectionDataset(DATASET_PATH, split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    total_severity = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)

            lung_pred = torch.sigmoid(lung_model(imgs))
            infection_pred = torch.sigmoid(infection_model(imgs))

            lung = (lung_pred.squeeze().cpu().numpy() > 0.5).astype(int)
            infection = (infection_pred.squeeze().cpu().numpy() > 0.5).astype(int)

            severity = compute_severity(lung, infection)
            total_severity += severity

    return total_severity / len(loader)


if __name__ == "__main__":
    print("Running Ablation Study...\n")

    dice_score = evaluate_without_severity()
    severity_score = evaluate_with_severity()

    print("\n📊 RESULTS:")
    print(f"Segmentation Dice Score: {dice_score:.4f}")
    print(f"Average Severity Score: {severity_score:.4f}")