import torch
import matplotlib.pyplot as plt

from models.unet import UNet
from datasets.lung_dataset import LungDataset
from config.config import *

# ========================
# LOAD MODEL
# ========================

model = UNet().to(DEVICE)
model.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/{MODEL_NAME}", map_location=DEVICE)
)
model.eval()

# ========================
# LOAD DATA
# ========================

dataset = LungDataset(DATASET_PATH, split="test")

img, mask = dataset[0]

img_tensor = img.unsqueeze(0).to(DEVICE)

# ========================
# INFERENCE
# ========================

with torch.no_grad():
    pred = model(img_tensor)
    pred = torch.sigmoid(pred)   # 🔥 IMPORTANT

pred = pred.squeeze().cpu().numpy()
img = img.squeeze().numpy()
mask = mask.squeeze().numpy()

# Binary mask
pred_bin = (pred > 0.5).astype(float)

# ========================
# VISUALIZATION
# ========================

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred_bin, cmap="gray")
plt.axis("off")

plt.show()