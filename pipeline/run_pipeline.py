import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models.unet import UNet
from explainability.gradcam import GradCAM
from utils.severity import get_severity_info
from utils.visualization import overlay_masks
from config.config import *

print("🚀 Using device:", DEVICE)
print("📁 CHECKPOINT_DIR:", CHECKPOINT_DIR)

# ========================
# LOAD MODELS
# ========================

lung_model = UNet().to(DEVICE)
infection_model = UNet().to(DEVICE)

lung_path = os.path.join(CHECKPOINT_DIR, "lung_model.pth")
infection_path = os.path.join(CHECKPOINT_DIR, "infection_model.pth")

print("Looking for lung model at:", lung_path)
print("Looking for infection model at:", infection_path)

if not os.path.exists(lung_path):
    raise FileNotFoundError(f"❌ Lung model not found at {lung_path}")

if not os.path.exists(infection_path):
    raise FileNotFoundError(f"❌ Infection model not found at {infection_path}")


# ========================
# SAFE LOAD FUNCTION 🔥
# ========================

def load_model(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    # New format (your current training)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"✅ Loaded checkpoint (with optimizer info) from {path}")

    # Old format (weights only)
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded raw weights from {path}")


load_model(lung_model, lung_path)
load_model(infection_model, infection_path)

lung_model.eval()
infection_model.eval()

# Ensure Grad-CAM works properly
infection_model.requires_grad_(True)

# ========================
# INIT GRAD-CAM 🔥
# ========================

target_layer = infection_model.dec1
gradcam = GradCAM(infection_model, target_layer)

# ========================
# PREPROCESS
# ========================

def preprocess_image(img_path):
    img = cv2.imread(img_path, 0)

    if img is None:
        raise ValueError(f"❌ Image not found at {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE), img


# ========================
# PIPELINE
# ========================

def run_pipeline(img_path):
    img_tensor, original_img = preprocess_image(img_path)

    with torch.no_grad():
        lung_pred = torch.sigmoid(lung_model(img_tensor))
        infection_pred = torch.sigmoid(infection_model(img_tensor))

    lung_mask = lung_pred.squeeze().cpu().numpy()
    infection_mask = infection_pred.squeeze().cpu().numpy()

    lung_bin = (lung_mask > 0.5).astype(np.uint8)
    infection_bin = (infection_mask > 0.5).astype(np.uint8)

    # ========================
    # SEVERITY
    # ========================
    severity_info = get_severity_info(lung_bin, infection_bin)

    # ========================
    # GRAD-CAM 🔥
    # ========================
    cam = gradcam.generate(img_tensor)

    return original_img, lung_bin, infection_bin, cam, severity_info


# ========================
# VISUALIZATION
# ========================

def visualize_results(img, lung, infection, cam, severity_info):
    overlay = overlay_masks(img, lung, infection)

    # 🔥 Normalize CAM properly
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap = cv2.applyColorMap(
        (cam_norm * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    combined = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Segmentation Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM + Overlay 🔥")
    plt.imshow(combined)
    plt.axis("off")

    plt.suptitle(
        f"Severity: {severity_info['severity_percent']}% ({severity_info['category']})",
        fontsize=14,
        color="red"
    )

    plt.tight_layout()
    plt.show()


# ========================
# TEST
# ========================

if __name__ == "__main__":
    test_image = os.path.join(
        DATASET_PATH,
        "Infection Segmentation Data",
        "test",
        "COVID-19",
        "images",
        "covid_1615.png"
    )

    print("\n🧪 Running pipeline...\n")

    img, lung, infection, cam, severity = run_pipeline(test_image)

    print("\n🔍 Severity Info:")
    print(severity)

    visualize_results(img, lung, infection, cam, severity)