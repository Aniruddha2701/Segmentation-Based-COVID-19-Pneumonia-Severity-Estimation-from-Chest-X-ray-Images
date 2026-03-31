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

# ========================
# LOAD MODELS
# ========================

lung_model = UNet().to(DEVICE)
infection_model = UNet().to(DEVICE)

lung_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "lung_model.pth"), map_location=DEVICE)
)

infection_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "infection_model.pth"), map_location=DEVICE)
)

lung_model.eval()
infection_model.eval()

# ========================
# INIT GRAD-CAM 🔥
# ========================

# Target last decoder layer (strong spatial info)
target_layer = infection_model.dec1 # target_layer = infection_model.dec2 if gradcam is too noisy.
gradcam = GradCAM(infection_model, target_layer)


# ========================
# PREPROCESS
# ========================

def preprocess_image(img_path):
    img = cv2.imread(img_path, 0)

    if img is None:
        raise ValueError("Image not found")

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

    # Convert to numpy
    lung_mask = lung_pred.squeeze().cpu().numpy()
    infection_mask = infection_pred.squeeze().cpu().numpy()

    lung_bin = (lung_mask > 0.5).astype(np.uint8)
    infection_bin = (infection_mask > 0.5).astype(np.uint8)

    # Severity
    severity_info = get_severity_info(lung_bin, infection_bin)

    # 🔥 Grad-CAM (no torch.no_grad here!)
    cam = gradcam.generate(img_tensor)

    return original_img, lung_bin, infection_bin, cam, severity_info


# ========================
# VISUALIZATION
# ========================

def visualize_results(img, lung, infection, cam, severity_info):
    # Base overlay (lung + infection)
    overlay = overlay_masks(img, lung, infection)

    # 🔥 Heatmap
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend Grad-CAM with overlay
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

    plt.show()


# ========================
# TEST
# ========================

if __name__ == "__main__":
    test_image = os.path.join(
        DATASET_PATH,
        "Lung Segmentation Data",
        "test",
        "COVID-19",
        "images",
        "1.png"
    )

    img, lung, infection, cam, severity = run_pipeline(test_image)

    print("\n🔍 Severity Info:")
    print(severity)

    visualize_results(img, lung, infection, cam, severity)