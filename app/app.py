import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

from models.unet import UNet
from explainability.gradcam import GradCAM
from utils.severity import get_severity_info
from utils.visualization import overlay_masks
from config.config import *


# ========================
# PAGE CONFIG
# ========================

st.set_page_config(page_title="Pneumonia AI", layout="wide")


# ========================
# SAFE LOAD FUNCTION 🔥
# ========================

def load_model(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)


# ========================
# LOAD MODELS
# ========================

@st.cache_resource
def load_models():
    lung_model = UNet().to(DEVICE)
    infection_model = UNet().to(DEVICE)

    load_model(lung_model, f"{CHECKPOINT_DIR}/lung_model.pth")
    load_model(infection_model, f"{CHECKPOINT_DIR}/infection_model.pth")

    lung_model.eval()
    infection_model.eval()

    infection_model.requires_grad_(True)

    target_layer = infection_model.dec1
    gradcam = GradCAM(infection_model, target_layer)

    return lung_model, infection_model, gradcam


lung_model, infection_model, gradcam = load_models()


# ========================
# SIDEBAR 🔥
# ========================

st.sidebar.title("⚙️ Model Info")
st.sidebar.write("Model: U-Net")
st.sidebar.write("Task: Pneumonia Severity Analysis")
st.sidebar.write(f"Device: {DEVICE}")


# ========================
# PREPROCESS
# ========================

def preprocess(image):
    image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        pass
    else:
        raise ValueError("Unsupported image format")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE), image


# ========================
# PREDICT
# ========================

def predict(image):
    tensor, original = preprocess(image)

    with torch.no_grad():
        lung_pred = torch.sigmoid(lung_model(tensor))
        infection_pred = torch.sigmoid(infection_model(tensor))

    lung_mask = (lung_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    infection_mask = (infection_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    severity = get_severity_info(lung_mask, infection_mask)

    overlay = overlay_masks(original, lung_mask, infection_mask)

    # Grad-CAM
    cam = gradcam.generate(tensor)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    combined = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)

    return original, overlay, combined, severity


# ========================
# UI
# ========================

st.title("🧠 Pneumonia Severity Analysis + Explainability 🔥")

st.info(
    "This system segments lung and infection regions, then estimates severity "
    "based on infected lung area."
)

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing... 🔍"):
            original, overlay, cam_overlay, severity = predict(image)

        st.subheader("🔍 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

        with col2:
            st.image(cam_overlay, caption="Grad-CAM Explainability 🔥", use_container_width=True)

        st.markdown("### 📊 Severity")

        # 🔥 METRIC + PROGRESS BAR
        st.metric("Severity %", f"{severity['severity_percent']:.2f}%")
        st.progress(int(severity["severity_percent"]))

        # 🔥 COLOR FEEDBACK
        if severity["category"] == "Severe":
            st.error("🔴 Severe Infection")
        elif severity["category"] == "Moderate":
            st.warning("🟠 Moderate Infection")
        else:
            st.success("🟢 Mild Infection")

        # 🔥 DOWNLOAD BUTTON
        _, buffer = cv2.imencode(".png", cam_overlay)
        st.download_button(
            label="📥 Download Result",
            data=buffer.tobytes(),
            file_name="pneumonia_result.png",
            mime="image/png"
        )