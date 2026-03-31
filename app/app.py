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
# LOAD MODELS
# ========================

@st.cache_resource
def load_models():
    lung_model = UNet().to(DEVICE)
    infection_model = UNet().to(DEVICE)

    lung_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/lung_model.pth", map_location=DEVICE))
    infection_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/infection_model.pth", map_location=DEVICE))

    lung_model.eval()
    infection_model.eval()

    # GradCAM
    target_layer = infection_model.dec1
    gradcam = GradCAM(infection_model, target_layer)

    return lung_model, infection_model, gradcam


lung_model, infection_model, gradcam = load_models()


# ========================
# PREPROCESS
# ========================

def preprocess(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
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

    # 🔥 GradCAM
    cam = gradcam.generate(tensor)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    combined = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)

    return original, overlay, combined, severity


# ========================
# UI
# ========================

st.title("🧠 Pneumonia Severity Analysis + Explainability 🔥")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        original, overlay, cam_overlay, severity = predict(image)

        st.subheader("🔍 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

        with col2:
            st.image(cam_overlay, caption="Grad-CAM Explainability 🔥", use_container_width=True)

        st.markdown("### 📊 Severity")

        st.write(f"Score: {severity['severity_score']}")
        st.write(f"Percent: {severity['severity_percent']}%")
        st.write(f"Category: {severity['category']}")

        if severity["category"] == "Severe":
            st.error("⚠️ Severe Infection")
        elif severity["category"] == "Moderate":
            st.warning("⚠️ Moderate Infection")
        else:
            st.success("✅ Mild Infection")