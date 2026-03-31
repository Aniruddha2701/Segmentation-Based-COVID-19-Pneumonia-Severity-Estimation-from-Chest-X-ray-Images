import cv2
import numpy as np


def overlay_masks(image, lung_mask, infection_mask):
    """
    Overlay lung (green) and infection (red) on image
    """

    # Convert grayscale → RGB
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay = image.copy()

    # 🫁 Lung = GREEN
    lung_layer = np.zeros_like(overlay)
    lung_layer[:, :, 1] = lung_mask * 255

    # 🦠 Infection = RED
    infection_layer = np.zeros_like(overlay)
    infection_layer[:, :, 0] = infection_mask * 255  # RGB → R channel = 0

    # Blend
    overlay = cv2.addWeighted(overlay, 1.0, lung_layer, 0.3, 0)
    overlay = cv2.addWeighted(overlay, 1.0, infection_layer, 0.6, 0)

    return overlay