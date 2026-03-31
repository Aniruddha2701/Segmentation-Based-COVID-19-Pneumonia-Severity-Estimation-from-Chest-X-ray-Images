import numpy as np


# ========================
# BASIC SEVERITY FUNCTION
# ========================
def compute_severity(lung_mask, infection_mask):
    """
    Computes severity as ratio of infection area to lung area
    """

    # Convert to binary
    lung = (lung_mask > 0.5).astype(np.uint8)
    infection = (infection_mask > 0.5).astype(np.uint8)

    lung_area = np.sum(lung)
    infection_area = np.sum(infection)

    if lung_area == 0:
        return 0.0

    severity = infection_area / lung_area
    return severity


# ========================
# SEVERITY CATEGORY
# ========================
def classify_severity(severity):
    """
    Categorize severity into levels
    """

    if severity < 0.25:
        return "Mild"
    elif severity < 0.5:
        return "Moderate"
    else:
        return "Severe"


# ========================
# FULL PIPELINE FUNCTION
# ========================
def get_severity_info(lung_mask, infection_mask):
    severity = compute_severity(lung_mask, infection_mask)
    category = classify_severity(severity)

    return {
        "severity_score": round(severity, 4),
        "severity_percent": round(severity * 100, 2),
        "category": category
    }