import numpy as np


# ========================
# BASIC SEVERITY FUNCTION
# ========================
def compute_severity(lung_mask, infection_mask):
    """
    Computes severity as ratio of infection area to lung area
    """

    lung_area = np.sum(lung_mask)
    infection_area = np.sum(infection_mask)

    # Safety check
    if lung_area < 10:   # avoid noise / division instability
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