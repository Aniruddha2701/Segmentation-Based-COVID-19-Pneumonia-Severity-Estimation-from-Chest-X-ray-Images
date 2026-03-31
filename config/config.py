import os
import torch

# ========================
# PATHS
# ========================

# Automatically detect project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(BASE_DIR, "Dataset"))

# ========================
# TRAINING
# ========================

IMG_SIZE = 256
EPOCHS = 20
LR = 1e-4

# Dynamic batch size
BATCH_SIZE = 8 if torch.cuda.is_available() else 2

# ========================
# DEVICE
# ========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# SAVE PATHS
# ========================

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_NAME = "lung_model.pth"