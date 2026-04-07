import os
import torch

# ========================
# PATHS
# ========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "/Re-creation/Dataset"
)

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ========================
# TRAINING
# ========================

IMG_SIZE = 256
EPOCHS = 10   # 🔥 allows proper resume + early stopping
LR = 5e-5

# 🔥 safer for infection segmentation
BATCH_SIZE = 4 if torch.cuda.is_available() else 2

# ========================
# DEVICE
# ========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# PERFORMANCE BOOST
# ========================

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True