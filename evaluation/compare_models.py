import pandas as pd
import matplotlib.pyplot as plt
import os

# ========================
# LOAD LOGS
# ========================

lung_path = "logs/lung_training_log.csv"
inf_path = "logs/infection_training_log.csv"

if not os.path.exists(lung_path) or not os.path.exists(inf_path):
    raise FileNotFoundError("❌ Log files not found in /logs folder")

lung_df = pd.read_csv(lung_path)
inf_df = pd.read_csv(inf_path)

# ========================
# CLEAN EPOCH AXIS
# ========================

lung_epochs = range(1, len(lung_df) + 1)
inf_epochs = range(1, len(inf_df) + 1)

# ========================
# DICE COMPARISON
# ========================

plt.figure(figsize=(10, 6))

plt.plot(lung_epochs, lung_df["Dice"], label="Lung (Train)", linewidth=2)
plt.plot(inf_epochs, inf_df["Dice"], label="Infection (Train)", linewidth=2)

# 🔥 If validation exists
if "ValDice" in lung_df.columns:
    plt.plot(lung_epochs, lung_df["ValDice"], '--', label="Lung (Val)")
if "ValDice" in inf_df.columns:
    plt.plot(inf_epochs, inf_df["ValDice"], '--', label="Infection (Val)")

# 🔥 Highlight best infection Dice
best_idx = inf_df["Dice"].idxmax()
best_epoch = best_idx + 1
best_val = inf_df["Dice"].max()

plt.scatter(best_epoch, best_val)
plt.text(best_epoch, best_val, f"{best_val:.3f}", fontsize=9)

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Model Comparison (Dice Score)")
plt.legend()
plt.grid()

plt.savefig("logs/comparison_dice.png", dpi=300)
plt.show()

# ========================
# IoU COMPARISON
# ========================

plt.figure(figsize=(10, 6))

plt.plot(lung_epochs, lung_df["IoU"], label="Lung (Train)", linewidth=2)
plt.plot(inf_epochs, inf_df["IoU"], label="Infection (Train)", linewidth=2)

if "ValIoU" in lung_df.columns:
    plt.plot(lung_epochs, lung_df["ValIoU"], '--', label="Lung (Val)")
if "ValIoU" in inf_df.columns:
    plt.plot(inf_epochs, inf_df["ValIoU"], '--', label="Infection (Val)")

plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.title("Model Comparison (IoU)")
plt.legend()
plt.grid()

plt.savefig("logs/comparison_iou.png", dpi=300)
plt.show()

# ========================
# LOSS CURVES (🔥 VERY IMPORTANT FOR REPORT)
# ========================

plt.figure(figsize=(10, 6))

if "TrainLoss" in lung_df.columns:
    plt.plot(lung_epochs, lung_df["TrainLoss"], label="Lung Train Loss")
if "ValLoss" in lung_df.columns:
    plt.plot(lung_epochs, lung_df["ValLoss"], '--', label="Lung Val Loss")

if "TrainLoss" in inf_df.columns:
    plt.plot(inf_epochs, inf_df["TrainLoss"], label="Infection Train Loss")
if "ValLoss" in inf_df.columns:
    plt.plot(inf_epochs, inf_df["ValLoss"], '--', label="Infection Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()

plt.savefig("logs/comparison_loss.png", dpi=300)
plt.show()