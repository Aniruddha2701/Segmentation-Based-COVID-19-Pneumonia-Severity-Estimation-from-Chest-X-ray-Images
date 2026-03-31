import pandas as pd
import matplotlib.pyplot as plt

# Load logs
lung_df = pd.read_csv("logs/lung_training_log.csv")
inf_df = pd.read_csv("logs/infection_training_log.csv")

# ========================
# DICE COMPARISON
# ========================

plt.figure(figsize=(8, 5))

plt.plot(lung_df["Dice"], label="Lung Model", linewidth=2)
plt.plot(inf_df["Dice"], label="Infection Model", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Model Comparison (Dice)")
plt.legend()
plt.grid()

plt.savefig("logs/comparison_dice.png")
plt.show()


# ========================
# IoU COMPARISON (BONUS 🔥)
# ========================

plt.figure(figsize=(8, 5))

plt.plot(lung_df["IoU"], label="Lung Model", linewidth=2)
plt.plot(inf_df["IoU"], label="Infection Model", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.title("Model Comparison (IoU)")
plt.legend()
plt.grid()

plt.savefig("logs/comparison_iou.png")
plt.show()