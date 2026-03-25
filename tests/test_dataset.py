import os
import sys
import matplotlib.pyplot as plt

# ✅ Ensure project root is in path (MUST be first)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.lung_dataset import LungDataset
from datasets.infection_dataset import InfectionDataset


DATASET_PATH = r"D:\Re-creation\Dataset"


def show_sample(img, mask, title_prefix=""):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"{title_prefix} Image")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"{title_prefix} Mask")
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.axis("off")

    plt.show()


def main():
    print("\n🔍 Loading datasets...\n")

    # Load datasets
    lung_data = LungDataset(DATASET_PATH, "train")
    infection_data = InfectionDataset(DATASET_PATH, "train")

    # Print sizes
    print("\n📊 Dataset Sizes:")
    print(f"Lung Dataset Size: {len(lung_data)}")
    print(f"Infection Dataset Size: {len(infection_data)}")

    # ✅ Sanity checks
    assert len(lung_data) > 0, "Lung dataset is empty!"
    assert len(infection_data) > 0, "Infection dataset is empty!"

    # Test Lung sample
    print("\n🫁 Testing Lung Dataset...")
    img, mask = lung_data[0]

    print("Lung sample shape:", img.shape, mask.shape)
    show_sample(img, mask, "Lung")

    # Test Infection sample
    print("\n🦠 Testing Infection Dataset...")
    img, mask = infection_data[0]

    print("Infection sample shape:", img.shape, mask.shape)
    show_sample(img, mask, "Infection")

    print("\n✅ Dataset verification complete.\n")


if __name__ == "__main__":
    main()