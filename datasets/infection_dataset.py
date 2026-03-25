import os
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class InfectionDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.mask_paths = []

        dataset_path = os.path.join(root_dir, "Infection Segmentation Data", split)
        covid_path = os.path.join(dataset_path, "COVID-19")

        img_dir = os.path.join(covid_path, "images")
        mask_dir = os.path.join(covid_path, "infection masks")

        # Safety check
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Check dataset path: {img_dir}")

        files = sorted(os.listdir(img_dir))

        for file in files:
            # Only valid image files
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(img_dir, file)
            mask_path = os.path.join(mask_dir, file)

            # Ensure mask exists
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        # ✅ FIXED: print inside __init__
        print(f"[InfectionDataset] Loaded {len(self.image_paths)} samples from '{split}' split.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        # Safety check
        if img is None or mask is None:
            raise ValueError(f"Error loading file: {img_path}")

        # Resize
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Optional transform (future use)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Normalize
        img = img / 255.0
        mask = mask / 255.0

        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask


# 🔍 TEST BLOCK (SAFE TO KEEP HERE OR MOVE TO tests/)
if __name__ == "__main__":
    dataset = InfectionDataset(root_dir=r"D:\Re-creation\Dataset", split="train")

    print("Dataset size:", len(dataset))

    img, mask = dataset[0]

    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")

    # Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.axis("off")

    plt.show()