import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from config.config import IMG_SIZE


class LungDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        dataset_path = os.path.join(root_dir, "Lung Segmentation Data", split)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"{dataset_path} not found")

        for cls in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, cls)

            img_dir = os.path.join(class_path, "images")
            mask_dir = os.path.join(class_path, "lung masks")

            if not os.path.exists(img_dir):
                continue

            for file in sorted(os.listdir(img_dir)):
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(img_dir, file)
                mask_path = os.path.join(mask_dir, file)

                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        print(f"[LungDataset] {split}: {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], 0)
        mask = cv2.imread(self.mask_paths[idx], 0)

        if img is None or mask is None:
            raise ValueError(f"Error loading {self.image_paths[idx]}")

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask