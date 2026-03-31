import os
import pandas as pd
import cv2
import numpy as np

from config.config import DATASET_PATH   # ✅ USE CONFIG

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def count_files(path):
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(IMG_EXT)])


def count_non_empty_masks(path):
    total, non_empty = 0, 0

    if not os.path.exists(path):
        return 0, 0

    for f in os.listdir(path):
        if not f.lower().endswith(IMG_EXT):
            continue

        total += 1
        mask = cv2.imread(os.path.join(path, f), 0)

        if mask is not None and np.sum(mask) > 0:
            non_empty += 1

    return total, non_empty


def analyze():
    records = []

    for dataset_type in os.listdir(DATASET_PATH):
        dpath = os.path.join(DATASET_PATH, dataset_type)

        if not os.path.isdir(dpath):
            continue

        for split in ["train", "val", "test"]:
            spath = os.path.join(dpath, split)

            if not os.path.exists(spath):
                continue

            for cls in os.listdir(spath):
                cpath = os.path.join(spath, cls)

                if not os.path.isdir(cpath):
                    continue

                img_path = os.path.join(cpath, "images")
                lung_path = os.path.join(cpath, "lung masks")
                inf_path = os.path.join(cpath, "infection masks")

                images = count_files(img_path)
                lung_masks = count_files(lung_path)
                inf_total, inf_valid = count_non_empty_masks(inf_path)

                records.append({
                    "Dataset": dataset_type,
                    "Split": split,
                    "Class": cls,
                    "Images": images,
                    "Lung Masks": lung_masks,
                    "Inf Masks Total": inf_total,
                    "Inf Masks Valid": inf_valid,
                    "Valid %": round((inf_valid / inf_total * 100) if inf_total else 0, 2)
                })

    df = pd.DataFrame(records)

    if df.empty:
        print("❌ No data found. Check DATASET_PATH")
        return

    print("\n📊 FULL DATA:\n", df)

    summary = df.groupby(["Dataset", "Class"])[
        ["Images", "Lung Masks", "Inf Masks Valid"]
    ].sum()

    print("\n📈 SUMMARY:\n", summary)

    df.to_csv("dataset_full.csv", index=False)
    summary.to_csv("dataset_summary.csv")

    print("\n✅ CSVs saved successfully")


if __name__ == "__main__":
    analyze()