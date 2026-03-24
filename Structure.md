project_root/
│
├── data/
│   ├── raw/                         # Original COVID-QU-Ex dataset
│   ├── processed/
│   │   ├── lung/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── infection/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│
├── models/
│   ├── unet.py                      # Lung segmentation model
│   ├── attention_unet.py           # Infection segmentation model
│
├── datasets/
│   ├── lung_dataset.py
│   ├── infection_dataset.py
│
├── preprocessing/
│   ├── transforms.py                # CLAHE, resize, normalize
│   ├── mask_utils.py                # mask cleaning, thresholding
│
├── training/
│   ├── train_lung.py
│   ├── train_infection.py
│
├── inference/
│   ├── predict.py                   # full pipeline
│   ├── severity.py                  # severity calculation
│
├── utils/
│   ├── metrics.py                   # Dice, IoU
│   ├── visualization.py             # overlay masks
│   ├── config.py                    # hyperparameters
│
├── checkpoints/
│   ├── lung_model.pth
│   ├── infection_model.pth
│
├── results/
│   ├── outputs/
│   ├── plots/
│
└── main.py                          # entry point (optional)