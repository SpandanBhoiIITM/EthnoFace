import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class UTKFaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # ---- Clean filename robustly ----
        # Handles both ".jpg" and ".jpg.chip.jpg"
        base = img_name
        while base.endswith(".jpg"):
            base = os.path.splitext(base)[0]

        # Example: "24_1_3_20170109142408075"
        parts = base.split("_")

        # Ethnicity (race) is always the 3rd field
        try:
            ethnicity = int(parts[2])
        except Exception as e:
            print(f"Skipping bad file: {img_name}, error: {e}")
            ethnicity = 4  # "Others"

        # ---- Load and preprocess image ----
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, ethnicity
