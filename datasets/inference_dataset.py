import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir)) if mask_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        fname = self.image_filenames[index]
        img_path = os.path.join(self.image_dir, fname)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = None
        if self.mask_filenames:
            mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])
            mask = np.array(Image.open(mask_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask) if mask is not None else self.transform(image=image)
            image = augmented['image']
            mask = augmented.get('mask') if mask is not None else None

        if mask is not None:
            mask = torch.from_numpy(self.encode_mask(mask)).long()
            return image, mask
        return image, fname

    @staticmethod
    def encode_mask(mask):
        target_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        target_mask[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 0
        target_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 1
        target_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255)] = 2
        return target_mask
