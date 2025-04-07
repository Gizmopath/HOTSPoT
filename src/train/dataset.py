import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, geometric_augmentations=None, color_augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.geometric_augmentations = geometric_augmentations
        self.color_augmentations = color_augmentations

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_filenames[index])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))

        if self.geometric_augmentations:
            augmented = self.geometric_augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.color_augmentations:
            image = self.color_augmentations(image=image)['image']

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image = image.float() / 255.0

        mask = torch.from_numpy(self.encode_mask(mask)).long()
        return image, mask

    @staticmethod
    def encode_mask(mask):
        target_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        target_mask[(mask == [0, 0, 0]).all(axis=-1)] = 0
        target_mask[(mask == [255, 0, 0]).all(axis=-1)] = 1
        target_mask[(mask == [255, 255, 255]).all(axis=-1)] = 2
        return target_mask
