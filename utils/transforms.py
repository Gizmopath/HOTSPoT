import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_geometric_augmentations():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(alpha=50, sigma=4, alpha_affine=0, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.5, p=0.3),
        A.Resize(256, 256)
    ])

def get_color_augmentations():
    return A.Compose([
        A.ColorJitter(brightness=0.1, contrast=0.8, saturation=0.8, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.2, 3.0), p=0.3),
        ToTensorV2()
    ])

def get_validation_pipeline():
    return A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])
