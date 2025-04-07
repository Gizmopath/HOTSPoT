import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from augmentation import get_geometric_augmentations, get_color_augmentations, get_validation_pipeline
from dataset import CustomDataset
from early_stopping import EarlyStopping
from model import MultiClassSegFormer
from metrics import calculate_metrics_per_class
from train import run_epoch

# Configurazione dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path ai dati
train_img_dir = "data/train/images"
train_mask_dir = "data/train/masks"
val_img_dir = "data/val/images"
val_mask_dir = "data/val/masks"
model_save_path = "segformer_model.pth"

# Hyperparametri
batch_size = 8
num_epochs = 50
num_classes = 3
learning_rate = 1e-4
patience = 5

# Augmentation e dataset
train_geometric_aug = get_geometric_augmentations()
train_color_aug = get_color_augmentations()
val_pipeline = get_validation_pipeline()

train_dataset = CustomDataset(train_img_dir, train_mask_dir, train_geometric_aug, train_color_aug)
val_dataset = CustomDataset(val_img_dir, val_mask_dir, geometric_augmentations=val_pipeline)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Modello, loss e ottimizzatore
model = MultiClassSegFormer(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler = GradScaler()
early_stopping = EarlyStopping(patience=patience, verbose=True)

# Addestramento
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Training
    train_loss, train_iou, train_accuracy, train_class_ious, train_class_accuracies = run_epoch(
        model, train_loader, criterion, num_classes, optimizer, scaler, is_train=True
    )

    print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Validation
    val_loss, val_iou, val_accuracy, val_class_ious, val_class_accuracies = run_epoch(
        model, val_loader, criterion, num_classes, is_train=False
    )

    print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Scheduler e early stopping
    scheduler.step(val_loss)
    early_stopping(val_loss)

    if early_stopping.early_stop:
        print("Early stopping activated. Stopping training.")
        break

# Salvataggio del modello
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}.")
