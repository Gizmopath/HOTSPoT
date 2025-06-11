# Full training script
from datasets.custom_dataset import CustomDataset
from utils.transforms import get_geometric_augmentations, get_color_augmentations, get_validation_pipeline
from models.segformer import MultiClassSegFormer
from utils.early_stopping import EarlyStopping
from utils.metrics import calculate_metrics_per_class
import torch, os, numpy as np, contextlib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './data/Original'
mask_dir = './data/Mask'
batch_size = 64
num_epochs = 50
learning_rate = 0.0001

geometric_augmentations = get_geometric_augmentations()
color_augmentations = get_color_augmentations()
dataset = CustomDataset(data_dir, mask_dir, geometric_augmentations, color_augmentations)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

model = MultiClassSegFormer(num_classes=3).to(device)
class_weights = torch.tensor([1.0, 1.0, 5.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
early_stopping = EarlyStopping(patience=5, verbose=True)
writer = SummaryWriter()
best_iou = 0.0

def run_epoch(model, loader, criterion, num_classes, optimizer=None, scaler=None, is_train=True):
    model.train() if is_train else model.eval()
    epoch_loss, overall_ious, overall_accuracies = 0.0, [], []
    class_ious = [[] for _ in range(num_classes)]
    class_accuracies = [[] for _ in range(num_classes)]

    loop = tqdm(loader, leave=False)
    loop.set_description("Training" if is_train else "Validation")
    with torch.no_grad() if not is_train else contextlib.nullcontext():
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(images)
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
            if is_train:
                scaler.scale(loss).backward() if scaler else loss.backward()
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            overall_iou, overall_accuracy, ious, accuracies = calculate_metrics_per_class(outputs, masks, num_classes)
            overall_ious.append(overall_iou)
            overall_accuracies.append(overall_accuracy)
            for cls in range(num_classes):
                class_ious[cls].append(ious[cls])
                class_accuracies[cls].append(accuracies[cls])
            loop.set_postfix(loss=loss.item())
    mean_iou = np.mean(overall_ious).tolist()
    mean_acc = np.mean(overall_accuracies).tolist()
    return epoch_loss / len(loader), mean_iou, mean_acc, [np.mean(i).tolist() for i in class_ious], [np.mean(a).tolist() for a in class_accuracies]

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_iou, train_acc, _, _ = run_epoch(model, train_loader, criterion, 3, optimizer, scaler, True)
    val_loss, val_iou, val_acc, _, _ = run_epoch(model, val_loader, criterion, 3, None, None, False)
    scheduler.step(val_loss)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('IoU/train', train_iou, epoch)
    writer.add_scalar('IoU/validation', val_iou, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/validation', val_acc, epoch)
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved Best Model.")
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break
writer.close()
