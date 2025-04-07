import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from inference import InferenceDataset, calculate_metrics, transform


# Paths and DataLoader setup
image_dir = ''  # Update with your path
mask_dir = ''  # Update with your path
batch_size = 64

# Prepare dataset and loader
dataset = InferenceDataset(image_dir, mask_dir, transform=transform)
inference_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Load the traced model
model = torch.jit.load('segformer0-28_traced.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Perform inference and compute metrics
epoch_loss, overall_ious, overall_accuracies = 0.0, [], []
class_ious = [[] for _ in range(3)]  # Number of classes (3 classes in this case)
class_accuracies = [[] for _ in range(3)]  # Number of classes (3 classes in this case)

start_time = time.time()

with torch.no_grad():
    for images, masks in tqdm(inference_loader, desc="Inference", leave=False):
        images, masks = images.to(device), masks.to(device)

        # Ensure images are in the correct format (e.g., normalized between 0 and 1)
        images = images.float() / 255.0  # Normalize to [0, 1] range if needed

        # Ensure that input tensor is of the expected shape (batch_size, 3, 256, 256)
        if len(images.shape) == 3:  # Single image case, add batch dimension
            images = images.unsqueeze(0)

        # Output directly from traced model
        outputs = model(images)  # Output directly from traced model

        # Resize outputs to match the mask shape
        outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

        overall_iou, overall_accuracy, ious, accuracies = calculate_metrics(outputs, masks, 3)
        overall_ious.append(overall_iou)
        overall_accuracies.append(overall_accuracy)
        for cls in range(3):
            class_ious[cls].append(ious[cls])
            class_accuracies[cls].append(accuracies[cls])

end_time = time.time()

mean_overall_iou = np.mean(overall_ious).tolist()
mean_overall_accuracy = np.mean(overall_accuracies).tolist()
mean_class_ious = [np.mean(cls_ious).tolist() for cls_ious in class_ious]
mean_class_accuracies = [np.mean(cls_accs).tolist() for cls_accs in class_accuracies]

time_taken = end_time - start_time

# Print metrics
print("\nInference Results:")
print(f"Time Taken: {time_taken:.2f} seconds")
print(f"Overall IoU: {mean_overall_iou:.4f}")
print(f"Overall Accuracy: {mean_overall_accuracy:.4f}")
print(f"Per-Class IoUs: {mean_class_ious}")
print(f"Per-Class Accuracies: {mean_class_accuracies}")
