import os, time, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.inference_dataset import InferenceDataset
from utils.transforms import get_validation_pipeline
from utils.metrics import calculate_metrics_per_class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

transform = get_validation_pipeline()
image_dir = './test/Original'
mask_dir = './test/Mask'
dataset = InferenceDataset(image_dir, mask_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

model = torch.jit.load('').to(device).eval() #model location

epoch_loss, overall_ious, overall_accuracies = 0.0, [], []
class_ious = [[] for _ in range(3)]
class_accuracies = [[] for _ in range(3)]

start_time = time.time()
with torch.no_grad():
    for images, masks in tqdm(loader, desc="Inference"):
        images, masks = images.to(device), masks.to(device)
        images = images.float() / 255.0
        outputs = model(images)
        outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

        overall_iou, overall_accuracy, ious, accuracies = calculate_metrics_per_class(outputs, masks, 3)
        overall_ious.append(overall_iou)
        overall_accuracies.append(overall_accuracy)
        for cls in range(3):
            class_ious[cls].append(ious[cls])
            class_accuracies[cls].append(accuracies[cls])
end_time = time.time()

print("\nInference Results:")
print(f"Time Taken: {end_time - start_time:.2f} seconds")
print(f"Overall IoU: {np.mean(overall_ious):.4f}")
print(f"Overall Accuracy: {np.mean(overall_accuracies):.4f}")
print(f"Per-Class IoUs: {[np.mean(i) for i in class_ious]}")
print(f"Per-Class Accuracies: {[np.mean(a) for a in class_accuracies]}")
