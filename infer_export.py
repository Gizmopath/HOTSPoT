import os, json, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.inference_dataset import InferenceDataset
from utils.transforms import get_validation_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

image_dir = './inference_images'
output_mask_dir = './outputs/predicted_masks'
output_json_dir = './outputs/predictions_json'
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

transform = get_validation_pipeline()
loader = DataLoader(InferenceDataset(image_dir, transform=transform), batch_size=64, shuffle=False, num_workers=4)

model = torch.jit.load('').to(device).eval()  #model location

with torch.no_grad():
    for images, filenames in tqdm(loader, desc="Inference"):
        images = images.to(device).float() / 255.0
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        for mask_arr, fname in zip(preds, filenames):
            out_mask = Image.fromarray((mask_arr.astype(np.uint8) * 127))
            out_mask.save(os.path.join(output_mask_dir, fname))

            with open(os.path.join(output_json_dir, fname.replace('.png', '.json')), 'w') as f:
                json.dump(mask_arr.tolist(), f)
