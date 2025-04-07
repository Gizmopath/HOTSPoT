# HOTSPoT
Hematoxylin & Eosin-based Open-access Tool for Segmentation of Portal Tracts

This repository contains the weighted model and primary resources for deploying a segformer-based semantic segmentation model applied to Hematoxylin and Eosin (H&E)-stained liver biopsy images. This model is designed to automate the division of liver biopsy into portal and lobular zones, acting both as a pre-processing model for second-level tasks, allowing inference only on the desired zone, and to automate the annotation of areas for various types of spatial experiments.

![Study_design](https://github.com/Gizmopath/HOTSPoT/blob/main/images/study.jpg)

## Methods

Whole Slide Images (WSIs) of liver biopsies were collected from multiple liver pathology reference institutions. The workflow included:

- **Anonymization** and **digitization** using various scanners and magnifications.
- Automatic identification of liver tissue using QuPath's custom tissue finder.
- Portal tracts annotated by hepatopathologists, with 256x256-pixel tiles extracted from regions of interest at 1μm/px resolution.

Images and corresponding masks (for background, lobule, and portal areas) were used to fine-tune into a pretrained segformerB0 model (https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512).

---

## Results

### Model Performance

- **Mean Accuracy**:
  - Training Set: **0.96**
  - Validation Set: **0.96**
  - Test Set: **0.96**
- **Mean IoU**:
  - Training Set: **0.86**
  - Validation Set: **0.85**
  - Test Set: **0.84**

### Test Details

The model was tested on 29 cases from six institutions, generating 8,789 **tiles**. Performance metrics for each site are shown below:

| Institution       | Tiles | Diseases         | Inference Time | Accuracy | Mean IoU |
|-------------------|-------|------------------|----------------|----------|----------|
| Barcelona         | 1,186 | AIH             | 7s             | 0.97     | 0.86     |
| Buenos Aires      | 950   | AIH             | 5s             | 0.96     | 0.85     |
| Coimbra           | 1,187 | AIH             | 6s             | 0.98     | 0.88     |
| Monza             | 1,744 | GVHD            | 9s             | 0.97     | 0.84     |
| Maastricht        | 1,107 | AIH, PBC        | 6s             | 0.96     | 0.84     |
| Palermo           | 1,605 | PBC             | 5s             | 0.93     | 0.82     |
| **Full Test Set** | 8,789 | AIH, PBC, GVHD  | 38s            | **0.96** | **0.84** |

---

## Data Details

Images were acquired from various institutions, using scanners with different specifications. The table below summarizes the dataset:

| Institution       | Scanner         | Type | N° | Format | Magnification | Resolution (µm/px) |
|-------------------|-----------------|------|----|--------|---------------|---------------------|
| Monza             | Aperio          | TV   | 71 | .svs   | 20X           | 0.4940              |
| Monza             | Hamamatsu       | TV   | 30 | .ndpi  | 20X           | 0.4416              |
| Monza             | 3DHISTECH       | TV   | 64 | .mrxs  | 67X           | 0.1725              |
| Hannover          | Aperio          | TV   | 64 | .svs   | 40X           | 0.2513              |
| Milan             | Epredia         | TV   | 43 | .mrxs  | 20X           | 0.2425              |
| Barcelona         | Hamamatsu       | Test | 5  | .ndpi  | 40X           | 0.2200              |
| Buenos Aires      | Micro Visioneer | Test | 4  | .tif   | 10X           | 0.5860              |
| Coimbra           | Roche           | Test | 5  | .tiff  | 40X           | 0.2500              |
| Maastricht        | 3DHISTECH       | Test | 5  | .mrxs  | 20X           | 0.2425              |
| Monza             | Leica           | Test | 5  | .svs   | 40X           | 0.2621              |
| Palermo           | Philips         | Test | 5  | .tiff  | 40X           | 0.2500              |

---

![Inference](https://github.com/Gizmopath/HOTSPoT/blob/main/images/results.jpg)

## Model Training

1. **Data Preparation**: 
   - Organize the dataset into the following structure:
     ```
     data/
     ├── train/
     │   ├── images/  # Training images
     │   └── masks/   # Training masks
     └── val/
         ├── images/  # Validation images
         └── masks/   # Validation masks
     ```

2. **Run the Training Script**:
   Execute `train.py`:
   ```bash
   python train.py

## Model Inference
Execute 'infer.py' after specifying the patch directories:
```bash
python infer.py

## Acknowledgements
The pipeline is designed for deployment using a modified version of the WSInfer library, enabling efficient WSI-level inference with models in TorchScript format.
Please refer to this repository: https://github.com/Vsc0/nutshell

## Reference
If you find our work useful, please consider citing:

## License
Academic Free License v3.0.
