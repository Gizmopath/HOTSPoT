# <img src="https://github.com/Gizmopath/HOTSPoT/blob/main/images/icon.png" alt="ICON" style="width:-80%;"> **Hematoxylin & Eosin-based Open-access Tool for Segmentation of Portal Tracts**

HOTSPoT is a semantic segmentation tool based on **Segformer**, designed to automatically identify **portal tracts** in Hematoxylin and Eosin (H&E)-stained liver biopsies.  
It can be used both as a **pre-processing module** for downstream tasks (e.g., analysis on portal-only regions) and as an **annotation assistant** in spatial tissue experiments.

![APPLICATIONS](https://github.com/Gizmopath/HOTSPoT/blob/main/images/Figure%206.jpg)
---

## 🧪 Methods

![COVER](https://github.com/Gizmopath/HOTSPoT/blob/main/images/study.jpg)

- Whole Slide Images (WSIs) collected from liver pathology reference institutions.
- WSIs were anonymized and digitized using various scanners.
- Portal tracts manually annotated by expert hepatopathologists.
- 256×256 pixel tiles extracted from annotated regions at 1μm/px resolution.
- Fine-tuning performed on the [pretrained SegformerB0 model](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512).

## 🧬 Dataset Summary

| Institution       | Scanner         | Type | N°  | Format | Magnification | Resolution (µm/px) |
|-------------------|-----------------|------|-----|--------|----------------|---------------------|
| Monza             | Aperio          | TV   | 71  | .svs   | 20X            | 0.4940              |
| Monza             | Hamamatsu       | TV   | 30  | .ndpi  | 20X            | 0.4416              |
| Monza             | 3DHISTECH       | TV   | 64  | .mrxs  | 67X            | 0.1725              |
| Hannover          | Aperio          | TV   | 64  | .svs   | 40X            | 0.2513              |
| Milan             | Epredia         | TV   | 43  | .mrxs  | 20X            | 0.2425              |
| Barcelona         | Hamamatsu       | Test | 5   | .ndpi  | 40X            | 0.2200              |
| Buenos Aires      | Micro Visioneer | Test | 4   | .tif   | 10X            | 0.5860              |
| Coimbra           | Roche           | Test | 5   | .tiff  | 40X            | 0.2500              |
| Maastricht        | 3DHISTECH       | Test | 5   | .mrxs  | 20X            | 0.2425              |
| Monza             | Leica           | Test | 5   | .svs   | 40X            | 0.2621              |
| Palermo           | Philips         | Test | 5   | .tiff  | 40X            | 0.2500              |

---

## 📈 Results

**Model performance summary:**

| Dataset    | Accuracy | Mean IoU |
|------------|----------|----------|
| Training   | 0.96     | 0.86     |
| Validation | 0.96     | 0.85     |
| Test       | 0.96     | 0.84     |

**Site-wise evaluation (8,789 total tiles):**

| Institution       | Tiles | Diseases        | Inference Time | Accuracy | Mean IoU |
|-------------------|-------|------------------|----------------|----------|----------|
| Barcelona         | 1,186 | AIH             | 7s             | 0.97     | 0.86     |
| Buenos Aires      | 950   | AIH             | 5s             | 0.96     | 0.85     |
| Coimbra           | 1,187 | AIH             | 6s             | 0.98     | 0.88     |
| Monza             | 1,744 | GVHD            | 9s             | 0.97     | 0.84     |
| Maastricht        | 1,107 | AIH, PBC        | 6s             | 0.96     | 0.84     |
| Palermo           | 1,605 | PBC             | 5s             | 0.93     | 0.82     |
| **Full Test Set** | 8,789 | AIH, PBC, GVHD  | 38s            | **0.96** | **0.84** |

![Inference Results](https://github.com/Gizmopath/HOTSPoT/blob/main/images/results.jpg)

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

## 🏋️ Training

```bash
python train.py
```

## 🔍 Inference with metrics

```bash
python infer_with_metrics.py
```

## 📤 Inference (export only)

```bash
python infer_export.py
```

---

## 🚀 QuPath Deployment via WSInfer

For fast WSI-level inference and integration with **QuPath**, use the modified [WSInfer](https://github.com/Vsc0/nutshell) library.  
TorchScript models from this repo are compatible for direct deployment.

---

## 📜 License

This repository is released under the **Limited Use License for Scientific Research**.

```
Limited Use License for Scientific Research

Preamble
The rights holder (hereinafter referred to as the "Affirmer") hereby makes available the Work under the following conditions, granting usage rights solely for scientific research purposes and excluding any commercial or clinical applications.

1. Granted Rights
The Affirmer grants users the right to use, reproduce, adapt, distribute, communicate, and modify the Work exclusively for scientific research activities, including experimentation, academic publication, teaching, and non-commercial dissemination.

2. Usage Restrictions

No Commercial Use: Any use of the Work aimed at direct or indirect profit, including use in products, services, advertising, or any other commercial activity, is expressly prohibited.

No Clinical Use: The Work shall not be used for diagnosis, medical treatment, clinical procedures, clinical trials, or any direct clinical application involving patients.

3. Citation Requirement
Any use of the Work for research purposes must include proper citation of the original Work and/or the repository from which it was obtained, in accordance with academic standards.

4. Term and Territory
This license applies worldwide for the maximum duration of copyright and related rights as provided by applicable law.

5. Disclaimer of Warranty and Liability
The Affirmer provides the Work "as-is" without any warranties of any kind and disclaims any liability for improper or unauthorized use, as well as for any third-party rights that may apply.

6. Compliance with Laws
Users are responsible for complying with all applicable laws regarding the use of the Work, including third-party rights and informed consent where relevant.
```

---

## 📚 Citation
If you use the HOTSPoT model in your work or integrate it into your project via GitHub, please cite the following publication:
Cazzaniga G. Automating liver biopsy segmentation with a robust, open-source tool for pathology research: the HOTSPoT model. npj Digital Medicine. 2025;8:455. https://doi.org/10.1038/s41746-025-01870-1
