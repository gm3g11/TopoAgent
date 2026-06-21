# Benchmark3 Dataset Survey

26 medical image classification datasets (11 MedMNIST + 15 external) used in the Benchmark3 experiment.

## Dataset Table

### MedMNIST Datasets (11)

All MedMNIST datasets are stored as pre-saved 224x224 NPZ files under the MedMNIST path. Only the **train split** is used for cross-validation.

| # | Dataset | Classes | Total | Train | Val | Test | Original Size | Modality | Task |
|---|---------|--------:|------:|------:|----:|-----:|---------------|----------|------|
| 1 | BloodMNIST | 8 | 17,092 | 11,959 | 1,712 | 3,421 | 224x224 (orig 360x363) | Microscopy | Blood cell types (8 subtypes) |
| 2 | BreastMNIST | 2 | 780 | 546 | 78 | 156 | 224x224 (orig 500x500) | Ultrasound | Breast tumor (benign/malignant) |
| 3 | DermaMNIST | 7 | 10,015 | 7,007 | 1,003 | 2,005 | 224x224 (orig 600x450) | Dermatoscopy | Skin lesion (7 types) |
| 4 | OCTMNIST | 4 | 109,309 | 97,477 | 10,832 | 1,000 | 224x224 (orig ~500x700) | OCT | Retinal OCT (4 conditions) |
| 5 | OrganAMNIST | 11 | 58,830 | 34,561 | 6,491 | 17,778 | 224x224 (orig 28x28) | CT (Axial) | Abdominal organs (11 types) |
| 6 | OrganCMNIST | 11 | 23,583 | 12,975 | 2,392 | 8,216 | 224x224 (orig 28x28) | CT (Coronal) | Abdominal organs (11 types) |
| 7 | OrganSMNIST | 11 | 25,211 | 13,932 | 2,452 | 8,827 | 224x224 (orig 28x28) | CT (Sagittal) | Abdominal organs (11 types) |
| 8 | PathMNIST | 9 | 107,180 | 89,996 | 10,004 | 7,180 | 224x224 (orig 28x28) | Microscopy (H&E) | Colorectal cancer tissue (9 types) |
| 9 | PneumoniaMNIST | 2 | 5,856 | 4,708 | 524 | 624 | 224x224 (orig ~1000x1000) | X-ray | Chest pneumonia (normal/pneumonia) |
| 10 | RetinaMNIST | 5 | 1,600 | 1,080 | 120 | 400 | 224x224 (orig ~1700x1700) | Fundus Camera | Diabetic retinopathy grading (5) |
| 11 | TissueMNIST | 8 | 236,386 | 165,466 | 23,640 | 47,280 | 224x224 (orig 32x32) | Microscopy | Kidney cortex cells (8 types) |

### External Datasets (15)

| # | Dataset | Classes | Total | Train | Val | Test | Original Size | Modality | Task |
|---|---------|--------:|------:|------:|----:|-----:|---------------|----------|------|
| 12 | ISIC2019 | 8 | 25,331 | 25,331 | 0 | 0 | ~600x450 | Dermatoscopy | Skin lesion types (8 classes) |
| 13 | Kvasir | 8 | 4,000 | 4,000 | 0 | 0 | ~720x576 | Endoscopy | GI tract findings (8 classes) |
| 14 | BrainTumorMRI | 4 | 7,023 | 5,712 | 0 | 1,311 | ~512x512 | MRI | Brain tumor types (4 classes) |
| 15 | MURA | 14 | 40,005 | 36,808 | 3,197 | 0 | variable | X-ray | Musculoskeletal abnormality (7 body parts x 2) |
| 16 | BreakHis | 8 | 7,909 | 7,909 | 0 | 0 | 700x460 | Microscopy (H&E) | Breast tumor histology (8 subtypes) |
| 17 | NCT_CRC_HE | 9 | 100,000 | 100,000 | 0 | 0 | 224x224 | Microscopy (H&E) | Colorectal cancer tissue (9 types) |
| 18 | MalariaCell | 2 | 27,560 | 27,560 | 0 | 0 | ~130x130 | Microscopy | Malaria cell detection (2 classes) |
| 19 | IDRiD | 5 | 516 | 413 | 0 | 103 | ~4288x2848 | Fundus Camera | Diabetic retinopathy grading (5) |
| 20 | PCam | 2 | 294,912 | 262,144 | 0 | 32,768 | 96x96 | Microscopy (H&E) | Lymph node metastasis (2 classes) |
| 21 | LC25000 | 5 | 25,000 | 25,000 | 0 | 0 | 768x768 | Microscopy (H&E) | Lung & colon histopathology (5 classes) |
| 22 | SIPaKMeD | 5 | 966 | 966 | 0 | 0 | ~2048x1536 | Microscopy (Pap) | Cervical cell types (5 classes) |
| 23 | AML_Cytomorphology | 15 | 18,365 | 18,365 | 0 | 0 | ~400x400 | Microscopy | AML blood cell morphology (15 classes) |
| 24 | APTOS2019 | 5 | 3,294 | 2,929 | 365 | 0 | ~2000x1500 | Fundus Camera | Diabetic retinopathy grading (5) |
| 25 | GasHisSDB | 2 | 33,284 | 33,284 | 0 | 0 | 160x160 | Microscopy (H&E) | Gastric histopathology (2 classes) |
| 26 | Chaoyang | 4 | 6,160 | 4,021 | 0 | 2,139 | variable | Microscopy (H&E) | Colorectal pathology (4 classes) |

## Summary Statistics

**By modality (13 modalities):**

| Modality | Count | Datasets |
|----------|------:|----------|
| Microscopy (H&E) | 7 | PathMNIST, BreakHis, NCT_CRC_HE, PCam, LC25000, GasHisSDB, Chaoyang |
| Microscopy | 4 | BloodMNIST, TissueMNIST, MalariaCell, AML_Cytomorphology |
| Fundus Camera | 3 | RetinaMNIST, IDRiD, APTOS2019 |
| X-ray | 2 | PneumoniaMNIST, MURA |
| Dermatoscopy | 2 | DermaMNIST, ISIC2019 |
| CT (Axial) | 1 | OrganAMNIST |
| CT (Coronal) | 1 | OrganCMNIST |
| CT (Sagittal) | 1 | OrganSMNIST |
| OCT | 1 | OCTMNIST |
| Ultrasound | 1 | BreastMNIST |
| Endoscopy | 1 | Kvasir |
| MRI | 1 | BrainTumorMRI |
| Microscopy (Pap) | 1 | SIPaKMeD |

**By class count:**

| Classes | Count | Datasets |
|--------:|------:|----------|
| 2 | 5 | BreastMNIST, GasHisSDB, MalariaCell, PCam, PneumoniaMNIST |
| 4 | 3 | BrainTumorMRI, Chaoyang, OCTMNIST |
| 5 | 5 | APTOS2019, IDRiD, LC25000, RetinaMNIST, SIPaKMeD |
| 7 | 1 | DermaMNIST |
| 8 | 5 | BloodMNIST, BreakHis, ISIC2019, Kvasir, TissueMNIST |
| 9 | 2 | NCT_CRC_HE, PathMNIST |
| 11 | 3 | OrganAMNIST, OrganCMNIST, OrganSMNIST |
| 14 | 1 | MURA |
| 15 | 1 | AML_Cytomorphology |

**Scale:** 516 (IDRiD, smallest) to 294,912 (PCam, largest). Total across all 26 datasets: ~1.16M images.

## Dataset Paths

All paths are defined in `scripts/run_benchmark3/config.py`.

### MedMNIST (NPZ files)

```
MEDMNIST_PATH = /users/gmeng/afs/.medmnist/

Files:
  bloodmnist_224.npz      breastmnist_224.npz     dermamnist_224.npz
  octmnist_224.npz        organamnist_224.npz     organcmnist_224.npz
  organsmnist_224.npz     pathmnist_224.npz       pneumoniamnist_224.npz
  retinamnist_224.npz     tissuemnist_224.npz
```

Each NPZ contains: `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images`, `test_labels` (224x224 resolution).

### External Datasets

All external datasets live under the shared datasets root unless noted:

```
EXTERNAL_DATASETS_ROOT = /afs/crc/group/ball_lab/gmeng_cl/cl_new/datasets/
```

| Dataset | Path | Directory Structure |
|---------|------|---------------------|
| ISIC2019 | `/users/gmeng/afs/isic2019/train/` | `{AK,BCC,BKL,DF,MEL,NV,SCC,VASC}/*.jpg` |
| Kvasir | `/users/gmeng/afs/kvasir-dataset/` | `{class_name}/*.jpg` (8 folders, 500 each) |
| BrainTumorMRI | `EXTERNAL/data/BrainTumorMRI/Training/` | `{glioma,meningioma,notumor,pituitary}/*.jpg` (Testing/ sibling) |
| MURA | `EXTERNAL/MURA-v1.1/` | `train/XR_{PART}/patient/study/*.png` (7 body parts x pos/neg) |
| BreakHis | `EXTERNAL/ambarish/breakhis/.../breast/` | `{benign,malignant}/SOB/{subtype}/patient/{40X,100X,200X,400X}/*.png` |
| NCT_CRC_HE | `EXTERNAL/nct-crc-he-100k/.../NCT-CRC-HE-100K/` | `{ADI,BACK,DEB,LYM,MUC,MUS,NORM,STR,TUM}/*.tif` |
| MalariaCell | `EXTERNAL/data/MalariaCell/cell_images/` | `{Parasitized,Uninfected}/*.png` |
| IDRiD | `EXTERNAL/idrid/Disease_Grading/` | `1. Original Images/{a. Training Set, b. Testing Set}/*.jpg` + CSV labels in `2. Groundtruths/` |
| PCam | `EXTERNAL/pcam/` | HDF5 files: `camelyonpatch_level_2_split_{train,test}_{x,y}.h5` |
| LC25000 | `EXTERNAL/lc25000/lung_colon_image_set/` | `{colon,lung}_image_sets/{class}/*.jpeg` (5 classes, 5000 each) |
| SIPaKMeD | `EXTERNAL/sipakmed/` | `im_{Class}/im_{Class}/CROPPED/*.bmp` (5 classes, cropped single cells) |
| AML_Cytomorphology | `EXTERNAL/aml_cytomorphology/data/data/` | `{BAS,EBO,EOS,...}/*.tiff` (15 classes, excl. `augmented/`) |
| APTOS2019 | `EXTERNAL/aptos2019/` | `train_images/*.png` + `train_1.csv`, `val_images/*.png` + `valid.csv` |
| GasHisSDB | `EXTERNAL/gashissdb/GasHisSDB/` | `160/{Abnormal,Normal}/*.png` (160x160 sub-database) |
| Chaoyang | `EXTERNAL/chaoyang/` | `chaoyang-data/{train,test}/*.jpg` + `{train,test}.json` labels |

(`EXTERNAL` = `EXTERNAL_DATASETS_ROOT`)

## Unified Data Loader

All 26 datasets are loaded through a single function in `scripts/run_benchmark3/data_loader.py`:

```python
from scripts.run_benchmark3.data_loader import load_dataset

# Load any dataset by name (stratified sampling to n_samples)
images, labels, class_names = load_dataset("DermaMNIST", n_samples=5000, seed=42)
# images: (N, 224, 224) float32 [0, 1] grayscale
# labels: (N,) int
# class_names: list of str
```

### Loader Details by Dataset Type

| Type | Datasets | Loader Method | Notes |
|------|----------|---------------|-------|
| MedMNIST NPZ | 11 MedMNIST | `_load_medmnist()` | Reads from `{name}_224.npz`, uses train split, RGB-to-grayscale conversion |
| Folder (class subfolders) | ISIC2019, Kvasir, BrainTumorMRI, NCT_CRC_HE, MalariaCell | `_load_folder_dataset()` | Scans `root/{class}/*.{jpg,png,...}` |
| MURA (nested studies) | MURA | `_load_mura()` | `train/XR_{PART}/patient/study/*.png`, 14 classes (7 parts x pos/neg) |
| BreakHis (nested subtypes) | BreakHis | `_load_breakhis()` | `{benign,malignant}/SOB/{subtype}/patient/{mag}/*.png`, 8 classes |
| IDRiD (CSV labels) | IDRiD | `_load_idrid()` | CSV label files in `2. Groundtruths/`, images in `1. Original Images/` |
| PCam (HDF5) | PCam | `_load_pcam()` | Reads from `.h5` files (must be decompressed from `.h5.gz` first) |
| LC25000 (two groups) | LC25000 | `_load_lc25000()` | Two parent dirs: `colon_image_sets/`, `lung_image_sets/` |
| SIPaKMeD (cropped cells) | SIPaKMeD | `_load_sipakmed()` | Uses `im_{Class}/im_{Class}/CROPPED/*.bmp` single-cell images |
| AML (class folders) | AML_Cytomorphology | `_load_aml()` | 15 class folders, excludes `augmented/` |
| APTOS (CSV labels) | APTOS2019 | `_load_aptos()` | CSV files (`train_1.csv`, `valid.csv`) + image directories |
| GasHisSDB (subfolder) | GasHisSDB | `_load_gashissdb()` | Uses `160/` sub-database with `{Abnormal,Normal}/` |
| Chaoyang (JSON labels) | Chaoyang | `_load_chaoyang()` | JSON annotations (`train.json`, `test.json`) + image directories |

### Common Preprocessing

All loaders output the same format:
- **Grayscale**: RGB images converted via luminosity (0.299R + 0.587G + 0.114B)
- **Resize**: All images resized to 224x224 (LANCZOS interpolation)
- **Normalize**: Pixel values scaled to [0, 1] float32
- **Stratified sampling**: Proportional class representation when `n_samples` < total

### Quick Test

```bash
# Test a single dataset
python scripts/run_benchmark3/data_loader.py DermaMNIST

# Test all 26 datasets (loads 100 samples each)
python scripts/run_benchmark3/data_loader.py
```
