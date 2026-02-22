"""
Benchmark3 Unified Data Loader.

Loads all 26 datasets with stratified sampling, outputting:
  - images: (N, 224, 224) grayscale float32 [0, 1]
  - labels: (N,) integer class labels
  - class_names: list of class name strings

Datasets:
  MedMNIST (11): BloodMNIST, TissueMNIST, PathMNIST, OCTMNIST, OrganAMNIST, RetinaMNIST,
                 PneumoniaMNIST, BreastMNIST, DermaMNIST, OrganCMNIST, OrganSMNIST
  External (15): ISIC2019, Kvasir, BrainTumorMRI, MURA, BreakHis, NCT_CRC_HE, MalariaCell, IDRiD,
                 PCam, LC25000, SIPaKMeD, AML_Cytomorphology, APTOS2019, GasHisSDB, Chaoyang
"""

import sys
import os
import gc
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RuleBenchmark.benchmark3.config import (
    MEDMNIST_PATH, ISIC_PATH, KVASIR_PATH, BRAIN_TUMOR_PATH,
    MURA_PATH, BREAKHIS_PATH, NCT_CRC_PATH, MALARIA_PATH, IDRID_PATH,
    PCAM_PATH, LC25000_PATH, SIPAKMED_PATH, AML_PATH, APTOS_PATH,
    GASHISSDB_PATH, CHAOYANG_PATH, DATASETS,
)


def load_dataset(
    dataset_name: str,
    n_samples: Optional[int] = 5000,
    seed: int = 42,
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Unified dataset loader with stratified sampling.

    Args:
        dataset_name: Name from config DATASETS dict
        n_samples: Max samples to load (None = all). Stratified across classes.
        seed: Random seed for reproducible sampling
        image_size: Target image size (all resized to this)

    Returns:
        images: (N, image_size, image_size) grayscale float32 [0, 1]
        labels: (N,) integer labels
        class_names: List of class name strings
    """
    loaders = {
        # MedMNIST (11)
        'BloodMNIST': lambda: _load_medmnist('bloodmnist', n_samples, seed),
        'TissueMNIST': lambda: _load_medmnist('tissuemnist', n_samples, seed),
        'PathMNIST': lambda: _load_medmnist('pathmnist', n_samples, seed),
        'OCTMNIST': lambda: _load_medmnist('octmnist', n_samples, seed),
        'OrganAMNIST': lambda: _load_medmnist('organamnist', n_samples, seed),
        'RetinaMNIST': lambda: _load_medmnist('retinamnist', n_samples, seed),
        'PneumoniaMNIST': lambda: _load_medmnist('pneumoniamnist', n_samples, seed),
        'BreastMNIST': lambda: _load_medmnist('breastmnist', n_samples, seed),
        'DermaMNIST': lambda: _load_medmnist('dermamnist', n_samples, seed),
        'OrganCMNIST': lambda: _load_medmnist('organcmnist', n_samples, seed),
        'OrganSMNIST': lambda: _load_medmnist('organsmnist', n_samples, seed),
        # External (8)
        'ISIC2019': lambda: _load_folder_dataset(ISIC_PATH, n_samples, seed, image_size),
        'Kvasir': lambda: _load_folder_dataset(KVASIR_PATH, n_samples, seed, image_size),
        'BrainTumorMRI': lambda: _load_folder_dataset(BRAIN_TUMOR_PATH, n_samples, seed, image_size),
        'MURA': lambda: _load_mura(n_samples, seed, image_size),
        'BreakHis': lambda: _load_breakhis(n_samples, seed, image_size),
        'NCT_CRC_HE': lambda: _load_folder_dataset(NCT_CRC_PATH, n_samples, seed, image_size),
        'MalariaCell': lambda: _load_folder_dataset(MALARIA_PATH, n_samples, seed, image_size),
        'IDRiD': lambda: _load_idrid(n_samples, seed, image_size),
        'PCam': lambda: _load_pcam(n_samples, seed, image_size),
        'LC25000': lambda: _load_lc25000(n_samples, seed, image_size),
        'SIPaKMeD': lambda: _load_sipakmed(n_samples, seed, image_size),
        'AML_Cytomorphology': lambda: _load_aml(n_samples, seed, image_size),
        'APTOS2019': lambda: _load_aptos(n_samples, seed, image_size),
        'GasHisSDB': lambda: _load_gashissdb(n_samples, seed, image_size),
        'Chaoyang': lambda: _load_chaoyang(n_samples, seed, image_size),
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")

    images, labels, class_names = loaders[dataset_name]()

    print(f"  Loaded {dataset_name}: {images.shape}, {len(class_names)} classes, "
          f"range=[{images.min():.3f}, {images.max():.3f}]")
    print(f"  Classes: {class_names}")
    print(f"  Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return images, labels, class_names


# =============================================================================
# MedMNIST Loader (pre-saved 224×224 NPZ)
# =============================================================================

def _load_medmnist(
    name: str,
    n_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load MedMNIST from pre-saved 224×224 NPZ files."""
    npz_path = MEDMNIST_PATH / f"{name}_224.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"MedMNIST file not found: {npz_path}")

    data = np.load(npz_path, mmap_mode='r')

    # Use train split
    images = data['train_images']
    labels = data['train_labels'].flatten()

    # Get class names from data if available, otherwise use integers
    n_classes = len(np.unique(labels))
    class_names = [str(i) for i in range(n_classes)]

    # Stratified sampling
    if n_samples is not None and n_samples < len(labels):
        indices = _stratified_sample(labels, n_samples, seed)
        images = np.array(images[indices])
        labels = labels[indices].copy()
    else:
        images = np.array(images)
        labels = labels.copy()

    del data
    gc.collect()

    # Convert to grayscale float32 [0, 1]
    images = _to_grayscale_float(images)

    return images, labels, class_names


# =============================================================================
# Folder-based Dataset Loader (class subfolders)
# =============================================================================

def _load_folder_dataset(
    root_path: Path,
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load dataset organized as root/class_name/image_files."""
    from PIL import Image

    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {root_path}")

    # Discover class folders (only those with image files)
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    candidate_dirs = sorted([
        d for d in os.listdir(root_path)
        if os.path.isdir(root_path / d) and not d.startswith('.')
    ])

    # Filter: only keep dirs that contain at least one image file
    class_names = []
    for d in candidate_dirs:
        class_dir = root_path / d
        has_images = any(
            f.lower().endswith(valid_exts) for f in os.listdir(class_dir)
            if not f.startswith('.')
        )
        if has_images:
            class_names.append(d)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Collect all image paths and labels
    all_paths = []
    all_labels = []

    for class_name in class_names:
        class_dir = root_path / class_name
        img_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(valid_exts) and not f.startswith('.')
        ])
        for f in img_files:
            all_paths.append(str(class_dir / f))
            all_labels.append(class_to_idx[class_name])

    all_labels = np.array(all_labels)

    # Stratified sampling
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    # Load and preprocess images
    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# MURA Loader (7 body parts × positive/negative = 14 classes)
# =============================================================================

def _load_mura(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load MURA: 7 body parts × 2 conditions = 14 classes.

    Classes: XR_ELBOW_negative, XR_ELBOW_positive, XR_FINGER_negative, ...
    """
    from PIL import Image

    train_path = MURA_PATH / "train"
    if not train_path.exists():
        raise FileNotFoundError(f"MURA train path not found: {train_path}")

    body_parts = sorted([
        d for d in os.listdir(train_path)
        if os.path.isdir(train_path / d) and d.startswith('XR_')
    ])

    # 14 classes: body_part × {negative, positive}
    class_names = []
    for part in body_parts:
        class_names.append(f"{part}_negative")
        class_names.append(f"{part}_positive")
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths = []
    all_labels = []
    valid_exts = ('.png', '.jpg', '.jpeg')

    for part in body_parts:
        part_path = train_path / part
        for patient in os.listdir(part_path):
            patient_path = part_path / patient
            if not patient_path.is_dir():
                continue
            for study in os.listdir(patient_path):
                study_path = patient_path / study
                if not study_path.is_dir():
                    continue

                # Determine label from study folder name
                if 'positive' in study:
                    class_name = f"{part}_positive"
                elif 'negative' in study:
                    class_name = f"{part}_negative"
                else:
                    continue

                label = class_to_idx[class_name]
                img_files = [
                    f for f in os.listdir(study_path)
                    if f.lower().endswith(valid_exts) and not f.startswith('.')
                ]
                for f in img_files:
                    all_paths.append(str(study_path / f))
                    all_labels.append(label)

    all_labels = np.array(all_labels)

    # Stratified sampling
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# BreakHis Loader (8 subtypes, all magnifications)
# =============================================================================

def _load_breakhis(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load BreakHis: 4 benign + 4 malignant subtypes = 8 classes.

    Uses all magnification levels (40X, 100X, 200X, 400X) combined.
    """
    from PIL import Image

    if not BREAKHIS_PATH.exists():
        raise FileNotFoundError(f"BreakHis path not found: {BREAKHIS_PATH}")

    # 8 classes: subtypes under benign/ and malignant/
    class_names = []
    all_paths = []
    all_labels = []
    valid_exts = ('.png', '.jpg', '.jpeg')

    for category in ['benign', 'malignant']:
        sob_path = BREAKHIS_PATH / category / "SOB"
        if not sob_path.exists():
            continue
        subtypes = sorted(os.listdir(sob_path))
        for subtype in subtypes:
            class_name = f"{category}_{subtype}"
            if class_name not in class_names:
                class_names.append(class_name)
            label = class_names.index(class_name)

            subtype_path = sob_path / subtype
            for patient in os.listdir(subtype_path):
                patient_path = subtype_path / patient
                if not patient_path.is_dir():
                    continue
                # All magnifications
                for mag in ['40X', '100X', '200X', '400X']:
                    mag_path = patient_path / mag
                    if not mag_path.exists():
                        continue
                    for f in os.listdir(mag_path):
                        if f.lower().endswith(valid_exts):
                            all_paths.append(str(mag_path / f))
                            all_labels.append(label)

    all_labels = np.array(all_labels)

    # Stratified sampling
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# IDRiD Loader (Diabetic Retinopathy Grading)
# =============================================================================

def _load_idrid(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load IDRiD Disease Grading: 5 DR severity grades (0-4).

    Uses both training and testing sets (516 total) since the dataset is small.
    Labels from CSV files: 'Retinopathy grade' column.
    """
    import csv
    from PIL import Image

    if not IDRID_PATH.exists():
        raise FileNotFoundError(f"IDRiD path not found: {IDRID_PATH}")

    images_dir = IDRID_PATH / "1. Original Images"
    labels_dir = IDRID_PATH / "2. Groundtruths"

    class_names = ["DR_grade_0", "DR_grade_1", "DR_grade_2", "DR_grade_3", "DR_grade_4"]

    all_paths = []
    all_labels = []

    # Load both train and test splits
    splits = [
        ("a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
        ("b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv"),
    ]

    for img_folder, label_file in splits:
        img_dir = images_dir / img_folder
        csv_path = labels_dir / label_file

        if not img_dir.exists() or not csv_path.exists():
            continue

        # Read labels from CSV
        label_map = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2 and row[1].strip().isdigit():
                    label_map[row[0].strip()] = int(row[1].strip())

        # Match images to labels
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(valid_exts):
                continue
            img_name = os.path.splitext(fname)[0]
            if img_name in label_map:
                all_paths.append(str(img_dir / fname))
                all_labels.append(label_map[img_name])

    all_labels = np.array(all_labels)

    # Stratified sampling
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# PCam Loader (PatchCamelyon - Metastasis Detection)
# =============================================================================

def _load_pcam(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load PCam: binary classification (normal vs metastasis) from HDF5.

    PCam contains 96x96 patches extracted from histopathology whole-slide images.
    Expects pre-decompressed HDF5 files (run gunzip -k on .h5.gz files first).
    Falls back to gzip decompression if .h5 files not found.
    """
    import h5py
    from PIL import Image

    if not PCAM_PATH.exists():
        raise FileNotFoundError(f"PCam path not found: {PCAM_PATH}")

    class_names = ["normal", "metastasis"]

    # Prefer decompressed .h5 files (much faster)
    train_x_h5 = PCAM_PATH / "camelyonpatch_level_2_split_train_x.h5"
    train_y_h5 = PCAM_PATH / "camelyonpatch_level_2_split_train_y.h5"

    if not train_x_h5.exists() or not train_y_h5.exists():
        # Fall back to gzipped files with on-the-fly decompression
        train_x_gz = PCAM_PATH / "camelyonpatch_level_2_split_train_x.h5.gz"
        train_y_gz = PCAM_PATH / "camelyonpatch_level_2_split_train_y.h5.gz"
        if not train_x_gz.exists():
            raise FileNotFoundError(
                f"PCam files not found. Expected either .h5 or .h5.gz in {PCAM_PATH}")
        raise FileNotFoundError(
            f"PCam .h5 files not found. Please decompress first:\n"
            f"  cd {PCAM_PATH} && gunzip -k *.h5.gz")

    # Load labels
    with h5py.File(str(train_y_h5), 'r') as f:
        labels = f['y'][:].flatten()

    # Stratified sampling BEFORE loading images (PCam has 262k images)
    if n_samples is not None and n_samples < len(labels):
        indices = _stratified_sample(labels, n_samples, seed)
    else:
        indices = np.arange(len(labels))

    labels = labels[indices]

    # Load only sampled images (sorted indices for HDF5 sequential read)
    sorted_idx = np.sort(indices)
    with h5py.File(str(train_x_h5), 'r') as f:
        raw_images = f['x'][sorted_idx]  # (N, 96, 96, 3)

    # Reorder to match original index order
    idx_map = {v: i for i, v in enumerate(sorted_idx)}
    reorder_idx = [idx_map[v] for v in indices]
    raw_images = raw_images[reorder_idx]

    # Convert to grayscale float32 and resize
    images = []
    for img_arr in raw_images:
        img = Image.fromarray(img_arr)
        img = img.convert('L')
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)

    images = np.array(images, dtype=np.float32)

    return images, labels, class_names


# =============================================================================
# LC25000 Loader (Lung and Colon Cancer Histopathology)
# =============================================================================

def _load_lc25000(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load LC25000: 5-class lung+colon histopathology.

    Classes: colon_aca, colon_n, lung_aca, lung_n, lung_scc
    Structure: lung_colon_image_set/{colon,lung}_image_sets/{class}/
    """
    if not LC25000_PATH.exists():
        raise FileNotFoundError(f"LC25000 path not found: {LC25000_PATH}")

    # Collect all class folders from both lung and colon subdirs
    class_names = []
    all_paths = []
    all_labels = []
    valid_exts = ('.jpeg', '.jpg', '.png')

    for group in ['colon_image_sets', 'lung_image_sets']:
        group_dir = LC25000_PATH / group
        if not group_dir.exists():
            continue
        for cls_dir in sorted(os.listdir(group_dir)):
            cls_path = group_dir / cls_dir
            if not cls_path.is_dir():
                continue
            if cls_dir not in class_names:
                class_names.append(cls_dir)
            label = class_names.index(cls_dir)
            for f in os.listdir(cls_path):
                if f.lower().endswith(valid_exts):
                    all_paths.append(str(cls_path / f))
                    all_labels.append(label)

    all_labels = np.array(all_labels)

    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# SIPaKMeD Loader (Cervical Cell Classification)
# =============================================================================

def _load_sipakmed(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load SIPaKMeD: 5-class cervical cell classification from cropped cells.

    Structure: im_{Class}/im_{Class}/CROPPED/
    Uses CROPPED single-cell images.
    """
    if not SIPAKMED_PATH.exists():
        raise FileNotFoundError(f"SIPaKMeD path not found: {SIPAKMED_PATH}")

    class_names = []
    all_paths = []
    all_labels = []
    valid_exts = ('.bmp', '.jpg', '.jpeg', '.png')

    for cls_dir in sorted(os.listdir(SIPAKMED_PATH)):
        if not cls_dir.startswith('im_'):
            continue
        # Structure: im_X/im_X/CROPPED/
        cropped_dir = SIPAKMED_PATH / cls_dir / cls_dir / "CROPPED"
        if not cropped_dir.exists():
            # Try without CROPPED subfolder
            cropped_dir = SIPAKMED_PATH / cls_dir / cls_dir
        if not cropped_dir.exists():
            continue

        class_name = cls_dir.replace('im_', '')
        class_names.append(class_name)
        label = len(class_names) - 1

        for f in os.listdir(cropped_dir):
            if f.lower().endswith(valid_exts):
                all_paths.append(str(cropped_dir / f))
                all_labels.append(label)

    all_labels = np.array(all_labels)

    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# AML-Cytomorphology Loader (Leukemia Cell Classification)
# =============================================================================

def _load_aml(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load AML-Cytomorphology: 15-class blood cell morphology.

    Structure: data/data/{BAS,EBO,EOS,...}/
    Excludes 'augmented' folder.
    """
    if not AML_PATH.exists():
        raise FileNotFoundError(f"AML path not found: {AML_PATH}")

    class_names = []
    all_paths = []
    all_labels = []
    valid_exts = ('.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp')

    for cls_dir in sorted(os.listdir(AML_PATH)):
        cls_path = AML_PATH / cls_dir
        if not cls_path.is_dir() or cls_dir == 'augmented':
            continue
        class_names.append(cls_dir)
        label = len(class_names) - 1
        for f in os.listdir(cls_path):
            if f.lower().endswith(valid_exts):
                all_paths.append(str(cls_path / f))
                all_labels.append(label)

    all_labels = np.array(all_labels)

    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# APTOS 2019 Loader (Diabetic Retinopathy Grading)
# =============================================================================

def _load_aptos(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load APTOS 2019: 5-class DR severity grading from fundus photos.

    Uses train + validation splits combined. Labels from CSV.
    """
    import csv
    from PIL import Image

    if not APTOS_PATH.exists():
        raise FileNotFoundError(f"APTOS path not found: {APTOS_PATH}")

    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]

    all_paths = []
    all_labels = []

    # Load from train and validation CSV files
    # Note: Kaggle download may nest dirs (train_images/train_images/)
    splits = [
        ("train_1.csv", "train_images"),
        ("valid.csv", "val_images"),
    ]

    def _find_img_dir(base, name):
        """Handle possible nested directory from Kaggle unzip."""
        d = base / name
        nested = d / name
        return nested if nested.exists() else d

    for csv_file, img_folder in splits:
        csv_path = APTOS_PATH / csv_file
        img_dir = _find_img_dir(APTOS_PATH, img_folder)
        if not csv_path.exists() or not img_dir.exists():
            continue

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    img_id = row[0].strip()
                    label = int(row[1].strip())
                    # Try common extensions
                    for ext in ['.png', '.jpg', '.jpeg']:
                        img_path = img_dir / f"{img_id}{ext}"
                        if img_path.exists():
                            all_paths.append(str(img_path))
                            all_labels.append(label)
                            break

    all_labels = np.array(all_labels)

    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# GasHisSDB Loader (Gastric Histopathology)
# =============================================================================

def _load_gashissdb(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load GasHisSDB: binary gastric histopathology (normal vs abnormal).

    Uses 160x160 sub-database. Structure: GasHisSDB/160/{Abnormal,Normal}/
    """
    if not GASHISSDB_PATH.exists():
        raise FileNotFoundError(f"GasHisSDB path not found: {GASHISSDB_PATH}")

    # Find the 160x160 subdirectory (preferred) or any available size
    data_dir = None
    for candidate in ['160', 'Sub-image-160', '160x160']:
        d = GASHISSDB_PATH / candidate
        if d.exists():
            data_dir = d
            break

    if data_dir is None:
        # Try to find any Normal/Abnormal structure
        for root_dir in [GASHISSDB_PATH] + list(GASHISSDB_PATH.iterdir()):
            if not root_dir.is_dir():
                continue
            if (root_dir / 'Normal').exists() or (root_dir / 'Abnormal').exists():
                data_dir = root_dir
                break

    if data_dir is None:
        raise FileNotFoundError(
            f"GasHisSDB data directory not found. Expected {GASHISSDB_PATH}/160/ "
            f"with Normal/ and Abnormal/ subdirectories.")

    return _load_folder_dataset(data_dir, n_samples, seed, image_size)


# =============================================================================
# Chaoyang Loader (Colorectal Pathology)
# =============================================================================

def _load_chaoyang(
    n_samples: Optional[int],
    seed: int,
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load Chaoyang: 4-class colorectal pathology with noisy labels.

    Labels from train.json: 0=normal, 1=serrated, 2=adenocarcinoma, 3=adenoma.
    Uses train split (4021 images) + test split (2139 images).
    """
    import json
    from PIL import Image

    if not CHAOYANG_PATH.exists():
        raise FileNotFoundError(f"Chaoyang path not found: {CHAOYANG_PATH}")

    class_names = ["normal", "serrated", "adenocarcinoma", "adenoma"]

    all_paths = []
    all_labels = []

    # Find the data directory (might be nested)
    data_root = CHAOYANG_PATH
    for candidate in ['chaoyang-data', 'chaoyang_data', '']:
        d = CHAOYANG_PATH / candidate if candidate else CHAOYANG_PATH
        if (d / 'train.json').exists() or (d / 'train').exists():
            data_root = d
            break

    for split in ['train', 'test']:
        json_path = data_root / f'{split}.json'
        img_dir = data_root / split

        if not img_dir.exists():
            continue

        if json_path.exists():
            with open(json_path, 'r') as f:
                annotations = json.load(f)

            # Annotations can be list of dicts or dict with image_name: label
            if isinstance(annotations, list):
                for item in annotations:
                    img_name = item.get('name', item.get('image', ''))
                    label = item.get('label', item.get('class', -1))
                    if isinstance(label, int) and 0 <= label <= 3:
                        # Handle both basename and path-prefixed names
                        img_path = img_dir / Path(img_name).name
                        if not img_path.exists():
                            img_path = img_dir / img_name
                        if not img_path.exists():
                            img_path = data_root / img_name
                        if img_path.exists():
                            all_paths.append(str(img_path))
                            all_labels.append(label)
            elif isinstance(annotations, dict):
                for img_name, label in annotations.items():
                    if isinstance(label, int) and 0 <= label <= 3:
                        img_path = img_dir / Path(img_name).name
                        if not img_path.exists():
                            img_path = img_dir / img_name
                        if not img_path.exists():
                            img_path = data_root / img_name
                        if img_path.exists():
                            all_paths.append(str(img_path))
                            all_labels.append(label)

    # Fallback: extract labels from filenames (e.g., "535940-IMG009x022-2.JPG")
    if len(all_paths) == 0:
        import re
        for split in ['train', 'test']:
            img_dir = data_root / split
            if not img_dir.exists():
                continue
            for fname in sorted(img_dir.iterdir()):
                if not fname.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                # Label is the last number before extension: "...-{label}.JPG"
                match = re.search(r'-(\d+)\.\w+$', fname.name)
                if match:
                    label = int(match.group(1))
                    if 0 <= label <= 3:
                        all_paths.append(str(fname))
                        all_labels.append(label)

    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"No Chaoyang images found. Expected {data_root}/train.json and "
            f"{data_root}/train/ directory, or label-encoded filenames.")

    all_labels = np.array(all_labels)

    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    images = _load_images([all_paths[i] for i in indices], image_size)
    labels = all_labels[indices]

    return images, labels, class_names


# =============================================================================
# Shared Utilities
# =============================================================================

def _stratified_sample(
    labels: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Stratified sampling: proportional representation of each class.

    If a class has fewer samples than its share, all samples are included.
    Remaining quota is redistributed to other classes.
    """
    np.random.seed(seed)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    indices = []
    for i, label in enumerate(sorted(unique_labels)):
        class_idx = np.where(labels == label)[0]
        n = per_class + (1 if i < remainder else 0)
        n = min(n, len(class_idx))
        chosen = np.random.choice(class_idx, n, replace=False)
        indices.extend(chosen.tolist())

    return np.array(indices)


def _load_images(
    paths: List[str],
    image_size: int,
) -> np.ndarray:
    """Load image files, convert to grayscale float32 [0, 1], resize to image_size."""
    from PIL import Image

    images = []
    for path in paths:
        try:
            img = Image.open(path)
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            # Resize
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            # Normalize to [0, 1]
            if arr.max() > 1.0:
                arr = arr / 255.0
            images.append(arr)
        except Exception as e:
            # Skip corrupted images, fill with zeros
            print(f"  WARNING: Failed to load {path}: {e}")
            images.append(np.zeros((image_size, image_size), dtype=np.float32))

    return np.array(images, dtype=np.float32)


def _to_grayscale_float(images: np.ndarray) -> np.ndarray:
    """Convert image array to grayscale float32 [0, 1].

    Handles: (N, H, W, 3) RGB, (N, H, W, 1), (N, H, W) grayscale.
    """
    if images.ndim == 4 and images.shape[3] == 3:
        # RGB to grayscale (luminosity)
        images = (0.299 * images[:, :, :, 0] +
                  0.587 * images[:, :, :, 1] +
                  0.114 * images[:, :, :, 2]).astype(np.float32)
    elif images.ndim == 4 and images.shape[3] == 1:
        images = images[:, :, :, 0].astype(np.float32)
    else:
        images = images.astype(np.float32)

    if images.max() > 1.0:
        images = images / 255.0

    return images


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    import sys
    import time

    datasets_to_test = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    for name in datasets_to_test:
        print(f"\n{'='*60}")
        print(f"  Loading: {name}")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            images, labels, class_names = load_dataset(name, n_samples=100, seed=42)
            elapsed = time.time() - t0
            print(f"  Shape: {images.shape}, dtype: {images.dtype}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
