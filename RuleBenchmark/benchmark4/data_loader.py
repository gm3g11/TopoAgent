"""
Benchmark4 Data Loader.

Extends benchmark3 data_loader with RGB support for per-channel PH.

Returns:
  Grayscale: (N, H, W) float32 [0, 1]
  RGB:       (N, H, W, 3) float32 [0, 1]
  + labels:  (N,) integer
  + class_names: list[str]
"""

import sys
import os
import gc
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RuleBenchmark.benchmark4.config import (
    MEDMNIST_PATH, ISIC_PATH, KVASIR_PATH, BRAIN_TUMOR_PATH,
    MURA_PATH, BREAKHIS_PATH, NCT_CRC_PATH, MALARIA_PATH, IDRID_PATH,
    PCAM_PATH, LC25000_PATH, SIPAKMED_PATH, AML_PATH, APTOS_PATH,
    GASHISSDB_PATH, CHAOYANG_PATH, DATASETS, PH_CONFIG,
)

# Reuse benchmark3 loader internals (but NOT _stratified_sample — we override it below)
from RuleBenchmark.benchmark3.data_loader import (
    _load_mura, _load_breakhis, _load_idrid, _load_pcam,
    _load_lc25000, _load_sipakmed, _load_aml, _load_aptos,
    _load_gashissdb, _load_chaoyang,
)


def _stratified_sample(
    labels: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Proportional stratified sampling preserving class distribution.

    Each class gets floor(n_samples * class_fraction) samples.
    Remaining slots allocated by largest fractional remainder.
    If a class has fewer samples than its quota, take all and redistribute.

    Returns array of selected indices.
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    n_total = len(labels)

    # Build per-class index pools
    class_indices = {}
    for label in unique_labels:
        class_indices[label] = np.where(labels == label)[0]

    # Compute proportional quotas
    raw_quotas = {label: n_samples * len(class_indices[label]) / n_total
                  for label in unique_labels}

    # Floor quotas, capped by available
    quotas = {label: min(int(np.floor(raw_quotas[label])),
                         len(class_indices[label]))
              for label in unique_labels}

    # Distribute remaining slots by largest fractional remainder
    remaining = n_samples - sum(quotas.values())
    if remaining > 0:
        remainders = {label: raw_quotas[label] - quotas[label]
                      for label in unique_labels}
        # Sort by remainder descending; break ties by label for reproducibility
        sorted_labels = sorted(remainders.keys(),
                               key=lambda l: (-remainders[l], l))
        for label in sorted_labels:
            if remaining <= 0:
                break
            if quotas[label] < len(class_indices[label]):
                quotas[label] += 1
                remaining -= 1

    # Sample from each class
    indices = []
    for label in sorted(unique_labels):
        n = quotas[label]
        if n > 0:
            chosen = rng.choice(class_indices[label], n, replace=False)
            indices.extend(chosen.tolist())

    return np.array(indices)


def load_dataset(
    dataset_name: str,
    n_samples: Optional[int] = 5000,
    seed: int = 42,
    color_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load dataset with optional RGB preservation.

    Args:
        dataset_name: Dataset name from config
        n_samples: Max samples (None = all)
        seed: Random seed
        color_mode: 'grayscale', 'per_channel', or None (auto from config)

    Returns:
        images: (N, H, W) for grayscale or (N, H, W, 3) for RGB
        labels: (N,) integer labels
        class_names: list of class name strings
    """
    if color_mode is None:
        color_mode = DATASETS[dataset_name].get('color_mode', 'grayscale')

    # Determine target image size: keep original if <= 1024, else 1024
    cfg = DATASETS[dataset_name]
    native_size = cfg['image_size']
    if isinstance(native_size, str):  # 'variable'
        target_size = PH_CONFIG['max_image_size']
    elif native_size > PH_CONFIG['max_image_size']:
        target_size = PH_CONFIG['max_image_size']
    else:
        target_size = native_size

    if cfg['source'] == 'medmnist':
        images, labels, class_names = _load_medmnist_rgb(
            dataset_name, n_samples, seed, color_mode)
    else:
        images, labels, class_names = _load_external_rgb(
            dataset_name, n_samples, seed, target_size, color_mode)

    print(f"  Loaded {dataset_name}: shape={images.shape}, classes={len(class_names)}, "
          f"color={color_mode}, range=[{images.min():.3f}, {images.max():.3f}]")
    return images, labels, class_names


def _load_medmnist_rgb(
    dataset_name: str,
    n_samples: Optional[int],
    seed: int,
    color_mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load MedMNIST2 (224x224) with optional RGB preservation."""
    name_map = {
        'BloodMNIST': 'bloodmnist', 'TissueMNIST': 'tissuemnist',
        'PathMNIST': 'pathmnist', 'OCTMNIST': 'octmnist',
        'OrganAMNIST': 'organamnist', 'RetinaMNIST': 'retinamnist',
        'PneumoniaMNIST': 'pneumoniamnist', 'BreastMNIST': 'breastmnist',
        'DermaMNIST': 'dermamnist', 'OrganCMNIST': 'organcmnist',
        'OrganSMNIST': 'organsmnist',
    }
    npz_name = name_map[dataset_name]
    npz_path = MEDMNIST_PATH / f"{npz_name}_224.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"MedMNIST file not found: {npz_path}")

    data = np.load(npz_path, mmap_mode='r')
    images = data['train_images']
    labels = data['train_labels'].flatten()
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

    # Convert based on color mode
    if color_mode == 'per_channel' and images.ndim == 4 and images.shape[3] == 3:
        # Keep RGB: (N, H, W, 3) float32 [0, 1]
        images = images.astype(np.float32)
        if images.max() > 1.0:
            images = images / 255.0
    else:
        # Convert to grayscale
        images = _to_grayscale_float(images)

    return images, labels, class_names


def _load_external_rgb(
    dataset_name: str,
    n_samples: Optional[int],
    seed: int,
    target_size: int,
    color_mode: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load external dataset with optional RGB preservation.

    For 'per_channel' mode, returns (N, H, W, 3) RGB images.
    For 'grayscale' mode, returns (N, H, W) grayscale images.
    """
    # Loader map: dataset → (paths, labels, class_names) — no image loading yet
    loader_map = {
        'ISIC2019': lambda: _load_folder_paths(ISIC_PATH, n_samples, seed),
        'Kvasir': lambda: _load_folder_paths(KVASIR_PATH, n_samples, seed),
        'BrainTumorMRI': lambda: _load_folder_paths(BRAIN_TUMOR_PATH, n_samples, seed),
        'NCT_CRC_HE': lambda: _load_folder_paths(NCT_CRC_PATH, n_samples, seed),
        'MalariaCell': lambda: _load_folder_paths(MALARIA_PATH, n_samples, seed),
        'GasHisSDB': lambda: _load_gashissdb_paths(n_samples, seed),
        'BreakHis': lambda: _load_breakhis_paths(n_samples, seed),
        'LC25000': lambda: _load_lc25000_paths(n_samples, seed),
        'SIPaKMeD': lambda: _load_sipakmed_paths(n_samples, seed),
        'AML_Cytomorphology': lambda: _load_aml_paths(n_samples, seed),
        'APTOS2019': lambda: _load_aptos_paths(n_samples, seed),
        'IDRiD': lambda: _load_idrid_paths(n_samples, seed),
        'PCam': lambda: _load_pcam_rgb(n_samples, seed, target_size),
        'Chaoyang': lambda: _load_chaoyang_paths(n_samples, seed),
    }

    if dataset_name == 'MURA':
        # MURA is grayscale X-ray — always use benchmark3 loader
        from RuleBenchmark.benchmark3.data_loader import load_dataset as bm3_load
        return bm3_load(dataset_name, n_samples, seed, target_size)

    if dataset_name == 'PCam':
        # PCam has special HDF5 loading (returns RGB already)
        images, labels_arr, class_names = loader_map['PCam']()
        if color_mode == 'grayscale':
            images = _to_grayscale_float(images)
        return images, labels_arr, class_names

    # Path-based loading for all other external datasets
    paths, labels_arr, class_names = loader_map[dataset_name]()

    if color_mode == 'grayscale':
        # Load as RGB then convert to grayscale (ensures our _stratified_sample)
        images = _load_images_rgb(paths, target_size)
        images = _to_grayscale_float(images)
    else:
        images = _load_images_rgb(paths, target_size)

    return images, labels_arr, class_names


def _load_folder_paths(
    root_path: Path,
    n_samples: Optional[int],
    seed: int,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Get file paths and labels from folder-based dataset."""
    root_path = Path(root_path)
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    class_names = sorted([
        d for d in os.listdir(root_path)
        if os.path.isdir(root_path / d) and not d.startswith('.')
        and any(f.lower().endswith(valid_exts)
                for f in os.listdir(root_path / d) if not f.startswith('.'))
    ])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    all_paths, all_labels = [], []
    for cn in class_names:
        class_dir = root_path / cn
        for f in sorted(os.listdir(class_dir)):
            if f.lower().endswith(valid_exts) and not f.startswith('.'):
                all_paths.append(str(class_dir / f))
                all_labels.append(class_to_idx[cn])

    all_labels = np.array(all_labels)
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    paths = [all_paths[i] for i in indices]
    labels = all_labels[indices]
    return paths, labels, class_names


def _load_gashissdb_paths(n_samples, seed):
    """Get paths for GasHisSDB."""
    data_dir = None
    for candidate in ['160', 'Sub-image-160', '160x160']:
        d = GASHISSDB_PATH / candidate
        if d.exists():
            data_dir = d
            break
    if data_dir is None:
        for root_dir in [GASHISSDB_PATH] + list(GASHISSDB_PATH.iterdir()):
            if not root_dir.is_dir():
                continue
            if (root_dir / 'Normal').exists() or (root_dir / 'Abnormal').exists():
                data_dir = root_dir
                break
    if data_dir is None:
        raise FileNotFoundError(f"GasHisSDB not found at {GASHISSDB_PATH}")
    return _load_folder_paths(data_dir, n_samples, seed)


def _load_breakhis_paths(n_samples, seed):
    """Get paths for BreakHis."""
    class_names, all_paths, all_labels = [], [], []
    valid_exts = ('.png', '.jpg', '.jpeg')
    for category in ['benign', 'malignant']:
        sob_path = BREAKHIS_PATH / category / "SOB"
        if not sob_path.exists():
            continue
        for subtype in sorted(os.listdir(sob_path)):
            cn = f"{category}_{subtype}"
            if cn not in class_names:
                class_names.append(cn)
            label = class_names.index(cn)
            subtype_path = sob_path / subtype
            for patient in os.listdir(subtype_path):
                pp = subtype_path / patient
                if not pp.is_dir():
                    continue
                for mag in ['40X', '100X', '200X', '400X']:
                    mp = pp / mag
                    if not mp.exists():
                        continue
                    for f in os.listdir(mp):
                        if f.lower().endswith(valid_exts):
                            all_paths.append(str(mp / f))
                            all_labels.append(label)
    all_labels = np.array(all_labels)
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))
    return [all_paths[i] for i in indices], all_labels[indices], class_names


def _load_lc25000_paths(n_samples, seed):
    """Get paths for LC25000."""
    class_names, all_paths, all_labels = [], [], []
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
    return [all_paths[i] for i in indices], all_labels[indices], class_names


def _load_sipakmed_paths(n_samples, seed):
    """Get paths for SIPaKMeD."""
    class_names, all_paths, all_labels = [], [], []
    valid_exts = ('.bmp', '.jpg', '.jpeg', '.png')
    for cls_dir in sorted(os.listdir(SIPAKMED_PATH)):
        if not cls_dir.startswith('im_'):
            continue
        cropped_dir = SIPAKMED_PATH / cls_dir / cls_dir / "CROPPED"
        if not cropped_dir.exists():
            cropped_dir = SIPAKMED_PATH / cls_dir / cls_dir
        if not cropped_dir.exists():
            continue
        cn = cls_dir.replace('im_', '')
        class_names.append(cn)
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
    return [all_paths[i] for i in indices], all_labels[indices], class_names


def _load_aml_paths(n_samples, seed):
    """Get paths for AML_Cytomorphology."""
    class_names, all_paths, all_labels = [], [], []
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
    return [all_paths[i] for i in indices], all_labels[indices], class_names


def _load_aptos_paths(n_samples, seed):
    """Get paths for APTOS2019."""
    import csv
    class_names = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]
    all_paths, all_labels = [], []

    def _find_img_dir(base, name):
        d = base / name
        nested = d / name
        return nested if nested.exists() else d

    for csv_file, img_folder in [("train_1.csv", "train_images"), ("valid.csv", "val_images")]:
        csv_path = APTOS_PATH / csv_file
        img_dir = _find_img_dir(APTOS_PATH, img_folder)
        if not csv_path.exists() or not img_dir.exists():
            continue
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    img_id = row[0].strip()
                    label = int(row[1].strip())
                    for ext in ['.png', '.jpg', '.jpeg']:
                        ip = img_dir / f"{img_id}{ext}"
                        if ip.exists():
                            all_paths.append(str(ip))
                            all_labels.append(label)
                            break
    all_labels = np.array(all_labels)
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))
    return [all_paths[i] for i in indices], all_labels[indices], class_names


def _load_idrid_paths(n_samples, seed):
    """Get paths for IDRiD."""
    import csv
    class_names = ["DR_grade_0", "DR_grade_1", "DR_grade_2", "DR_grade_3", "DR_grade_4"]
    all_paths, all_labels = [], []
    images_dir = IDRID_PATH / "1. Original Images"
    labels_dir = IDRID_PATH / "2. Groundtruths"
    splits = [
        ("a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
        ("b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv"),
    ]
    for img_folder, label_file in splits:
        img_dir = images_dir / img_folder
        csv_path = labels_dir / label_file
        if not img_dir.exists() or not csv_path.exists():
            continue
        label_map = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2 and row[1].strip().isdigit():
                    label_map[row[0].strip()] = int(row[1].strip())
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(valid_exts):
                continue
            img_name = os.path.splitext(fname)[0]
            if img_name in label_map:
                all_paths.append(str(img_dir / fname))
                all_labels.append(label_map[img_name])
    all_labels = np.array(all_labels)
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))
    return [all_paths[i] for i in indices], all_labels[indices], class_names


def _load_chaoyang_paths(n_samples, seed):
    """Get paths for Chaoyang."""
    import json, re
    class_names = ["normal", "serrated", "adenocarcinoma", "adenoma"]
    all_paths, all_labels = [], []
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
            if isinstance(annotations, list):
                for item in annotations:
                    img_name = item.get('name', item.get('image', ''))
                    label = item.get('label', item.get('class', -1))
                    if isinstance(label, int) and 0 <= label <= 3:
                        ip = img_dir / Path(img_name).name
                        if not ip.exists():
                            ip = data_root / img_name
                        if ip.exists():
                            all_paths.append(str(ip))
                            all_labels.append(label)
    # Fallback: filename-encoded labels
    if not all_paths:
        for split in ['train', 'test']:
            img_dir = data_root / split
            if not img_dir.exists():
                continue
            for fname in sorted(img_dir.iterdir()):
                if fname.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                match = re.search(r'-(\d+)\.\w+$', fname.name)
                if match:
                    label = int(match.group(1))
                    if 0 <= label <= 3:
                        all_paths.append(str(fname))
                        all_labels.append(label)
    all_labels = np.array(all_labels)
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))
    return [all_paths[i] for i in indices], all_labels[indices], class_names


def _load_pcam_rgb(n_samples, seed, target_size):
    """Load PCam as RGB."""
    import h5py
    from PIL import Image

    class_names = ["normal", "metastasis"]
    train_x_h5 = PCAM_PATH / "camelyonpatch_level_2_split_train_x.h5"
    train_y_h5 = PCAM_PATH / "camelyonpatch_level_2_split_train_y.h5"
    if not train_x_h5.exists():
        raise FileNotFoundError(f"PCam .h5 files not found at {PCAM_PATH}")

    with h5py.File(str(train_y_h5), 'r') as f:
        labels = f['y'][:].flatten()

    if n_samples is not None and n_samples < len(labels):
        indices = _stratified_sample(labels, n_samples, seed)
    else:
        indices = np.arange(len(labels))

    labels = labels[indices]
    sorted_idx = np.sort(indices)
    with h5py.File(str(train_x_h5), 'r') as f:
        raw_images = f['x'][sorted_idx]  # (N, 96, 96, 3)

    idx_map = {v: i for i, v in enumerate(sorted_idx)}
    reorder_idx = [idx_map[v] for v in indices]
    raw_images = raw_images[reorder_idx]

    # Resize and keep RGB
    images = []
    for img_arr in raw_images:
        img = Image.fromarray(img_arr)
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr)

    images = np.array(images, dtype=np.float32)
    return images, labels, class_names


# =============================================================================
# Image Loading Utilities
# =============================================================================

def _load_images_rgb(paths: List[str], image_size: int) -> np.ndarray:
    """Load images as RGB float32 [0, 1], resized to image_size."""
    from PIL import Image

    images = []
    for path in paths:
        try:
            img = Image.open(path).convert('RGB')
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
        except Exception as e:
            print(f"  WARNING: Failed to load {path}: {e}")
            images.append(np.zeros((image_size, image_size, 3), dtype=np.float32))

    return np.array(images, dtype=np.float32)


def _to_grayscale_float(images: np.ndarray) -> np.ndarray:
    """Convert to grayscale float32 [0, 1]."""
    if images.ndim == 4 and images.shape[3] == 3:
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


def rgb_to_channels(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split RGB images into R, G, B channels.

    Args:
        images: (N, H, W, 3) float32 [0, 1]

    Returns:
        R, G, B: each (N, H, W) float32 [0, 1]
    """
    assert images.ndim == 4 and images.shape[3] == 3, \
        f"Expected (N, H, W, 3), got {images.shape}"
    return images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]


# =============================================================================
# Batched Loading for PH Precomputation (memory-efficient)
# =============================================================================

# MedMNIST name mapping (shared with _load_medmnist_rgb)
_MEDMNIST_NAME_MAP = {
    'BloodMNIST': 'bloodmnist', 'TissueMNIST': 'tissuemnist',
    'PathMNIST': 'pathmnist', 'OCTMNIST': 'octmnist',
    'OrganAMNIST': 'organamnist', 'RetinaMNIST': 'retinamnist',
    'PneumoniaMNIST': 'pneumoniamnist', 'BreastMNIST': 'breastmnist',
    'DermaMNIST': 'dermamnist', 'OrganCMNIST': 'organcmnist',
    'OrganSMNIST': 'organsmnist',
}


def prepare_batched_loader(
    dataset_name: str,
    n_samples: Optional[int] = 5000,
    seed: int = 42,
):
    """Prepare dataset for memory-efficient batched image loading.

    Returns (labels, class_names, load_batch_fn) where:
      - labels: (N,) integer array
      - class_names: list of str
      - load_batch_fn(start, end) -> images (batch_size, H, W) or (batch_size, H, W, 3)

    The load_batch_fn loads only the requested slice into memory.
    For MedMNIST: uses numpy mmap (zero-copy until sliced).
    For external: stores file paths, loads on demand.
    """
    cfg = DATASETS[dataset_name]
    color_mode = cfg['color_mode']

    if cfg['source'] == 'medmnist':
        return _prepare_medmnist_batched(dataset_name, n_samples, seed, color_mode)
    else:
        return _prepare_external_batched(dataset_name, n_samples, seed, color_mode)


def _prepare_medmnist_batched(dataset_name, n_samples, seed, color_mode):
    """MedMNIST batched loader using numpy mmap."""
    npz_name = _MEDMNIST_NAME_MAP[dataset_name]
    npz_path = MEDMNIST_PATH / f"{npz_name}_224.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"MedMNIST file not found: {npz_path}")

    # mmap — no memory allocated for images yet
    data = np.load(str(npz_path), mmap_mode='r')
    all_images = data['train_images']
    all_labels = data['train_labels'].flatten()
    n_classes = len(np.unique(all_labels))
    class_names = [str(i) for i in range(n_classes)]

    # Get sampling indices
    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    labels = all_labels[indices].copy()

    def load_batch(start, end):
        batch_idx = indices[start:end]
        # Sort indices for efficient sequential mmap reads
        sorted_order = np.argsort(batch_idx)
        sorted_idx = batch_idx[sorted_order]
        raw = np.array(all_images[sorted_idx])
        # Restore original order
        images = np.empty_like(raw)
        images[sorted_order] = raw

        images = images.astype(np.float32)
        if images.max() > 1.0:
            images /= 255.0

        if color_mode == 'grayscale' or images.ndim == 3:
            images = _to_grayscale_float(images)
        return images

    return labels, class_names, load_batch


def _prepare_external_batched(dataset_name, n_samples, seed, color_mode):
    """External dataset batched loader using file paths."""
    from RuleBenchmark.benchmark4.config import get_ph_image_size
    target_size = get_ph_image_size(dataset_name)

    if dataset_name == 'MURA':
        # MURA is always grayscale, use benchmark3 full loading
        from RuleBenchmark.benchmark3.data_loader import load_dataset as bm3_load
        images, labels, class_names = bm3_load(dataset_name, n_samples, seed, target_size)
        # Wrap in a simple batch function (images already loaded)
        def load_batch(start, end):
            return images[start:end]
        return labels, class_names, load_batch

    if dataset_name == 'PCam':
        return _prepare_pcam_batched(n_samples, seed, target_size, color_mode)

    # Path-based datasets: get all paths first (lightweight)
    path_loaders = {
        'ISIC2019': lambda: _load_folder_paths(ISIC_PATH, n_samples, seed),
        'Kvasir': lambda: _load_folder_paths(KVASIR_PATH, n_samples, seed),
        'BrainTumorMRI': lambda: _load_folder_paths(BRAIN_TUMOR_PATH, n_samples, seed),
        'NCT_CRC_HE': lambda: _load_folder_paths(NCT_CRC_PATH, n_samples, seed),
        'MalariaCell': lambda: _load_folder_paths(MALARIA_PATH, n_samples, seed),
        'GasHisSDB': lambda: _load_gashissdb_paths(n_samples, seed),
        'BreakHis': lambda: _load_breakhis_paths(n_samples, seed),
        'LC25000': lambda: _load_lc25000_paths(n_samples, seed),
        'SIPaKMeD': lambda: _load_sipakmed_paths(n_samples, seed),
        'AML_Cytomorphology': lambda: _load_aml_paths(n_samples, seed),
        'APTOS2019': lambda: _load_aptos_paths(n_samples, seed),
        'IDRiD': lambda: _load_idrid_paths(n_samples, seed),
        'Chaoyang': lambda: _load_chaoyang_paths(n_samples, seed),
    }

    paths, labels, class_names = path_loaders[dataset_name]()

    def load_batch(start, end):
        batch_paths = paths[start:end]
        images = _load_images_rgb(batch_paths, target_size)
        if color_mode == 'grayscale':
            images = _to_grayscale_float(images)
        return images

    return labels, class_names, load_batch


def _prepare_pcam_batched(n_samples, seed, target_size, color_mode):
    """PCam batched loader using HDF5 slicing."""
    import h5py
    from PIL import Image

    class_names = ["normal", "metastasis"]
    train_x_h5 = PCAM_PATH / "camelyonpatch_level_2_split_train_x.h5"
    train_y_h5 = PCAM_PATH / "camelyonpatch_level_2_split_train_y.h5"
    if not train_x_h5.exists():
        raise FileNotFoundError(f"PCam .h5 files not found at {PCAM_PATH}")

    with h5py.File(str(train_y_h5), 'r') as f:
        all_labels = f['y'][:].flatten()

    if n_samples is not None and n_samples < len(all_labels):
        indices = _stratified_sample(all_labels, n_samples, seed)
    else:
        indices = np.arange(len(all_labels))

    labels = all_labels[indices]

    def load_batch(start, end):
        batch_idx = indices[start:end]
        # Sort for efficient HDF5 sequential reads
        sorted_order = np.argsort(batch_idx)
        sorted_idx = batch_idx[sorted_order]

        with h5py.File(str(train_x_h5), 'r') as f:
            raw = f['x'][sorted_idx]  # (batch, 96, 96, 3)

        # Restore original order
        images = np.empty_like(raw)
        images[sorted_order] = raw

        # Resize if needed and convert to float
        result = []
        for img_arr in images:
            img = Image.fromarray(img_arr)
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            result.append(arr)

        images = np.array(result, dtype=np.float32)
        if color_mode == 'grayscale':
            images = _to_grayscale_float(images)
        return images

    return labels, class_names, load_batch


if __name__ == '__main__':
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['BloodMNIST', 'OrganAMNIST']
    for name in datasets:
        print(f"\n{'='*50}")
        print(f"  {name} (color_mode={DATASETS[name]['color_mode']})")
        print(f"{'='*50}")
        try:
            images, labels, cnames = load_dataset(name, n_samples=100, seed=42)
            print(f"  Shape: {images.shape}, dtype: {images.dtype}")
            print(f"  Classes: {cnames[:5]}{'...' if len(cnames) > 5 else ''}")
        except Exception as e:
            print(f"  FAILED: {e}")
