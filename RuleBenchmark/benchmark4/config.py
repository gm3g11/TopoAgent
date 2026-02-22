"""
Benchmark4 Master Configuration.

26 datasets × 15 descriptors × 6 classifiers = 2,340 accuracy values.
5-fold stratified CV, balanced accuracy, n=min(5000, available).

Key changes from Benchmark3:
- 6 classifiers: TabPFN, XGBoost, CatBoost, RF, TabM, RealMLP (dropped KNN)
- Optimal params from exp4 rules (dimension, parameter, color)
- Per-channel RGB for 17 RGB datasets (9 grayscale)
- Image size: keep original if < 1024, resize to 1024 if > 1024
- PH via CuPH (GPU) with Cripser fallback
- TabPFN PCA-bagging for >2000D features
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = Path(__file__).parent
BENCHMARK3_DIR = PROJECT_ROOT / "benchmarks" / "benchmark3"

MEDMNIST_PATH = Path(os.environ.get("MEDMNIST_PATH", str(Path.home() / ".medmnist")))
RESULTS_PATH = PROJECT_ROOT / "results" / "benchmark4"
PH_CACHE_PATH = RESULTS_PATH / "ph_cache"
RAW_RESULTS_PATH = RESULTS_PATH / "raw"
SUMMARY_PATH = RESULTS_PATH / "summary"

# Exp4 rules (from benchmark3)
EXP4_RULES_PATH = PROJECT_ROOT / "results" / "benchmark3" / "exp4" / "exp4_final_recommendations.json"

# CuPH library (optional GPU-accelerated PH; set CUPH_PATH env var if available)
CUPH_PATH = Path(os.environ.get("CUPH_PATH", "/opt/CuPH"))

# External dataset paths (override via environment variables)
EXTERNAL_DATASETS_ROOT = Path(os.environ.get("EXTERNAL_DATASETS_ROOT", str(PROJECT_ROOT / "data" / "external")))
ISIC_PATH = Path(os.environ.get("ISIC_PATH", str(EXTERNAL_DATASETS_ROOT / "isic2019" / "train")))
KVASIR_PATH = Path(os.environ.get("KVASIR_PATH", str(EXTERNAL_DATASETS_ROOT / "kvasir-dataset")))
BRAIN_TUMOR_PATH = EXTERNAL_DATASETS_ROOT / "data" / "BrainTumorMRI" / "Training"
MURA_PATH = EXTERNAL_DATASETS_ROOT / "MURA-v1.1"
BREAKHIS_PATH = (EXTERNAL_DATASETS_ROOT / "ambarish" / "breakhis" / "versions" / "4"
                 / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast")
NCT_CRC_PATH = (EXTERNAL_DATASETS_ROOT / "nct-crc-he-100k" / "versions" / "1"
                / "NCT-CRC-HE-100K")
MALARIA_PATH = EXTERNAL_DATASETS_ROOT / "data" / "MalariaCell" / "cell_images"
IDRID_PATH = EXTERNAL_DATASETS_ROOT / "idrid" / "Disease_Grading"
PCAM_PATH = EXTERNAL_DATASETS_ROOT / "pcam"
LC25000_PATH = EXTERNAL_DATASETS_ROOT / "lc25000" / "lung_colon_image_set"
SIPAKMED_PATH = EXTERNAL_DATASETS_ROOT / "sipakmed"
AML_PATH = EXTERNAL_DATASETS_ROOT / "aml_cytomorphology" / "data" / "data"
APTOS_PATH = EXTERNAL_DATASETS_ROOT / "aptos2019"
GASHISSDB_PATH = EXTERNAL_DATASETS_ROOT / "gashissdb" / "GasHisSDB"
CHAOYANG_PATH = EXTERNAL_DATASETS_ROOT / "chaoyang"

# =============================================================================
# 1. DATASETS (26 total: 11 MedMNIST + 15 external)
# =============================================================================
# image_size: native resolution (MedMNIST2 = 224x224)
# color_mode: 'grayscale' or 'per_channel'
# object_type: one of 5 types from exp4 rules
# n_samples: min(5000, available)

DATASETS = {
    # --- MedMNIST2 (11) - 224x224 native ---
    'BloodMNIST': {
        'source': 'medmnist',
        'object_type': 'discrete_cells',
        'n_classes': 8,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'per_channel',  # RGB
    },
    'TissueMNIST': {
        'source': 'medmnist',
        'object_type': 'discrete_cells',
        'n_classes': 8,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'grayscale',  # n_channels=1
    },
    'PathMNIST': {
        'source': 'medmnist',
        'object_type': 'glands_lumens',
        'n_classes': 9,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'OCTMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 4,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'grayscale',
    },
    'OrganAMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 11,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'grayscale',
    },
    'RetinaMNIST': {
        'source': 'medmnist',
        'object_type': 'vessel_trees',
        'n_classes': 5,
        'image_size': 224,
        'n_samples': 1080,  # only 1080 training samples available
        'color_mode': 'per_channel',
    },
    'PneumoniaMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 2,
        'image_size': 224,
        'n_samples': 4708,
        'color_mode': 'grayscale',
    },
    'BreastMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 2,
        'image_size': 224,
        'n_samples': 546,  # small dataset
        'color_mode': 'grayscale',
    },
    'DermaMNIST': {
        'source': 'medmnist',
        'object_type': 'surface_lesions',
        'n_classes': 7,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'OrganCMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 11,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'grayscale',
    },
    'OrganSMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 11,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'grayscale',
    },
    # --- External (15) ---
    # image_size: native. PH precompute resizes to min(native, 1024).
    'ISIC2019': {
        'source': 'kaggle',
        'object_type': 'surface_lesions',
        'n_classes': 8,
        'image_size': 1022,  # resized to 1024 for PH
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'Kvasir': {
        'source': 'kaggle',
        'object_type': 'glands_lumens',
        'n_classes': 8,
        'image_size': 720,
        'n_samples': 4000,  # only 4000 available (500 per class × 8)
        'color_mode': 'per_channel',
    },
    'BrainTumorMRI': {
        'source': 'kaggle',
        'object_type': 'organ_shape',
        'n_classes': 4,
        'image_size': 512,
        'n_samples': 5000,
        'color_mode': 'grayscale',
    },
    'MURA': {
        'source': 'stanford',
        'object_type': 'organ_shape',
        'n_classes': 14,
        'image_size': 'variable',  # capped at 1024
        'n_samples': 5000,
        'color_mode': 'grayscale',
    },
    'BreakHis': {
        'source': 'kaggle',
        'object_type': 'glands_lumens',
        'n_classes': 8,
        'image_size': 700,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'NCT_CRC_HE': {
        'source': 'zenodo',
        'object_type': 'glands_lumens',
        'n_classes': 9,
        'image_size': 224,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'MalariaCell': {
        'source': 'kaggle',
        'object_type': 'discrete_cells',
        'n_classes': 2,
        'image_size': 142,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'IDRiD': {
        'source': 'ieee',
        'object_type': 'vessel_trees',
        'n_classes': 5,
        'image_size': 4288,  # resized to 1024 for PH
        'n_samples': 516,  # small dataset
        'color_mode': 'per_channel',
    },
    'PCam': {
        'source': 'github',
        'object_type': 'discrete_cells',
        'n_classes': 2,
        'image_size': 96,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'LC25000': {
        'source': 'kaggle',
        'object_type': 'glands_lumens',
        'n_classes': 5,
        'image_size': 768,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'SIPaKMeD': {
        'source': 'kaggle',
        'object_type': 'discrete_cells',
        'n_classes': 5,
        'image_size': 'variable',  # capped at 1024
        'n_samples': 4049,
        'color_mode': 'per_channel',
    },
    'AML_Cytomorphology': {
        'source': 'tcia',
        'object_type': 'discrete_cells',
        'n_classes': 15,
        'image_size': 'variable',
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'APTOS2019': {
        'source': 'kaggle',
        'object_type': 'vessel_trees',
        'n_classes': 5,
        'image_size': 1050,  # resized to 1024 for PH
        'n_samples': 3296,
        'color_mode': 'per_channel',
    },
    'GasHisSDB': {
        'source': 'figshare',
        'object_type': 'surface_lesions',
        'n_classes': 2,
        'image_size': 160,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
    'Chaoyang': {
        'source': 'google_drive',
        'object_type': 'glands_lumens',
        'n_classes': 4,
        'image_size': 512,
        'n_samples': 5000,
        'color_mode': 'per_channel',
    },
}

ALL_DATASETS = list(DATASETS.keys())
N_DATASETS = len(DATASETS)  # 26

# Convenience groupings
GRAYSCALE_DATASETS = [d for d, cfg in DATASETS.items() if cfg['color_mode'] == 'grayscale']
RGB_DATASETS = [d for d, cfg in DATASETS.items() if cfg['color_mode'] == 'per_channel']
MANY_CLASS_DATASETS = [d for d, cfg in DATASETS.items() if cfg['n_classes'] > 10]
SMALL_DATASETS = [d for d, cfg in DATASETS.items() if cfg['n_samples'] < 2000]

# Object type -> dataset mapping
OBJECT_TYPE_DATASETS = {}
for d, cfg in DATASETS.items():
    ot = cfg['object_type']
    OBJECT_TYPE_DATASETS.setdefault(ot, []).append(d)

# =============================================================================
# 2. DESCRIPTORS (15: 13 TDA + 2 baseline)
# =============================================================================
DESCRIPTORS = {
    'persistence_image':              {'type': 'TDA', 'ph_based': True,  'learned': False},
    'persistence_landscapes':         {'type': 'TDA', 'ph_based': True,  'learned': False},
    'betti_curves':                   {'type': 'TDA', 'ph_based': True,  'learned': False},
    'persistence_silhouette':         {'type': 'TDA', 'ph_based': True,  'learned': False},
    'persistence_entropy':            {'type': 'TDA', 'ph_based': True,  'learned': False},
    'persistence_statistics':         {'type': 'TDA', 'ph_based': True,  'learned': False},
    'tropical_coordinates':           {'type': 'TDA', 'ph_based': True,  'learned': False},
    'persistence_codebook':           {'type': 'TDA', 'ph_based': True,  'learned': True},
    'ATOL':                           {'type': 'TDA', 'ph_based': True,  'learned': True},
    'template_functions':             {'type': 'TDA', 'ph_based': True,  'learned': False},
    'minkowski_functionals':          {'type': 'TDA', 'ph_based': False, 'learned': False},
    'euler_characteristic_curve':     {'type': 'TDA', 'ph_based': False, 'learned': False},
    'euler_characteristic_transform': {'type': 'TDA', 'ph_based': False, 'learned': False},
    'edge_histogram':                 {'type': 'baseline', 'ph_based': False, 'learned': False},
    'lbp_texture':                    {'type': 'baseline', 'ph_based': False, 'learned': False},
}

ALL_DESCRIPTORS = list(DESCRIPTORS.keys())
N_DESCRIPTORS = len(DESCRIPTORS)  # 15
PH_BASED_DESCRIPTORS = [d for d, cfg in DESCRIPTORS.items() if cfg['ph_based']]
IMAGE_BASED_DESCRIPTORS = [d for d, cfg in DESCRIPTORS.items() if not cfg['ph_based']]
LEARNED_DESCRIPTORS = [d for d, cfg in DESCRIPTORS.items() if cfg['learned']]

# =============================================================================
# 3. CLASSIFIERS (6)
# =============================================================================
CLASSIFIERS = {
    'TabPFN': {
        'library': 'tabpfn',
        'params': {
            'device': 'cuda',
            'n_estimators': 16,
        },
        'max_features': 2000,  # PCA-bagging above this
        'max_classes_native': 10,  # ECOC above this
        'pca_bagging': {
            'n_projections': 5,
            'projection_dim': 500,
        },
    },
    'XGBoost': {
        'library': 'xgboost',
        'params': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'device': 'cpu',
            'verbosity': 0,
        },
    },
    'CatBoost': {
        'library': 'catboost',
        'params': {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'auto_class_weights': 'Balanced',
            'verbose': 0,
            'task_type': 'CPU',
        },
    },
    'RandomForest': {
        'library': 'sklearn',
        'params': {
            'n_estimators': 500,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'n_jobs': -1,
        },
    },
    'TabM': {
        'library': 'pytabkit',
        'params': {
            'n_epochs': 100,
        },
    },
    'RealMLP': {
        'library': 'pytabkit',
        'params': {
            'n_epochs': 100,
        },
    },
}

ACTIVE_CLASSIFIERS = list(CLASSIFIERS.keys())
N_CLASSIFIERS = len(CLASSIFIERS)  # 6: TabPFN, XGBoost, CatBoost, RF, TabM, RealMLP

# =============================================================================
# 4. EVALUATION CONFIG
# =============================================================================
EVALUATION = {
    'primary_metric': 'balanced_accuracy',
    'n_folds': 5,
    'seed': 42,
    'preprocessing': 'standard',  # StandardScaler + SimpleImputer(median)
}

# =============================================================================
# 5. PH CONFIG
# =============================================================================
PH_CONFIG = {
    'filtration': 'sublevel',
    'homology_dimensions': [0, 1],
    'max_image_size': 1024,  # resize if larger
    'batch_size_cuph': 256,
    'n_jobs_cripser': 16,
}

# =============================================================================
# 6. EXPERIMENT MATRIX
# =============================================================================
N_CONFIGURATIONS = N_DATASETS * N_DESCRIPTORS  # 390
TOTAL_EXPERIMENTS = N_CONFIGURATIONS * N_CLASSIFIERS  # 1,950

# =============================================================================
# 7. HELPER FUNCTIONS
# =============================================================================

def get_ph_image_size(dataset_name: str) -> int:
    """Get the image size to use for PH computation.

    Rule: keep original if <= 1024, resize to 1024 if > 1024.
    MedMNIST2: always 224 (native).
    """
    cfg = DATASETS[dataset_name]
    size = cfg['image_size']
    if isinstance(size, str):  # 'variable'
        return PH_CONFIG['max_image_size']
    return min(size, PH_CONFIG['max_image_size'])


def get_effective_n_samples(dataset_name: str) -> int:
    """Get the actual n_samples (capped by available)."""
    return DATASETS[dataset_name]['n_samples']


def get_object_type(dataset_name: str) -> str:
    """Get object type for a dataset."""
    return DATASETS[dataset_name]['object_type']


def get_color_mode(dataset_name: str) -> str:
    """Get color mode for a dataset."""
    return DATASETS[dataset_name]['color_mode']


def is_many_class(dataset_name: str) -> bool:
    """Check if dataset has >10 classes (needs ECOC for TabPFN)."""
    return DATASETS[dataset_name]['n_classes'] > 10


# =============================================================================
# 8. SUMMARY
# =============================================================================
SUMMARY = f"""
======================================================================
           Benchmark4 Configuration
======================================================================
 DATASETS:        {N_DATASETS}  (11 MedMNIST2 + 15 external)
   Grayscale:     {len(GRAYSCALE_DATASETS)}  {GRAYSCALE_DATASETS}
   RGB:           {len(RGB_DATASETS)}
 DESCRIPTORS:     {N_DESCRIPTORS}  (13 TDA + 2 baseline)
   PH-based:      {len(PH_BASED_DESCRIPTORS)}
   Image-based:   {len(IMAGE_BASED_DESCRIPTORS)}
   Learned:       {len(LEARNED_DESCRIPTORS)}  {LEARNED_DESCRIPTORS}
 CLASSIFIERS:      {N_CLASSIFIERS}  {ACTIVE_CLASSIFIERS}
 CV FOLDS:         {EVALUATION['n_folds']}
----------------------------------------------------------------------
 CONFIGURATIONS: {N_CONFIGURATIONS}  ({N_DATASETS} × {N_DESCRIPTORS})
 TOTAL EVALS:    {TOTAL_EXPERIMENTS}  ({N_CONFIGURATIONS} × {N_CLASSIFIERS})
----------------------------------------------------------------------
 PRIMARY METRIC:   balanced_accuracy
 PH LIBRARY:       CuPH (GPU) → Cripser (CPU) fallback
 IMAGE SIZE:       original if ≤1024, else 1024
======================================================================
"""

if __name__ == '__main__':
    print(SUMMARY)

    print("\nObject type mapping:")
    for ot, datasets in sorted(OBJECT_TYPE_DATASETS.items()):
        print(f"  {ot} ({len(datasets)}): {datasets}")

    print(f"\nMany-class datasets (>10): {MANY_CLASS_DATASETS}")
    print(f"Small datasets (<2000): {SMALL_DATASETS}")
