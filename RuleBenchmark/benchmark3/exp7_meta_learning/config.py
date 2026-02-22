"""Configuration for Exp7: Meta-Learning for Descriptor Selection."""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # TopoAgent root
EXP1_CSV = PROJECT_ROOT / "results" / "benchmark3" / "exp1" / "benchmark3_all_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark3" / "exp7"
CACHE_DIR = OUTPUT_DIR / "cache"
MODELS_DIR = OUTPUT_DIR / "models"

# MedMNIST and external dataset paths (from benchmark3 config)
MEDMNIST_PATH = Path(os.environ.get("MEDMNIST_PATH", str(Path.home() / ".medmnist")))
EXTERNAL_DATASETS_ROOT = Path(os.environ.get("EXTERNAL_DATASETS_ROOT", str(PROJECT_ROOT / "data" / "external")))

# ─── Datasets (13) ──────────────────────────────────────────────────────────

DATASETS = [
    "BloodMNIST",
    "TissueMNIST",
    "PathMNIST",
    "OCTMNIST",
    "OrganAMNIST",
    "RetinaMNIST",
    "ISIC2019",
    "Kvasir",
    "BrainTumorMRI",
    "MURA",
    "BreakHis",
    "NCT_CRC_HE",
    "MalariaCell",
]

# ─── Descriptors (15) ───────────────────────────────────────────────────────

DESCRIPTORS = [
    "persistence_statistics",
    "persistence_image",
    "persistence_landscapes",
    "persistence_silhouette",
    "betti_curves",
    "persistence_entropy",
    "persistence_codebook",
    "tropical_coordinates",
    "ATOL",
    "template_functions",
    "minkowski_functionals",
    "euler_characteristic_curve",
    "euler_characteristic_transform",
    "lbp_texture",
    "edge_histogram",
]

# ─── 25 Cheap Feature Names (11 groups) ─────────────────────────────────────

FEATURE_NAMES = [
    # Structure (4)
    "n_samples",
    "n_classes",
    "samples_per_class",
    "class_imbalance",
    # Intensity (6)
    "intensity_mean",
    "intensity_std",
    "intensity_skewness",
    "intensity_kurtosis",
    "intensity_p95_p5",
    "intensity_entropy",
    # Gradient (2)
    "edge_density",
    "gradient_mean",
    # Topology proxies (3)
    "otsu_components",
    "otsu_holes",
    "binary_fill_ratio",
    # Texture (2)
    "glcm_contrast",
    "glcm_homogeneity",
    # Frequency (1)
    "fft_low_freq_ratio",
    # Stability (1)
    "otsu_stability",
    # Components (2)
    "component_size_mean",
    "component_size_cv",
    # Polarity (2)
    "fg_bg_contrast",
    "polarity",
    # Scale (1)
    "largest_component_ratio",
    # Fine texture (1)
    "laplacian_variance",
]

N_FEATURES = len(FEATURE_NAMES)  # 25

# ─── Feature Computation Parameters ─────────────────────────────────────────

N_REPEATS = 5              # Number of random subsamples for stability
SAMPLES_PER_REPEAT = 100   # Images per subsample
DEFAULT_N_SAMPLES = 500    # Total images to load from each dataset

# ─── Model Hyperparameters ───────────────────────────────────────────────────

GBR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 3,
    "subsample": 0.8,
    "random_state": 42,
}

TREE_PARAMS = {
    "max_depth": 4,
    "min_samples_leaf": 3,
    "random_state": 42,
}

RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": 5,
}

N_RF_SEEDS = 10  # Number of RF models in ensemble for uncertainty

# ─── Classifier Filter ──────────────────────────────────────────────────────

TARGET_CLASSIFIER = "TabPFN"
TARGET_METRIC = "balanced_accuracy_mean"

# ─── Random Seed ─────────────────────────────────────────────────────────────

SEED = 42
