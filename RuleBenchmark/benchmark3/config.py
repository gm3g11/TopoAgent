"""
Benchmark3 Experiment 1 Configuration.

Baseline Benchmark: 15 Descriptors × 19 Datasets × 4 Classifiers.

Key changes from Benchmark2:
- Removed 100D cap (TabPFN 2.5 supports native dimensions)
- 15 descriptors (13 TDA + 2 baseline): added template_functions, ECT, edge_histogram;
  removed carlsson_coordinates, complex_polynomial, heat_kernel, topological_vector,
  persistence_lengths
- persistence_entropy: 200D CURVE (custom implementation), not 2D scalar
- persistence_landscapes: 400D (4 layers × 100 bins, single concat)
- 26 datasets (11 MedMNIST + 15 external)
- Classifiers: TabPFN, RandomForest, XGBoost, KNN (replaced LogReg with RF)
- Evaluation: balanced_accuracy primary, 5 folds × 1 seed
- PH computed at 224×224 resolution (GUDHI + joblib)
- StandardScaler preprocessing, NO PCA
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
MEDMNIST_PATH = Path(os.environ.get("MEDMNIST_PATH", str(Path.home() / ".medmnist")))
RESULTS_PATH = PROJECT_ROOT / "results" / "benchmark3"

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
LC25000_PATH = (EXTERNAL_DATASETS_ROOT / "lc25000" / "lung_colon_image_set")
SIPAKMED_PATH = EXTERNAL_DATASETS_ROOT / "sipakmed"
AML_PATH = EXTERNAL_DATASETS_ROOT / "aml_cytomorphology" / "data" / "data"
APTOS_PATH = EXTERNAL_DATASETS_ROOT / "aptos2019"
GASHISSDB_PATH = EXTERNAL_DATASETS_ROOT / "gashissdb" / "GasHisSDB"
CHAOYANG_PATH = EXTERNAL_DATASETS_ROOT / "chaoyang"

# =============================================================================
# 1. CLASSIFIERS CONFIGURATION
# =============================================================================
CLASSIFIERS = {
    'TabPFN': {
        'enabled': True,
        'library': 'tabpfn',
        'class': 'TabPFNClassifier',
        'params': {
            'device': 'cuda',
            'n_estimators': 32,
        },
        'max_samples': 10000,
        'supports_multiclass': True,
        'notes': 'Use native feature dimensions. No PCA preprocessing.',
    },
    'RandomForest': {
        'enabled': True,
        'library': 'sklearn.ensemble',
        'class': 'RandomForestClassifier',
        'params': {
            'n_estimators': 500,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': None,  # Set per-run
        },
        'supports_multiclass': True,
        'notes': 'Robust baseline, provides feature importance.',
    },
    'XGBoost': {
        'enabled': True,
        'library': 'xgboost',
        'class': 'XGBClassifier',
        'params': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': None,  # Set per-run
        },
        'supports_multiclass': True,
        'notes': 'Gradient boosting with GPU support.',
    },
    'KNN': {
        'enabled': True,
        'library': 'sklearn.neighbors',
        'class': 'KNeighborsClassifier',
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'cosine',
            'algorithm': 'brute',
            'n_jobs': -1,
        },
        'hyperparameter_grid': {
            'n_neighbors': [3, 5, 7, 11, 15, 21],
        },
        'supports_multiclass': True,
        'notes': 'Cosine distance for high-dim TDA features.',
    },
}

ACTIVE_CLASSIFIERS = [name for name, cfg in CLASSIFIERS.items() if cfg['enabled']]
# ['TabPFN', 'RandomForest', 'XGBoost', 'KNN']

# =============================================================================
# 2. DATASETS CONFIGURATION (26 total)
# =============================================================================
# NOTE: image_size is the NATIVE size. For PH computation:
#   - MedMNIST: uses pre-resized 224×224 NPZ files
#   - External: adaptive resize (if native > 1024 -> 1024, else keep native)
DATASETS = {
    # --- MedMNIST (11) - using 224×224 pre-resized versions ---
    'BloodMNIST': {
        'source': 'medmnist',
        'object_type': 'discrete_cells',
        'n_classes': 8,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'H0 dominant (isolated cells)',
    },
    'TissueMNIST': {
        'source': 'medmnist',
        'object_type': 'discrete_cells',
        'n_classes': 8,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'H0 dominant (cell clusters)',
    },
    'PathMNIST': {
        'source': 'medmnist',
        'object_type': 'glands_lumens',
        'n_classes': 9,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'H1 important (gland lumens)',
    },
    'OCTMNIST': {
        'source': 'medmnist',
        'object_type': 'layered_structure',
        'n_classes': 4,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'Layer boundaries, H0/H1 mixed',
    },
    'OrganAMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 11,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'Global shape, ECT relevant',
    },
    'RetinaMNIST': {
        'source': 'medmnist',
        'object_type': 'vessel_trees',
        'n_classes': 5,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 1600,  # Small dataset
        'topology_hypothesis': 'H1 dominant (vessel loops/branches)',
    },
    'PneumoniaMNIST': {
        'source': 'medmnist',
        'object_type': 'lung_opacity',
        'n_classes': 2,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 4708,  # Train split size
        'topology_hypothesis': 'Diffuse opacity patterns, H0 dominant',
    },
    'BreastMNIST': {
        'source': 'medmnist',
        'object_type': 'ultrasound_mass',
        'n_classes': 2,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 546,  # Small dataset (train split)
        'topology_hypothesis': 'Mass boundary topology, H0/H1 mixed',
    },
    'DermaMNIST': {
        'source': 'medmnist',
        'object_type': 'skin_lesion',
        'n_classes': 7,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'Boundary irregularity, similar to ISIC2019',
    },
    'OrganCMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 11,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'Coronal organ shape, different topology from axial',
    },
    'OrganSMNIST': {
        'source': 'medmnist',
        'object_type': 'organ_shape',
        'n_classes': 11,
        'image_size': 224,  # Native 28, using 224×224 version
        'n_samples': 5000,
        'topology_hypothesis': 'Sagittal organ shape, different topology from axial/coronal',
    },
    # --- External (15) ---
    # NOTE: image_size is the NATIVE size. PH precompute uses adaptive resize:
    #       if native > 1024 -> resize to 1024, else keep native size.
    'ISIC2019': {
        'source': 'kaggle',
        'object_type': 'surface_lesions',
        'n_classes': 8,
        'image_size': 1022,  # Actual: ~1022x767 (variable)
        'n_samples': 5000,
        'topology_hypothesis': 'Boundary irregularity, mixed H0/H1',
    },
    'Kvasir': {
        'source': 'kaggle',
        'object_type': 'lesions_cavities',
        'n_classes': 8,
        'image_size': 720,  # Actual: 720x576
        'n_samples': 5000,
        'topology_hypothesis': 'H1 for polyps/cavities',
    },
    'BrainTumorMRI': {
        'source': 'kaggle',
        'object_type': 'solid_tumor',
        'n_classes': 4,
        'image_size': 512,  # Actual: 512x512
        'n_samples': 5000,
        'topology_hypothesis': 'Tumor boundaries, H0 dominant',
    },
    'MURA': {
        'source': 'stanford',
        'object_type': 'bone_structure',
        'n_classes': 14,
        'image_size': 'variable',  # X-ray images vary in size
        'n_samples': 5000,
        'topology_hypothesis': 'Bone density patterns, mixed',
    },
    'BreakHis': {
        'source': 'kaggle',
        'object_type': 'tumor_architecture',
        'n_classes': 8,
        'image_size': 700,  # Actual: 700x460
        'n_samples': 5000,
        'topology_hypothesis': 'Glandular architecture, H1 relevant',
    },
    'NCT_CRC_HE': {
        'source': 'zenodo',
        'object_type': 'multiple_histology',
        'n_classes': 9,
        'image_size': 224,  # Actual: 224x224
        'n_samples': 5000,
        'topology_hypothesis': 'Tissue type dependent',
    },
    'MalariaCell': {
        'source': 'kaggle',
        'object_type': 'ring_topology',
        'n_classes': 2,
        'image_size': 142,  # Actual: ~142x148 (variable cell crops)
        'n_samples': 5000,
        'topology_hypothesis': 'H1 explicit test (parasite ring vs none)',
    },
    'IDRiD': {
        'source': 'ieee',
        'object_type': 'retinal_grading',
        'n_classes': 5,
        'image_size': 4288,  # High-res fundus
        'n_samples': 516,  # Small dataset (train + test)
        'topology_hypothesis': 'Vascular topology, severity grading like RetinaMNIST but high-res',
    },
    'PCam': {
        'source': 'github',
        'object_type': 'metastasis_detection',
        'n_classes': 2,
        'image_size': 96,
        'n_samples': 5000,
        'topology_hypothesis': 'Tumor vs normal tissue patches, H0/H1 mixed',
    },
    'LC25000': {
        'source': 'kaggle',
        'object_type': 'lung_colon_histopathology',
        'n_classes': 5,
        'image_size': 768,
        'n_samples': 5000,
        'topology_hypothesis': 'Glandular architecture, tissue type dependent',
    },
    'SIPaKMeD': {
        'source': 'kaggle',
        'object_type': 'cervical_cytology',
        'n_classes': 5,
        'image_size': 'variable',
        'n_samples': 4049,  # Cropped single-cell images
        'topology_hypothesis': 'Cell morphology, nuclear shape, H0 dominant',
    },
    'AML_Cytomorphology': {
        'source': 'tcia',
        'object_type': 'blood_cell_morphology',
        'n_classes': 15,
        'image_size': 'variable',
        'n_samples': 5000,
        'topology_hypothesis': 'Cell morphology, nuclear shape, blast vs normal',
    },
    'APTOS2019': {
        'source': 'kaggle',
        'object_type': 'retinal_grading',
        'n_classes': 5,
        'image_size': 1050,  # Actual: ~1050x1050 -> resized to 1024 for PH
        'n_samples': 3296,  # Train + val combined
        'topology_hypothesis': 'Vascular topology, full-res fundus DR grading',
    },
    'GasHisSDB': {
        'source': 'figshare',
        'object_type': 'gastric_histopathology',
        'n_classes': 2,
        'image_size': 160,
        'n_samples': 5000,
        'topology_hypothesis': 'Tissue architecture, normal vs abnormal',
    },
    'Chaoyang': {
        'source': 'google_drive',
        'object_type': 'colorectal_pathology',
        'n_classes': 4,
        'image_size': 512,
        'n_samples': 5000,
        'topology_hypothesis': 'Glandular architecture, similar to NCT_CRC_HE but noisy labels',
    },
}

ALL_DATASETS = list(DATASETS.keys())
N_DATASETS = len(DATASETS)  # 26

# Datasets needing ECOC wrapper (>10 classes for TabPFN)
MANY_CLASS_DATASETS = ['OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST', 'MURA', 'AML_Cytomorphology']

# =============================================================================
# 3. DESCRIPTORS CONFIGURATION (15 total: 13 TDA + 2 baseline)
# =============================================================================
DESCRIPTORS = {
    # =========================================================================
    # TDA DESCRIPTORS (13)
    # =========================================================================
    'persistence_statistics': {
        'type': 'TDA',
        'dim': 28,
        'library': 'custom',
        'params': {},
        'reference': 'Custom: 14 statistics × 2 homology dimensions',
    },
    'persistence_image': {
        'type': 'TDA',
        'dim': 800,
        'library': 'giotto-tda',
        'params': {'sigma': 0.1, 'n_bins': 20},
        'reference': 'Adams et al. 2017, JMLR',
    },
    'persistence_landscapes': {
        'type': 'TDA',
        'dim': 400,
        'library': 'giotto-tda',
        'params': {'n_layers': 4, 'n_bins': 100},
        'reference': 'Bubenik 2015, JMLR',
    },
    'persistence_silhouette': {
        'type': 'TDA',
        'dim': 200,
        'library': 'giotto-tda',
        'params': {'power': 1.0, 'n_bins': 100},
        'reference': 'Chazal et al. 2014',
    },
    'betti_curves': {
        'type': 'TDA',
        'dim': 200,
        'library': 'giotto-tda',
        'params': {'n_bins': 100},
        'reference': 'Classical PH vectorization',
    },
    'persistence_entropy': {
        'type': 'TDA',
        'dim': 200,
        'library': 'custom',
        'params': {'mode': 'vector', 'n_bins': 100},
        'reference': 'Chazal et al. 2009 (expanded to curve)',
        'notes': 'Custom entropy curve: entropy at each filtration threshold, NOT giotto-tda scalar',
    },
    'persistence_codebook': {
        'type': 'TDA',
        'dim': 64,
        'library': 'custom',
        'params': {'n_codewords': 64},
        'reference': 'Zielgelmeier et al. 2017 (BoW)',
        'leakage_warning': True,  # Must fit inside CV folds
    },
    'tropical_coordinates': {
        'type': 'TDA',
        'dim': 40,
        'library': 'TDAvec',
        'params': {},
        'reference': 'Kalisnik 2019, FoCM',
    },
    'ATOL': {
        'type': 'TDA',
        'dim': 32,
        'library': 'gudhi',
        'params': {'n_clusters': 16},
        'reference': 'Royer et al. 2021, AISTATS',
        'leakage_warning': True,  # Must fit inside CV folds
    },
    'template_functions': {
        'type': 'TDA',
        'dim': 50,
        'library': 'custom',
        'params': {'n_templates': 25},
        'reference': 'Perea, Munch, Khasawneh 2022, FoCM',
    },
    'minkowski_functionals': {
        'type': 'TDA',
        'dim': 30,
        'library': 'quantimpy',
        'params': {'n_thresholds': 10, 'adaptive': True},
        'reference': 'Integral geometry, Legland et al. 2007',
        'notes': 'Adaptive thresholds (percentile-based) to avoid sparse zeros',
    },
    'euler_characteristic_curve': {
        'type': 'TDA',
        'dim': 100,
        'library': 'custom',
        'params': {'n_bins': 100},
        'reference': 'Classical: chi(t) = beta_0(t) - beta_1(t)',
    },
    'euler_characteristic_transform': {
        'type': 'TDA',
        'dim': 640,
        'library': 'custom',
        'params': {'n_directions': 32, 'n_thresholds': 20},
        'reference': 'Turner et al. 2014',
    },
    # =========================================================================
    # BASELINE DESCRIPTORS (2)
    # =========================================================================
    'lbp_texture': {
        'type': 'baseline',
        'dim': 54,
        'library': 'skimage',
        'params': {'radius': [1, 2, 3], 'n_points': [8, 16, 24]},
        'reference': 'Local Binary Patterns',
    },
    'edge_histogram': {
        'type': 'baseline',
        'dim': 80,
        'library': 'opencv',
        'params': {'n_orientation_bins': 8, 'n_spatial_cells': 10},
        'reference': 'Canny + orientation histogram',
    },
}

ALL_DESCRIPTORS = list(DESCRIPTORS.keys())
N_DESCRIPTORS = len(DESCRIPTORS)  # 15
N_TDA = len([d for d in DESCRIPTORS.values() if d['type'] == 'TDA'])  # 13
N_BASELINE = len([d for d in DESCRIPTORS.values() if d['type'] == 'baseline'])  # 2

# Expected dimensions for validation
EXPECTED_DIMS = {name: cfg['dim'] for name, cfg in DESCRIPTORS.items()}
TOTAL_DIMS = sum(EXPECTED_DIMS.values())  # 2894D across all 15

# Descriptors that need fitting inside CV folds (avoid leakage)
LEARNED_DESCRIPTORS = ['ATOL', 'persistence_codebook']

# PH-based descriptors (need persistence computation first)
PH_BASED_DESCRIPTORS = [
    'persistence_statistics', 'persistence_image', 'persistence_landscapes',
    'persistence_silhouette', 'betti_curves', 'persistence_entropy',
    'persistence_codebook', 'tropical_coordinates', 'ATOL', 'template_functions',
]

# Image-based descriptors (no PH needed)
IMAGE_BASED_DESCRIPTORS = [
    'euler_characteristic_curve', 'euler_characteristic_transform',
    'minkowski_functionals', 'lbp_texture', 'edge_histogram',
]

# =============================================================================
# 4. PERSISTENT HOMOLOGY CONFIGURATION
# =============================================================================
PH_CONFIG = {
    'filtration_type': 'cubical',
    'homology_dimensions': [0, 1],
    'image_preprocessing': {
        'resize': 224,          # Use native 224x224 resolution (GUDHI + joblib is fast enough)
        'grayscale': True,
        'normalize': True,      # Normalize to [0, 1]
    },
    'library': 'gudhi',         # GUDHI CubicalComplex + joblib parallelization
    'parallelization': 'joblib',  # n_jobs=-1
}

# =============================================================================
# 5. EVALUATION CONFIGURATION
# =============================================================================
EVALUATION = {
    'primary_metric': 'balanced_accuracy',
    'metrics': [
        'accuracy',
        'balanced_accuracy',
        'f1_macro',
        'f1_weighted',
        'precision_macro',
        'recall_macro',
        'roc_auc_ovr',
        'cohen_kappa',
    ],
    'cv_strategy': 'stratified_kfold',
    'n_folds': 5,
    'shuffle': True,
    'statistical_tests': ['paired_ttest', 'wilcoxon'],
    'confidence_level': 0.95,
}

# =============================================================================
# 6. FEATURE PREPROCESSING
# =============================================================================
PREPROCESSING = {
    'normalization': 'standard',    # StandardScaler
    'handle_nan': 'median',         # SimpleImputer(strategy='median')
    'handle_inf': 'clip',
    'dim_reduction': {
        'enabled': False,           # NO PCA - use native dimensions
    },
}

# =============================================================================
# 7. EXPERIMENT EXECUTION
# =============================================================================
EXECUTION = {
    'seeds': [42],
    'n_runs': 1,
    'n_jobs': -1,
    'use_gpu': True,
    'gpu_device': 0,
    'batch_size': 1000,
    'cache_features': True,
    'cache_dir': str(RESULTS_PATH / 'cache' / 'exp1'),
    'output_dir': str(RESULTS_PATH / 'exp1'),
    'experiment_name': 'benchmark3_exp1_baseline',
    'save_predictions': True,
    'save_features': True,
    'log_level': 'INFO',
}

# =============================================================================
# 8. EXPERIMENT MATRIX
# =============================================================================
EXPERIMENT_MATRIX = {
    'n_datasets': N_DATASETS,                                   # 13
    'n_descriptors': N_DESCRIPTORS,                             # 15
    'n_classifiers': len(ACTIVE_CLASSIFIERS),                   # 4
    'n_folds': EVALUATION['n_folds'],                           # 5
    'n_runs': EXECUTION['n_runs'],                              # 5
    'n_configurations': N_DATASETS * N_DESCRIPTORS * len(ACTIVE_CLASSIFIERS),  # 780
    'total_experiments': (N_DATASETS * N_DESCRIPTORS * len(ACTIVE_CLASSIFIERS)
                         * EVALUATION['n_folds'] * EXECUTION['n_runs']),  # 19,500
}

# =============================================================================
# 9. HELPER FUNCTIONS
# =============================================================================

def get_device():
    """Get best available device."""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_classifier(name: str, seed: int = 42, dataset_name: Optional[str] = None, n_classes: int = 2):
    """Instantiate a classifier by name with a given seed.

    Args:
        name: Classifier name (TabPFN, RandomForest, XGBoost, KNN)
        seed: Random seed for reproducibility
        dataset_name: If provided and dataset is in MANY_CLASS_DATASETS,
                      wraps TabPFN with ECOC (OutputCodeClassifier) for >10 classes.
        n_classes: Number of classes (used to set XGBoost objective).
    """
    cfg = CLASSIFIERS[name]
    params = cfg['params'].copy()

    if 'random_state' in params:
        params['random_state'] = seed

    if name == 'TabPFN':
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(**params)

        # Wrap with ECOC for many-class datasets (TabPFN supports ≤10 classes natively)
        if dataset_name and dataset_name in MANY_CLASS_DATASETS:
            from sklearn.multiclass import OutputCodeClassifier
            clf = OutputCodeClassifier(
                estimator=clf,
                code_size=1.5,
                random_state=seed,
                n_jobs=-1,
            )
        return clf
    elif name == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif name == 'XGBoost':
        from xgboost import XGBClassifier
        # Set objective based on number of classes
        if n_classes <= 2:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        else:
            params['num_class'] = n_classes
        return XGBClassifier(**params)
    elif name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def get_preprocessing_pipeline():
    """Create sklearn preprocessing pipeline (StandardScaler + NaN handling)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    return Pipeline(steps)


def get_output_dir(experiment_name: str = 'exp1') -> Path:
    """Get output directory for an experiment."""
    output_dir = RESULTS_PATH / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# 10. SUMMARY
# =============================================================================

SUMMARY = """
======================================================================
           Benchmark3 Experiment 1: Configuration
======================================================================
 DATASETS:        26  (11 MedMNIST + 15 external)
 DESCRIPTORS:     15  (13 TDA + 2 baseline)
 CLASSIFIERS:      4  (TabPFN, RandomForest, XGBoost, KNN)
 CV FOLDS:         5  (Stratified K-Fold)
 RUNS:             1  (seed: 42)
----------------------------------------------------------------------
 CONFIGURATIONS: 780  (13 x 15 x 4)
 TOTAL EXPS:   3,900  (780 x 5 folds x 1 run)
----------------------------------------------------------------------
 PRIMARY METRIC:   balanced_accuracy
 PREPROCESSING:    StandardScaler (NO PCA - native dimensions)
 PH RESOLUTION:    224x224 (native, GUDHI + joblib parallelization)
 GPU ENABLED:      True
======================================================================
"""

if __name__ == '__main__':
    print(SUMMARY)

    print("\nClassifiers:")
    for clf in ACTIVE_CLASSIFIERS:
        print(f"  - {clf}")

    print(f"\nDescriptors ({N_TDA} TDA + {N_BASELINE} baseline):")
    for name, cfg in DESCRIPTORS.items():
        print(f"  - {name}: {cfg['dim']}D ({cfg['type']})")

    print(f"\nDatasets ({N_DATASETS}):")
    for name, cfg in DATASETS.items():
        print(f"  - {name}: {cfg['object_type']} ({cfg['n_classes']} classes, n={cfg['n_samples']})")

    print(f"\nTotal feature dimensions: {TOTAL_DIMS}D")
