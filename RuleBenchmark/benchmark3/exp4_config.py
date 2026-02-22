"""
Experiment 4: Hyperparameter & Dimension Selection Study Configuration.

This configuration file defines:
1. Dimension configurations (small/medium/large) for all 15 descriptors
2. Parameter grids for hyperparameter sensitivity study
3. Dataset groupings by object type
4. Classifier and evaluation settings
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "benchmark3" / "exp4"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Import dataset paths from main config
from RuleBenchmark.benchmark3.config import (
    MEDMNIST_PATH, DATASETS, CLASSIFIERS,
    get_classifier, get_preprocessing_pipeline,
)

# =============================================================================
# 1. DIMENSION CONFIGURATIONS (Small / Medium / Large)
# =============================================================================
DIMENSION_CONFIGS = {
    'persistence_image': {
        'small': {'resolution': 10, 'sigma': 0.1},           # 10×10×2 = 200D
        'medium': {'resolution': 20, 'sigma': 0.1},          # 20×20×2 = 800D
        'large': {'resolution': 30, 'sigma': 0.1},           # 30×30×2 = 1800D
    },
    'persistence_landscapes': {
        'small': {'n_layers': 2, 'n_bins': 50, 'combine_dims': True},   # 2×50 = 100D
        'medium': {'n_layers': 4, 'n_bins': 100, 'combine_dims': True}, # 4×100 = 400D
        'large': {'n_layers': 8, 'n_bins': 100, 'combine_dims': True},  # 8×100 = 800D
    },
    'persistence_silhouette': {
        'small': {'n_bins': 50, 'power': 1.0},    # 50×2 = 100D
        'medium': {'n_bins': 100, 'power': 1.0},  # 100×2 = 200D
        'large': {'n_bins': 200, 'power': 1.0},   # 200×2 = 400D
    },
    'betti_curves': {
        'small': {'n_bins': 50},    # 50×2 = 100D
        'medium': {'n_bins': 100},  # 100×2 = 200D
        'large': {'n_bins': 200},   # 200×2 = 400D
    },
    'persistence_entropy': {
        'small': {'mode': 'vector', 'n_bins': 50},    # 50×2 = 100D
        'medium': {'mode': 'vector', 'n_bins': 100},  # 100×2 = 200D
        'large': {'mode': 'vector', 'n_bins': 200},   # 200×2 = 400D
    },
    'persistence_codebook': {
        'small': {'codebook_size': 16},   # 16×2 = 32D
        'medium': {'codebook_size': 32},  # 32×2 = 64D
        'large': {'codebook_size': 64},   # 64×2 = 128D
    },
    'tropical_coordinates': {
        'small': {'max_terms': 3},   # 3×4×2 = 24D
        'medium': {'max_terms': 5},  # 5×4×2 = 40D
        'large': {'max_terms': 10},  # 10×4×2 = 80D
    },
    'ATOL': {
        'small': {'n_centers': 8},    # 8×2 = 16D
        'medium': {'n_centers': 16},  # 16×2 = 32D
        'large': {'n_centers': 32},   # 32×2 = 64D
    },
    'template_functions': {
        'small': {'n_templates': 9},    # 9×2 = 18D
        'medium': {'n_templates': 25},  # 25×2 = 50D
        'large': {'n_templates': 49},   # 49×2 = 98D
    },
    'minkowski_functionals': {
        'small': {'n_thresholds': 5, 'adaptive': True},   # 5×3 = 15D
        'medium': {'n_thresholds': 10, 'adaptive': True}, # 10×3 = 30D
        'large': {'n_thresholds': 20, 'adaptive': True},  # 20×3 = 60D
    },
    'euler_characteristic_curve': {
        'small': {'resolution': 50},    # 50D
        'medium': {'resolution': 100},  # 100D
        'large': {'resolution': 200},   # 200D
    },
    'euler_characteristic_transform': {
        'small': {'n_directions': 8, 'n_heights': 20},    # 8×20 = 160D
        'medium': {'n_directions': 32, 'n_heights': 20},  # 32×20 = 640D
        'large': {'n_directions': 32, 'n_heights': 40},   # 32×40 = 1280D
    },
    'persistence_statistics': {
        'small': {'subset': 'basic'},     # 14×2 = 28D
        'medium': {'subset': 'extended'}, # 21×2 = 42D
        'large': {'subset': 'full'},      # 31×2 = 62D
    },
    'lbp_texture': {
        'small': {'scales': [(8, 1.0)]},                            # 10D (uniform)
        'medium': {'scales': [(8, 1.0), (16, 2.0), (24, 3.0)]},     # 10+18+26 = 54D
        'large': {'scales': [(8, 1.0), (16, 2.0), (24, 3.0), (8, 2.0)]},  # +10 = 64D
    },
    'edge_histogram': {
        'small': {'n_orientation_bins': 4, 'n_spatial_cells': 8},   # 4×8 = 32D
        'medium': {'n_orientation_bins': 8, 'n_spatial_cells': 10}, # 8×10 = 80D
        'large': {'n_orientation_bins': 8, 'n_spatial_cells': 16},  # 8×16 = 128D
    },
}

# Expected dimensions for each configuration
EXPECTED_DIMS = {
    'persistence_image': {'small': 200, 'medium': 800, 'large': 1800},
    'persistence_landscapes': {'small': 100, 'medium': 400, 'large': 800},
    'persistence_silhouette': {'small': 100, 'medium': 200, 'large': 400},
    'betti_curves': {'small': 100, 'medium': 200, 'large': 400},
    'persistence_entropy': {'small': 100, 'medium': 200, 'large': 400},
    'persistence_codebook': {'small': 32, 'medium': 64, 'large': 128},
    'tropical_coordinates': {'small': 24, 'medium': 40, 'large': 80},
    'ATOL': {'small': 16, 'medium': 32, 'large': 64},
    'template_functions': {'small': 18, 'medium': 50, 'large': 98},
    'minkowski_functionals': {'small': 15, 'medium': 30, 'large': 60},
    'euler_characteristic_curve': {'small': 50, 'medium': 100, 'large': 200},
    'euler_characteristic_transform': {'small': 160, 'medium': 640, 'large': 1280},
    'persistence_statistics': {'small': 28, 'medium': 42, 'large': 62},
    'lbp_texture': {'small': 10, 'medium': 54, 'large': 64},
    'edge_histogram': {'small': 32, 'medium': 80, 'large': 128},
}

# =============================================================================
# 2. PARAMETER GRIDS FOR HYPERPARAMETER SENSITIVITY STUDY
# =============================================================================
PARAM_GRIDS = {
    'persistence_image': {
        'resolution': [10, 15, 20, 25, 30],
        'sigma': [0.05, 0.1, 0.15, 0.2, 0.25],
        'weight_function': ['linear', 'squared', 'const'],
    },  # 75 configs

    'persistence_landscapes': {
        'n_layers': [2, 3, 4, 5, 7, 10],
        'n_bins': [50, 75, 100, 150],
    },  # 24 configs

    'betti_curves': {
        'n_bins': [50, 75, 100, 150, 200],
        'normalize': [True, False],
    },  # 10 configs

    'persistence_silhouette': {
        'n_bins': [50, 75, 100, 150],
        'power': [0.5, 1.0, 1.5, 2.0],
    },  # 16 configs

    'persistence_entropy': {
        'n_bins': [50, 75, 100, 150, 200],
    },  # 5 configs

    'ATOL': {
        'n_centers': [4, 8, 12, 16, 24, 32],
    },  # 6 configs

    'persistence_codebook': {
        'codebook_size': [16, 32, 48, 64, 96],
    },  # 5 configs

    'tropical_coordinates': {
        'max_terms': [3, 5, 7, 10],
    },  # 4 configs

    'template_functions': {
        'n_templates': [9, 16, 25, 36, 49],
        'template_type': ['tent', 'gaussian'],
    },  # 10 configs

    'minkowski_functionals': {
        'n_thresholds': [5, 10, 15, 20],
        'adaptive': [True, False],
    },  # 8 configs

    'euler_characteristic_curve': {
        'resolution': [25, 50, 75, 100, 150, 200],
    },  # 6 configs

    'euler_characteristic_transform': {
        'n_directions': [8, 16, 32, 64],
        'n_heights': [10, 20, 30, 40],
    },  # 16 configs

    'persistence_statistics': {
        'subset': ['basic', 'extended', 'full'],
    },  # 3 configs

    'lbp_texture': {
        'P': [8, 16, 24],
        'R': [1.0, 2.0, 3.0],
    },  # 9 configs

    'edge_histogram': {
        'n_orientation_bins': [4, 8, 16],
        'n_spatial_cells': [4, 9, 16, 25],
    },  # 12 configs
}

# =============================================================================
# 3. DATASET GROUPINGS BY OBJECT TYPE
# =============================================================================
OBJECT_TYPE_DATASETS = {
    'discrete_cells': ['BloodMNIST', 'TissueMNIST', 'MalariaCell'],
    'glands_lumens': ['PathMNIST', 'BreakHis', 'NCT_CRC_HE'],
    'vessel_trees': ['RetinaMNIST', 'APTOS2019'],  # IDRiD excluded (too small)
    'surface_lesions': ['DermaMNIST', 'ISIC2019', 'GasHisSDB'],
    'organ_shape': ['OrganAMNIST', 'BrainTumorMRI'],
}

# Flatten for convenience
ALL_OBJECT_TYPES = list(OBJECT_TYPE_DATASETS.keys())

# Reverse mapping: dataset -> object_type
DATASET_OBJECT_TYPE = {}
for obj_type, datasets in OBJECT_TYPE_DATASETS.items():
    for dataset in datasets:
        DATASET_OBJECT_TYPE[dataset] = obj_type

# =============================================================================
# 4. CLASSIFIER CONFIGURATION (TabPFN 2.5)
# =============================================================================
CLASSIFIER_CONFIG = {
    'name': 'TabPFN',
    'params': {
        'device': 'cuda',
        'n_estimators': 32,
    },
    'ecoc_threshold': 10,  # Use ECOC for >10 classes
}

# =============================================================================
# 5. EVALUATION SETTINGS
# =============================================================================
EVALUATION = {
    'n_folds': 5,
    'seed': 42,
    'metric': 'balanced_accuracy',
    'n_samples_per_dataset': 5000,  # Full run
    'n_samples_trial': 50,          # Trial run
}

# =============================================================================
# 6. DESCRIPTOR LISTS
# =============================================================================
ALL_DESCRIPTORS = list(DIMENSION_CONFIGS.keys())

# PH-based descriptors (need persistence computation)
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

# Learned descriptors (need CV-scoped fitting)
LEARNED_DESCRIPTORS = ['ATOL', 'persistence_codebook']

# =============================================================================
# 7. TRIAL DATASETS (quick tests)
# =============================================================================
TRIAL_DATASETS = {
    'discrete_cells': ['BloodMNIST'],
    'glands_lumens': ['PathMNIST'],
    'vessel_trees': ['RetinaMNIST'],
    'surface_lesions': ['DermaMNIST'],
    'organ_shape': ['OrganAMNIST'],
}

# =============================================================================
# 8. MEMORY-SAFE FINE-GRAINED DIMENSION SEARCH (40GB GPU Limit)
# =============================================================================

MEMORY_SAFE_DIMENSION_SEARCH = {
    # === LOW MEMORY (run on any GPU) ===
    'persistence_statistics': {
        'control_param': 'subset',
        'values': ['basic', 'extended', 'full'],
        'dimensions': [28, 42, 62],
        'memory_tier': 'low',
        'fixed': {},
    },
    'tropical_coordinates': {
        'control_param': 'max_terms',
        'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
        'dimensions': [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160],
        'memory_tier': 'low',
        'fixed': {},
    },
    'lbp_texture': {
        'control_param': 'scales',
        'values': [
            [(8, 1.0), (16, 2.0), (24, 3.0), (32, 4.0), (40, 5.0), (48, 6.0)],
            [(8, 1.0), (16, 2.0), (24, 3.0), (32, 4.0), (40, 5.0), (48, 6.0), (56, 7.0)],
            [(8, 1.0), (16, 2.0), (24, 3.0), (32, 4.0), (40, 5.0), (48, 6.0), (56, 7.0), (64, 8.0)],
        ],
        'dimensions': [180, 238, 304],
        'memory_tier': 'low',
        'fixed': {'method': 'uniform'},
    },
    'euler_characteristic_curve': {
        'control_param': 'resolution',
        'values': [10, 20, 30, 40, 50, 60, 75, 90, 100, 120, 140, 160, 180, 200, 250, 300],
        'dimensions': [10, 20, 30, 40, 50, 60, 75, 90, 100, 120, 140, 160, 180, 200, 250, 300],
        'memory_tier': 'low',
        'fixed': {},
    },

    # === MEDIUM MEMORY (V100/A40 safe) ===
    'ATOL': {
        'control_param': 'n_centers',
        'values': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40],
        'dimensions': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72, 80],
        'memory_tier': 'medium',
        'fixed': {},
    },
    'persistence_codebook': {
        'control_param': 'codebook_size',
        'values': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 96, 112],
        'dimensions': [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 224],
        'memory_tier': 'medium',
        'fixed': {},
    },
    'minkowski_functionals': {
        'control_param': 'n_thresholds',
        'values': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40],
        'dimensions': [6, 9, 12, 15, 18, 21, 24, 27, 30, 36, 42, 48, 54, 60, 75, 90, 105, 120],
        'memory_tier': 'medium',
        'fixed': {'adaptive': True},
    },
    'edge_histogram': {
        'control_param': 'n_spatial_cells',
        'values': [40, 44, 48, 52, 56, 60, 64],
        'dimensions': [320, 352, 384, 416, 448, 480, 512],
        'memory_tier': 'medium',
        'fixed': {'n_orientation_bins': 8},
    },
    'betti_curves': {
        'control_param': 'n_bins',
        'values': [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
        'dimensions': [40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400],
        'memory_tier': 'medium',
        'fixed': {'normalize': False},
    },
    'persistence_silhouette': {
        'control_param': 'n_bins',
        'values': [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
        'dimensions': [40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400],
        'memory_tier': 'medium',
        'fixed': {'power': 1.0},
    },
    'persistence_entropy': {
        'control_param': 'n_bins',
        'values': [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300],
        'dimensions': [40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 500, 600],
        'memory_tier': 'medium',
        'fixed': {'mode': 'vector', 'normalized': True},
    },
    'template_functions': {
        'control_param': 'n_templates',
        'values': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196],
        'dimensions': [2, 8, 18, 32, 50, 72, 98, 128, 162, 200, 242, 288, 338, 392],
        'memory_tier': 'medium',
        'fixed': {'template_type': 'tent'},
    },

    # === HIGH MEMORY (A40 only) ===
    'persistence_landscapes': {
        'control_param': 'n_layers',
        'values': [15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40],
        'dimensions': [1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200, 3500, 3800, 4000],
        'memory_tier': 'high',
        'fixed': {'n_bins': 100, 'combine_dims': True},
    },
    'persistence_image': {
        'control_param': 'resolution',
        'values': [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        'dimensions': [72, 128, 200, 288, 392, 512, 648, 800, 968, 1152, 1352, 1568, 1800],
        'memory_tier': 'high',
        'fixed': {'sigma': 0.1, 'weight_function': 'linear'},
    },
    'euler_characteristic_transform': {
        'control_param': 'n_directions',
        'values': [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64],
        'dimensions': [80, 160, 240, 320, 400, 480, 560, 640, 800, 960, 1120, 1280],
        'memory_tier': 'high',
        'fixed': {'n_heights': 20},
    },
}

# =============================================================================
# 9. COMPLETE PARAMETER COVERAGE (Including Hidden Params)
# =============================================================================

ALL_PARAMETERS = {
    'persistence_image': {
        'dimension_param': 'resolution',
        'coupled_params': ['sigma', 'weight_function'],
        'hidden_params': [],
        'stability_params': ['sigma'],
        'param_grid': {
            'resolution': [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            'sigma': [0.05, 0.1, 0.15, 0.2, 0.25],
            'weight_function': ['linear', 'squared', 'const'],
        },
    },
    'persistence_landscapes': {
        'dimension_param': 'n_layers',
        'coupled_params': ['n_bins'],
        'hidden_params': ['combine_dims'],
        'stability_params': ['n_bins'],
        'param_grid': {
            'n_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            'n_bins': [50, 75, 100, 150],
            'combine_dims': [True, False],
        },
    },
    'persistence_silhouette': {
        'dimension_param': 'n_bins',
        'coupled_params': ['power'],
        'hidden_params': [],
        'stability_params': ['power'],
        'param_grid': {
            'n_bins': [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
            'power': [0.5, 1.0, 1.5, 2.0],
        },
    },
    'betti_curves': {
        'dimension_param': 'n_bins',
        'coupled_params': ['normalize'],
        'hidden_params': [],
        'stability_params': ['normalize'],
        'param_grid': {
            'n_bins': [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
            'normalize': [True, False],
        },
    },
    'persistence_entropy': {
        'dimension_param': 'n_bins',
        'coupled_params': ['mode', 'normalized'],
        'hidden_params': ['entropy_bins'],
        'stability_params': ['mode', 'normalized'],
        'param_grid': {
            'n_bins': [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
            'mode': ['scalar', 'vector'],
            'normalized': [True, False],
            'entropy_bins': [10, 20, 50],
        },
    },
    'persistence_codebook': {
        'dimension_param': 'codebook_size',
        'coupled_params': [],
        'hidden_params': ['random_state'],
        'stability_params': [],
        'param_grid': {
            'codebook_size': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 96, 112],
            'random_state': [42],
        },
    },
    'tropical_coordinates': {
        'dimension_param': 'max_terms',
        'coupled_params': [],
        'hidden_params': [],
        'stability_params': [],
        'param_grid': {
            'max_terms': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
        },
    },
    'ATOL': {
        'dimension_param': 'n_centers',
        'coupled_params': [],
        'hidden_params': ['random_state'],
        'stability_params': [],
        'param_grid': {
            'n_centers': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40],
            'random_state': [42],
        },
    },
    'template_functions': {
        'dimension_param': 'n_templates',
        'coupled_params': ['template_type'],
        'hidden_params': [],
        'stability_params': ['template_type'],
        'param_grid': {
            'n_templates': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196],
            'template_type': ['tent', 'gaussian', 'exponential'],
        },
    },
    'minkowski_functionals': {
        'dimension_param': 'n_thresholds',
        'coupled_params': ['adaptive'],
        'hidden_params': [],
        'stability_params': ['adaptive'],
        'param_grid': {
            'n_thresholds': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25],
            'adaptive': [True, False],
        },
    },
    'euler_characteristic_curve': {
        'dimension_param': 'resolution',
        'coupled_params': [],
        'hidden_params': [],
        'stability_params': [],
        'param_grid': {
            'resolution': [10, 20, 30, 40, 50, 60, 75, 90, 100, 120, 140, 160, 180, 200, 250, 300],
        },
    },
    'euler_characteristic_transform': {
        'dimension_param': 'n_directions',
        'coupled_params': ['n_heights'],
        'hidden_params': [],
        'stability_params': ['n_heights'],
        'param_grid': {
            'n_directions': [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64],
            'n_heights': [10, 15, 20, 30, 40],
        },
    },
    'persistence_statistics': {
        'dimension_param': 'subset',
        'coupled_params': [],
        'hidden_params': [],
        'stability_params': [],
        'param_grid': {
            'subset': ['basic', 'extended', 'full'],
        },
    },
    'lbp_texture': {
        'dimension_param': 'scales',
        'coupled_params': ['method'],
        'hidden_params': [],
        'stability_params': ['method'],
        'param_grid': {
            'scales': [[(8, 1.0)], [(8, 1.0), (16, 2.0)], [(8, 1.0), (16, 2.0), (24, 3.0)]],
            'method': ['uniform', 'default', 'ror', 'nri'],
        },
    },
    'edge_histogram': {
        'dimension_param': 'n_spatial_cells',
        'coupled_params': ['n_orientation_bins'],
        'hidden_params': [],
        'stability_params': ['n_orientation_bins'],
        'param_grid': {
            'n_spatial_cells': [2, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36],
            'n_orientation_bins': [4, 8, 12, 16],
        },
    },
}

# Descriptors suitable for binary search (unimodal accuracy curves)
BINARY_SEARCH_DESCRIPTORS = [
    'ATOL', 'persistence_codebook', 'persistence_image', 'persistence_landscapes',
    'betti_curves', 'persistence_silhouette', 'template_functions',
    'euler_characteristic_curve', 'euler_characteristic_transform',
]

# Descriptors requiring full search (monotonic or limited values)
FULL_SEARCH_DESCRIPTORS = [
    'persistence_statistics', 'lbp_texture', 'tropical_coordinates',
    'persistence_entropy', 'minkowski_functionals', 'edge_histogram',
]

# =============================================================================
# 11. COLOR HANDLING RULES BY OBJECT TYPE
# =============================================================================

COLOR_HANDLING_RULES = {
    # Object type -> recommended color handling method
    'discrete_cells': {
        'color_importance': 'high',
        'reason': 'Staining (H&E, Giemsa) carries diagnostic info',
        'recommended': 'per_channel',  # Compute PH on each RGB channel separately
        'alternatives': ['hsv_value', 'grayscale'],
    },
    'glands_lumens': {
        'color_importance': 'high',
        'reason': 'H&E staining: purple nuclei vs pink cytoplasm',
        'recommended': 'per_channel',
        'alternatives': ['he_deconvolution', 'grayscale'],
    },
    'vessel_trees': {
        'color_importance': 'medium',
        'reason': 'Red vessels on lighter background',
        'recommended': 'grayscale',  # Luminosity captures vessel contrast well
        'alternatives': ['red_channel', 'green_channel'],
    },
    'surface_lesions': {
        'color_importance': 'very_high',
        'reason': 'Melanin, erythema, color patterns are diagnostic',
        'recommended': 'per_channel',
        'alternatives': ['hsv_saturation', 'lab_ab'],
    },
    'organ_shape': {
        'color_importance': 'low',
        'reason': 'CT/MRI are inherently grayscale',
        'recommended': 'grayscale',
        'alternatives': [],
    },
}

# Color conversion methods (simplified - only meaningfully different methods)
COLOR_METHODS = {
    'grayscale': {
        'description': 'Luminosity formula: 0.299*R + 0.587*G + 0.114*B',
        'output_channels': 1,
        'formula': '0.299*R + 0.587*G + 0.114*B',
        'best_for': ['vessel_trees', 'organ_shape'],
    },
    'per_channel': {
        'description': 'Compute PH on R, G, B separately, concatenate features (3x features)',
        'output_channels': 3,
        'note': 'Captures full color information, but 3x computational cost',
        'best_for': ['discrete_cells', 'glands_lumens', 'surface_lesions'],
    },
    # Optional: hsv_saturation - only if color purity matters
    # 'hsv_saturation': {
    #     'description': 'S channel from HSV (color purity/saturation)',
    #     'output_channels': 1,
    #     'note': 'Different from intensity - captures how "colorful" pixels are',
    #     'best_for': ['surface_lesions'],
    # },
}

# Default color methods to test (simplified)
DEFAULT_COLOR_METHODS = ['grayscale', 'per_channel']

# Representative descriptors for color experiments (fast, representative)
COLOR_EXPERIMENT_DESCRIPTORS = [
    'persistence_image',      # Standard, sensitive to diagram changes
    'betti_curves',           # Simple, fast
    'persistence_statistics', # Interpretable
    'ATOL',                   # Learned, might be sensitive to input
]

# =============================================================================
# 10. HELPER FUNCTIONS
# =============================================================================

def get_dimension(descriptor: str, level: str) -> int:
    """Get expected dimension for descriptor at given level."""
    return EXPECTED_DIMS[descriptor][level]


def get_params(descriptor: str, level: str) -> Dict[str, Any]:
    """Get parameters for descriptor at given level."""
    return DIMENSION_CONFIGS[descriptor][level].copy()


def get_object_type(dataset: str) -> Optional[str]:
    """Get object type for a dataset."""
    return DATASET_OBJECT_TYPE.get(dataset)


if __name__ == '__main__':
    print("=" * 70)
    print("  Experiment 4 Configuration Summary")
    print("=" * 70)

    print(f"\nDescriptors: {len(ALL_DESCRIPTORS)}")
    for name in ALL_DESCRIPTORS:
        dims = EXPECTED_DIMS[name]
        print(f"  {name}: {dims['small']}D / {dims['medium']}D / {dims['large']}D")

    print(f"\nObject Types: {len(ALL_OBJECT_TYPES)}")
    for obj_type, datasets in OBJECT_TYPE_DATASETS.items():
        print(f"  {obj_type}: {datasets}")

    print(f"\nParameter Grids:")
    for desc, grid in PARAM_GRIDS.items():
        n_configs = 1
        for vals in grid.values():
            n_configs *= len(vals)
        print(f"  {desc}: {n_configs} configs")

    total_configs = sum(
        np.prod([len(v) for v in grid.values()])
        for grid in PARAM_GRIDS.values()
    )
    print(f"\nTotal parameter configs: {total_configs}")
