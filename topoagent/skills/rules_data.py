"""Benchmark-validated rules from Benchmark3 (parameter tuning) and Benchmark4 (large-scale).

Dimension and parameter rules derived from Benchmark3 (exp4_final_recommendations.json).
Rankings and accuracy data from Benchmark4 (26 datasets, 15 descriptors, 6 classifiers).
All 15 descriptors including ATOL and persistence_codebook (training-based).

Source (parameters): results/benchmark3/exp4/exp4_final_recommendations.json
Source (rankings):   results/topobenchmark/assets/accuracy_lookup.json
Benchmark3: 15 descriptors x 5 object types, n=2000, 5-fold stratified CV
Benchmark4: 15 descriptors x 26 datasets x 6 classifiers, n=5000, 5-fold stratified CV
"""

from typing import Any, Dict, List, Optional, Tuple, Sequence

# =============================================================================
# All 15 Descriptors (including ATOL and persistence_codebook)
# ATOL/persistence_codebook have per-image fitting fallback for agentic mode
# =============================================================================
SUPPORTED_DESCRIPTORS = [
    # PH-based (10)
    "persistence_image",
    "persistence_landscapes",
    "betti_curves",
    "persistence_silhouette",
    "persistence_entropy",
    "persistence_statistics",
    "tropical_coordinates",
    "template_functions",
    "ATOL",
    "persistence_codebook",
    # Image-based (5)
    "minkowski_functionals",
    "euler_characteristic_curve",
    "euler_characteristic_transform",
    "edge_histogram",
    "lbp_texture",
]

# Alias for backward compatibility
ALL_DESCRIPTORS = SUPPORTED_DESCRIPTORS

# 5 Object Types from Benchmark3/Benchmark4
OBJECT_TYPES = [
    "discrete_cells",
    "glands_lumens",
    "vessel_trees",
    "surface_lesions",
    "organ_shape",
]

# =============================================================================
# DIMENSION RULES
# Maps (descriptor, object_type) -> optimal dimension parameters
# =============================================================================
DIMENSION_RULES = {
    "persistence_image": {
        "control_param": "resolution",
        "formula": "resolution^2 * 2",
        "by_object_type": {
            "discrete_cells": {"resolution": 10, "dim": 200},
            "glands_lumens": {"resolution": 26, "dim": 1352},
            "vessel_trees": {"resolution": 14, "dim": 392},
            "surface_lesions": {"resolution": 12, "dim": 288},
            "organ_shape": {"resolution": 12, "dim": 288},
        },
        "default": {"resolution": 14, "dim": 392},
    },
    "persistence_landscapes": {
        "control_param": "n_layers",
        "formula": "n_layers * 100",
        "classifier_note": "Always use XGBoost (>2000D typical)",
        "by_object_type": {
            "discrete_cells": {"n_layers": 35, "dim": 3500},
            "glands_lumens": {"n_layers": 38, "dim": 3800},
            "vessel_trees": {"n_layers": 10, "dim": 1000},
            "surface_lesions": {"n_layers": 28, "dim": 2800},
            "organ_shape": {"n_layers": 35, "dim": 3500},
        },
        "default": {"n_layers": 28, "dim": 2800},
    },
    "betti_curves": {
        "control_param": "n_bins",
        "formula": "n_bins * 2",
        "by_object_type": {
            "discrete_cells": {"n_bins": 120, "dim": 240},
            "glands_lumens": {"n_bins": 200, "dim": 400},
            "vessel_trees": {"n_bins": 140, "dim": 280},
            "surface_lesions": {"n_bins": 140, "dim": 280},
            "organ_shape": {"n_bins": 50, "dim": 100},
        },
        "default": {"n_bins": 140, "dim": 280},
    },
    "persistence_silhouette": {
        "control_param": "n_bins",
        "formula": "n_bins * 2",
        "by_object_type": {
            "discrete_cells": {"n_bins": 120, "dim": 240},
            "glands_lumens": {"n_bins": 120, "dim": 240},
            "vessel_trees": {"n_bins": 140, "dim": 280},
            "surface_lesions": {"n_bins": 180, "dim": 360},
            "organ_shape": {"n_bins": 100, "dim": 200},
        },
        "default": {"n_bins": 140, "dim": 280},
    },
    "persistence_entropy": {
        "control_param": "n_bins",
        "formula": "n_bins * 2",
        "by_object_type": {
            "discrete_cells": {"n_bins": 200, "dim": 400},
            "glands_lumens": {"n_bins": 100, "dim": 200},
            "vessel_trees": {"n_bins": 100, "dim": 200},
            "surface_lesions": {"n_bins": 40, "dim": 80},
            "organ_shape": {"n_bins": 80, "dim": 160},
        },
        "default": {"n_bins": 100, "dim": 200},
    },
    "persistence_statistics": {
        "control_param": "subset",
        "formula": "fixed 62D",
        "by_object_type": {
            "discrete_cells": {"subset": "full", "dim": 62},
            "glands_lumens": {"subset": "full", "dim": 62},
            "vessel_trees": {"subset": "full", "dim": 62},
            "surface_lesions": {"subset": "full", "dim": 62},
            "organ_shape": {"subset": "full", "dim": 62},
        },
        "default": {"subset": "full", "dim": 62},
    },
    "tropical_coordinates": {
        "control_param": "max_terms",
        "formula": "max_terms * 8",
        "by_object_type": {
            "discrete_cells": {"max_terms": 18, "dim": 144},
            "glands_lumens": {"max_terms": 30, "dim": 240},
            "vessel_trees": {"max_terms": 10, "dim": 80},
            "surface_lesions": {"max_terms": 25, "dim": 200},
            "organ_shape": {"max_terms": 25, "dim": 200},
        },
        "default": {"max_terms": 18, "dim": 144},
    },
    "template_functions": {
        "control_param": "n_templates",
        "formula": "n_templates * 2",
        "by_object_type": {
            "discrete_cells": {"n_templates": 81, "dim": 162},
            "glands_lumens": {"n_templates": 100, "dim": 200},
            "vessel_trees": {"n_templates": 25, "dim": 50},
            "surface_lesions": {"n_templates": 16, "dim": 32},
            "organ_shape": {"n_templates": 121, "dim": 242},
        },
        "default": {"n_templates": 49, "dim": 98},
    },
    "minkowski_functionals": {
        "control_param": "n_thresholds",
        "formula": "n_thresholds * 3",
        "by_object_type": {
            "discrete_cells": {"n_thresholds": 60, "dim": 180},
            "glands_lumens": {"n_thresholds": 16, "dim": 48},
            "vessel_trees": {"n_thresholds": 45, "dim": 135},
            "surface_lesions": {"n_thresholds": 12, "dim": 36},
            "organ_shape": {"n_thresholds": 35, "dim": 105},
        },
        "default": {"n_thresholds": 25, "dim": 75},
    },
    "euler_characteristic_curve": {
        "control_param": "resolution",
        "formula": "resolution",
        "by_object_type": {
            "discrete_cells": {"resolution": 160, "dim": 160},
            "glands_lumens": {"resolution": 140, "dim": 140},
            "vessel_trees": {"resolution": 160, "dim": 160},
            "surface_lesions": {"resolution": 160, "dim": 160},
            "organ_shape": {"resolution": 60, "dim": 60},
        },
        "default": {"resolution": 160, "dim": 160},
    },
    "euler_characteristic_transform": {
        "control_param": "n_directions",
        "formula": "n_directions * n_heights",
        "by_object_type": {
            "discrete_cells": {"n_directions": 8, "n_heights": 15, "dim": 120},
            "glands_lumens": {"n_directions": 20, "n_heights": 20, "dim": 400},
            "vessel_trees": {"n_directions": 40, "n_heights": 20, "dim": 800},
            "surface_lesions": {"n_directions": 20, "n_heights": 20, "dim": 400},
            "organ_shape": {"n_directions": 12, "n_heights": 20, "dim": 240},
        },
        "default": {"n_directions": 20, "n_heights": 20, "dim": 400},
    },
    "edge_histogram": {
        "control_param": "n_spatial_cells",
        "formula": "n_spatial_cells * 8",
        "by_object_type": {
            "discrete_cells": {"n_spatial_cells": 60, "dim": 480},
            "glands_lumens": {"n_spatial_cells": 60, "dim": 480},
            "vessel_trees": {"n_spatial_cells": 48, "dim": 384},
            "surface_lesions": {"n_spatial_cells": 48, "dim": 384},
            "organ_shape": {"n_spatial_cells": 64, "dim": 512},
        },
        "default": {"n_spatial_cells": 48, "dim": 384},
    },
    "lbp_texture": {
        "control_param": "n_scales",
        "formula": "varies (8 scales=304D, 7 scales=238D, 6 scales=180D)",
        "by_object_type": {
            "discrete_cells": {"n_scales": 8, "dim": 304},
            "glands_lumens": {"n_scales": 7, "dim": 238},
            "vessel_trees": {"n_scales": 8, "dim": 304},
            "surface_lesions": {"n_scales": 8, "dim": 304},
            "organ_shape": {"n_scales": 6, "dim": 180},
        },
        "default": {"n_scales": 8, "dim": 304},
    },
    # --- Training-based descriptors (from Benchmark3 parameter tuning) ---
    "ATOL": {
        "control_param": "n_centers",
        "formula": "n_centers * 2",
        "by_object_type": {
            "discrete_cells": {"n_centers": 20, "dim": 40},
            "glands_lumens": {"n_centers": 24, "dim": 48},
            "vessel_trees": {"n_centers": 20, "dim": 40},
            "surface_lesions": {"n_centers": 18, "dim": 36},
            "organ_shape": {"n_centers": 24, "dim": 48},
        },
        "default": {"n_centers": 20, "dim": 40},
    },
    "persistence_codebook": {
        "control_param": "codebook_size",
        "formula": "codebook_size * 2",
        "by_object_type": {
            "discrete_cells": {"codebook_size": 48, "dim": 96},
            "glands_lumens": {"codebook_size": 56, "dim": 112},
            "vessel_trees": {"codebook_size": 96, "dim": 192},
            "surface_lesions": {"codebook_size": 24, "dim": 48},
            "organ_shape": {"codebook_size": 28, "dim": 56},
        },
        "default": {"codebook_size": 48, "dim": 96},
    },
}

# =============================================================================
# PARAMETER RULES
# Additional tunable parameters beyond dimension control
# =============================================================================
PARAMETER_RULES = {
    "persistence_image": {
        "tunable_params": ["sigma", "weight_function"],
        "default": {"sigma": 0.5, "weight_function": "squared"},
        "per_object_type": {
            "discrete_cells": {"sigma": 0.6, "weight_function": "squared"},
            "glands_lumens": {"sigma": 0.6, "weight_function": "linear"},
            "vessel_trees": {"sigma": 0.05, "weight_function": "squared"},
            "surface_lesions": {"sigma": 0.15, "weight_function": "linear"},
            "organ_shape": {"sigma": 0.5, "weight_function": "squared"},
        },
    },
    "persistence_landscapes": {
        "tunable_params": ["n_bins", "combine_dims"],
        "default": {"n_bins": 50, "combine_dims": False},
        "per_object_type": {
            "discrete_cells": {"n_bins": 50, "combine_dims": False},
            "glands_lumens": {"n_bins": 50, "combine_dims": False},
            "vessel_trees": {"n_bins": 50, "combine_dims": True},
            "surface_lesions": {"n_bins": 75, "combine_dims": False},
            "organ_shape": {"n_bins": 50, "combine_dims": False},
        },
    },
    "betti_curves": {
        "tunable_params": ["normalize"],
        "default": {"normalize": False},
        "per_object_type": {},  # Same for all: normalize=False
    },
    "persistence_silhouette": {
        "tunable_params": ["power"],
        "default": {"power": 0.5},
        "per_object_type": {
            "discrete_cells": {"power": 0.5},
            "glands_lumens": {"power": 0.5},
            "vessel_trees": {"power": 2.0},
            "surface_lesions": {"power": 1.0},
            "organ_shape": {"power": 0.5},
        },
    },
    "persistence_entropy": {
        "tunable_params": ["mode", "normalized"],
        "default": {"mode": "vector", "normalized": True},
        "per_object_type": {},  # Same for all
    },
    "persistence_statistics": {
        "tunable_params": [],
        "default": {},
        "per_object_type": {},  # Fixed descriptor, no tuning
    },
    "tropical_coordinates": {
        "tunable_params": [],
        "default": {},
        "per_object_type": {},
    },
    "template_functions": {
        "tunable_params": ["template_type"],
        "default": {"template_type": "tent"},
        "per_object_type": {},  # Tent wins for all
    },
    "minkowski_functionals": {
        "tunable_params": ["adaptive"],
        "default": {"adaptive": False},
        "per_object_type": {},  # adaptive=False wins for all
    },
    "euler_characteristic_curve": {
        "tunable_params": [],
        "default": {},
        "per_object_type": {},
    },
    "euler_characteristic_transform": {
        "tunable_params": ["n_heights"],
        "default": {"n_heights": 20},
        "per_object_type": {
            "discrete_cells": {"n_heights": 15},
            "glands_lumens": {"n_heights": 20},
            "vessel_trees": {"n_heights": 20},
            "surface_lesions": {"n_heights": 20},
            "organ_shape": {"n_heights": 20},
        },
    },
    "edge_histogram": {
        "tunable_params": ["n_orientation_bins"],
        "default": {"n_orientation_bins": 8},
        "per_object_type": {},  # 8 wins for all (DermaMNIST: 4 by noise margin)
    },
    "lbp_texture": {
        "tunable_params": ["method"],
        "default": {"method": "uniform"},
        "per_object_type": {},  # All methods produce identical results
    },
    # --- Training-based descriptors ---
    "ATOL": {
        "tunable_params": [],
        "default": {},
        "per_object_type": {},  # Centers learned from data; no additional tuning
    },
    "persistence_codebook": {
        "tunable_params": [],
        "default": {},
        "per_object_type": {},  # Codebook learned from data; no additional tuning
    },
}

# =============================================================================
# TOP PERFORMERS BY OBJECT TYPE — All 15 descriptors (Benchmark4)
# Mean best-classifier balanced accuracy across multiple datasets per type.
# Source: Benchmark4 results (26 datasets, 6 classifiers, n=5000, 5-fold CV)
#   discrete_cells: 6 datasets, glands_lumens: 6, vessel_trees: 3,
#   surface_lesions: 3, organ_shape: 8
# =============================================================================
TOP_PERFORMERS = {
    "discrete_cells": [
        {"descriptor": "ATOL", "accuracy": 0.7664},
        {"descriptor": "persistence_codebook", "accuracy": 0.7635},
        {"descriptor": "minkowski_functionals", "accuracy": 0.7605},
        {"descriptor": "template_functions", "accuracy": 0.7536},
        {"descriptor": "persistence_statistics", "accuracy": 0.7522},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.7504},
        {"descriptor": "betti_curves", "accuracy": 0.7459},
        {"descriptor": "persistence_landscapes", "accuracy": 0.7447},
        {"descriptor": "persistence_silhouette", "accuracy": 0.7282},
        {"descriptor": "persistence_entropy", "accuracy": 0.7125},
        {"descriptor": "lbp_texture", "accuracy": 0.7060},
        {"descriptor": "tropical_coordinates", "accuracy": 0.7023},
        {"descriptor": "persistence_image", "accuracy": 0.6949},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.5187},
        {"descriptor": "edge_histogram", "accuracy": 0.4887},
    ],
    "glands_lumens": [
        {"descriptor": "persistence_statistics", "accuracy": 0.9002},
        {"descriptor": "ATOL", "accuracy": 0.8998},
        {"descriptor": "persistence_codebook", "accuracy": 0.8995},
        {"descriptor": "minkowski_functionals", "accuracy": 0.8857},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.8778},
        {"descriptor": "template_functions", "accuracy": 0.8718},
        {"descriptor": "betti_curves", "accuracy": 0.8656},
        {"descriptor": "lbp_texture", "accuracy": 0.8606},
        {"descriptor": "persistence_landscapes", "accuracy": 0.8526},
        {"descriptor": "persistence_silhouette", "accuracy": 0.8479},
        {"descriptor": "persistence_entropy", "accuracy": 0.8419},
        {"descriptor": "tropical_coordinates", "accuracy": 0.8068},
        {"descriptor": "persistence_image", "accuracy": 0.8053},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.5411},
        {"descriptor": "edge_histogram", "accuracy": 0.5182},
    ],
    "vessel_trees": [
        {"descriptor": "lbp_texture", "accuracy": 0.5225},
        {"descriptor": "persistence_statistics", "accuracy": 0.5134},
        {"descriptor": "template_functions", "accuracy": 0.4655},
        {"descriptor": "tropical_coordinates", "accuracy": 0.4600},
        {"descriptor": "persistence_landscapes", "accuracy": 0.4524},
        {"descriptor": "persistence_codebook", "accuracy": 0.4472},
        {"descriptor": "ATOL", "accuracy": 0.4424},
        {"descriptor": "minkowski_functionals", "accuracy": 0.4338},
        {"descriptor": "persistence_entropy", "accuracy": 0.4316},
        {"descriptor": "betti_curves", "accuracy": 0.4292},
        {"descriptor": "persistence_silhouette", "accuracy": 0.4267},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.4212},
        {"descriptor": "edge_histogram", "accuracy": 0.4200},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.3666},
        {"descriptor": "persistence_image", "accuracy": 0.3260},
    ],
    "surface_lesions": [
        {"descriptor": "persistence_statistics", "accuracy": 0.6301},
        {"descriptor": "ATOL", "accuracy": 0.6126},
        {"descriptor": "minkowski_functionals", "accuracy": 0.6072},
        {"descriptor": "persistence_codebook", "accuracy": 0.6045},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.5805},
        {"descriptor": "template_functions", "accuracy": 0.5659},
        {"descriptor": "betti_curves", "accuracy": 0.5601},
        {"descriptor": "lbp_texture", "accuracy": 0.5513},
        {"descriptor": "tropical_coordinates", "accuracy": 0.5257},
        {"descriptor": "persistence_silhouette", "accuracy": 0.5218},
        {"descriptor": "persistence_landscapes", "accuracy": 0.5199},
        {"descriptor": "persistence_entropy", "accuracy": 0.5040},
        {"descriptor": "persistence_image", "accuracy": 0.4713},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.4080},
        {"descriptor": "edge_histogram", "accuracy": 0.3671},
    ],
    "organ_shape": [
        {"descriptor": "ATOL", "accuracy": 0.7792},
        {"descriptor": "minkowski_functionals", "accuracy": 0.7731},
        {"descriptor": "persistence_statistics", "accuracy": 0.7621},
        {"descriptor": "lbp_texture", "accuracy": 0.7611},
        {"descriptor": "persistence_codebook", "accuracy": 0.7526},
        {"descriptor": "edge_histogram", "accuracy": 0.7399},
        {"descriptor": "template_functions", "accuracy": 0.7359},
        {"descriptor": "persistence_landscapes", "accuracy": 0.7286},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.7215},
        {"descriptor": "betti_curves", "accuracy": 0.6990},
        {"descriptor": "persistence_silhouette", "accuracy": 0.6687},
        {"descriptor": "tropical_coordinates", "accuracy": 0.6558},
        {"descriptor": "persistence_entropy", "accuracy": 0.6490},
        {"descriptor": "persistence_image", "accuracy": 0.6054},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.5911},
    ],
}

# Benchmark3 (pilot) rankings preserved for reference — single dataset per type
# (BloodMNIST, PathMNIST, RetinaMNIST, DermaMNIST, OrganAMNIST)
BENCHMARK3_TOP_PERFORMERS = {
    "discrete_cells": [
        {"descriptor": "template_functions", "accuracy": 0.957},
        {"descriptor": "ATOL", "accuracy": 0.950},
        {"descriptor": "persistence_codebook", "accuracy": 0.949},
        {"descriptor": "betti_curves", "accuracy": 0.949},
        {"descriptor": "persistence_silhouette", "accuracy": 0.942},
    ],
    "glands_lumens": [
        {"descriptor": "ATOL", "accuracy": 0.930},
        {"descriptor": "persistence_codebook", "accuracy": 0.929},
        {"descriptor": "lbp_texture", "accuracy": 0.921},
        {"descriptor": "template_functions", "accuracy": 0.903},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.876},
    ],
    "vessel_trees": [
        {"descriptor": "lbp_texture", "accuracy": 0.423},
        {"descriptor": "persistence_landscapes", "accuracy": 0.376},
        {"descriptor": "persistence_image", "accuracy": 0.374},
        {"descriptor": "template_functions", "accuracy": 0.374},
        {"descriptor": "tropical_coordinates", "accuracy": 0.364},
    ],
    "surface_lesions": [
        {"descriptor": "ATOL", "accuracy": 0.468},
        {"descriptor": "lbp_texture", "accuracy": 0.449},
        {"descriptor": "template_functions", "accuracy": 0.441},
        {"descriptor": "persistence_image", "accuracy": 0.441},
        {"descriptor": "persistence_codebook", "accuracy": 0.424},
    ],
    "organ_shape": [
        {"descriptor": "edge_histogram", "accuracy": 0.837},
        {"descriptor": "ATOL", "accuracy": 0.757},
        {"descriptor": "persistence_statistics", "accuracy": 0.741},
        {"descriptor": "lbp_texture", "accuracy": 0.737},
        {"descriptor": "persistence_codebook", "accuracy": 0.723},
    ],
}

# =============================================================================
# COLOR RULES
# =============================================================================
COLOR_RULES = {
    "per_channel": {
        "rule": "Compute PH on R,G,B separately, vectorize each, concatenate",
        "dimension_multiplier": 3,
        "benefit": {
            "discrete_cells": "+3.1% avg",
            "glands_lumens": "+3.7% avg",
            "vessel_trees": "+16.7% avg",
            "surface_lesions": "+25.2% avg",
        },
    },
    "grayscale": {
        "rule": "Standard processing, no channel splitting",
        "dimension_multiplier": 1,
    },
}

# =============================================================================
# CLASSIFIER RULES
# =============================================================================
CLASSIFIER_RULES = {
    "default": "TabPFN",
    "high_dim_threshold": 2000,
    "high_dim_classifier": "XGBoost",
    "always_xgboost": ["persistence_landscapes"],
}

# =============================================================================
# DATASET -> OBJECT TYPE MAPPING
# =============================================================================
DATASET_TO_OBJECT_TYPE = {
    # MedMNIST2
    "BloodMNIST": "discrete_cells",
    "TissueMNIST": "discrete_cells",
    "PathMNIST": "glands_lumens",
    "OCTMNIST": "organ_shape",
    "OrganAMNIST": "organ_shape",
    "RetinaMNIST": "vessel_trees",
    "PneumoniaMNIST": "organ_shape",
    "BreastMNIST": "organ_shape",
    "DermaMNIST": "surface_lesions",
    "OrganCMNIST": "organ_shape",
    "OrganSMNIST": "organ_shape",
    # External
    "ISIC2019": "surface_lesions",
    "Kvasir": "glands_lumens",
    "BrainTumorMRI": "organ_shape",
    "MURA": "organ_shape",
    "BreakHis": "glands_lumens",
    "NCT_CRC_HE": "glands_lumens",
    "MalariaCell": "discrete_cells",
    "IDRiD": "vessel_trees",
    "PCam": "discrete_cells",
    "LC25000": "glands_lumens",
    "SIPaKMeD": "discrete_cells",
    "AML_Cytomorphology": "discrete_cells",
    "APTOS2019": "vessel_trees",
    "GasHisSDB": "surface_lesions",
    "Chaoyang": "glands_lumens",
}

# Dataset color modes
DATASET_COLOR_MODE = {
    "BloodMNIST": "per_channel",
    "TissueMNIST": "grayscale",
    "PathMNIST": "per_channel",
    "OCTMNIST": "grayscale",
    "OrganAMNIST": "grayscale",
    "RetinaMNIST": "per_channel",
    "PneumoniaMNIST": "grayscale",
    "BreastMNIST": "grayscale",
    "DermaMNIST": "per_channel",
    "OrganCMNIST": "grayscale",
    "OrganSMNIST": "grayscale",
    "ISIC2019": "per_channel",
    "Kvasir": "per_channel",
    "BrainTumorMRI": "grayscale",
    "MURA": "grayscale",
    "BreakHis": "per_channel",
    "NCT_CRC_HE": "per_channel",
    "MalariaCell": "per_channel",
    "IDRiD": "per_channel",
    "PCam": "per_channel",
    "LC25000": "per_channel",
    "SIPaKMeD": "per_channel",
    "AML_Cytomorphology": "per_channel",
    "APTOS2019": "per_channel",
    "GasHisSDB": "per_channel",
    "Chaoyang": "per_channel",
}

# Keywords that map to object types (for LLM query parsing)
OBJECT_TYPE_KEYWORDS = {
    "discrete_cells": [
        "cell", "cells", "blood", "cytology", "cytomorphology",
        "malaria", "tissue", "pcam", "sipakmed", "aml",
    ],
    "glands_lumens": [
        "gland", "glands", "lumen", "lumens", "pathology", "colon",
        "colorectal", "gastric", "histology", "endoscopy", "polyp",
        "kvasir", "breakhis", "nct", "lc25000", "chaoyang",
    ],
    "vessel_trees": [
        "vessel", "vascular", "retina", "retinal", "fundus",
        "angiography", "diabetic", "retinopathy", "idrid", "aptos",
    ],
    "surface_lesions": [
        "skin", "lesion", "dermoscopy", "dermatoscopy", "melanoma",
        "mole", "keratosis", "derma", "isic", "gashis",
    ],
    "organ_shape": [
        "organ", "brain", "tumor", "chest", "xray", "x-ray",
        "pneumonia", "breast", "bone", "musculoskeletal", "mura",
        "oct", "mammography",
    ],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_optimal_params(
    descriptor: str,
    object_type: str
) -> Dict[str, Any]:
    """Get optimal parameters for a descriptor + object_type combination.

    Returns a merged dict of dimension params and tunable params.

    Args:
        descriptor: One of 13 supported descriptors
        object_type: One of 5 object types

    Returns:
        Dict with all optimal parameters (dimension + tunable)
    """
    params = {}

    # Dimension parameters
    dim_rules = DIMENSION_RULES.get(descriptor, {})
    by_type = dim_rules.get("by_object_type", {})
    dim_params = by_type.get(object_type, dim_rules.get("default", {}))
    params.update(dim_params)

    # Tunable parameters
    param_rules = PARAMETER_RULES.get(descriptor, {})
    per_type = param_rules.get("per_object_type", {})
    tunable_params = per_type.get(object_type, param_rules.get("default", {}))
    params.update(tunable_params)

    return params


def get_top_descriptors(
    object_type: str,
    n: int = 3,
    supported_only: bool = True,
) -> List[Dict[str, Any]]:
    """Get top n descriptors for an object type.

    Uses SUPPORTED_TOP_PERFORMERS (pre-filtered, Benchmark4 rankings) when
    supported_only=True, or TOP_PERFORMERS (all 15, Benchmark4 rankings)
    when supported_only=False.

    Args:
        object_type: One of 5 object types
        n: Number of top descriptors to return
        supported_only: If True (default), use supported-only rankings
            (excludes ATOL and persistence_codebook)

    Returns:
        List of dicts with 'descriptor', 'accuracy', 'dim', 'params'
    """
    if supported_only:
        source = SUPPORTED_TOP_PERFORMERS.get(object_type, [])
    else:
        source = TOP_PERFORMERS.get(object_type, [])
    result = []
    for entry in source:
        desc = entry["descriptor"]
        params = get_optimal_params(desc, object_type)
        result.append({
            "descriptor": desc,
            "accuracy": entry["accuracy"],
            "dim": params.get("dim", 0),
            "params": params,
        })
        if len(result) >= n:
            break
    return result


def get_descriptor_dim(
    descriptor: str,
    object_type: str,
    color_mode: str = "grayscale"
) -> int:
    """Get the feature dimension for a descriptor configuration.

    Args:
        descriptor: Descriptor name
        object_type: Object type
        color_mode: 'grayscale' or 'per_channel'

    Returns:
        Total feature dimension
    """
    dim_rules = DIMENSION_RULES.get(descriptor, {})
    by_type = dim_rules.get("by_object_type", {})
    params = by_type.get(object_type, dim_rules.get("default", {}))
    base_dim = params.get("dim", 0)

    if color_mode == "per_channel":
        return base_dim * 3
    return base_dim


def get_classifier(descriptor: str, object_type: str, color_mode: str = "grayscale") -> str:
    """Determine which classifier to use based on feature dimension.

    Args:
        descriptor: Descriptor name
        object_type: Object type
        color_mode: 'grayscale' or 'per_channel'

    Returns:
        Classifier name ('TabPFN' or 'XGBoost')
    """
    if descriptor in CLASSIFIER_RULES["always_xgboost"]:
        return "XGBoost"

    total_dim = get_descriptor_dim(descriptor, object_type, color_mode)
    if total_dim > CLASSIFIER_RULES["high_dim_threshold"]:
        return CLASSIFIER_RULES["high_dim_classifier"]

    return CLASSIFIER_RULES["default"]


def get_object_type_for_dataset(dataset_name: str) -> Optional[str]:
    """Look up object type for a known dataset.

    Args:
        dataset_name: Dataset name (case-sensitive)

    Returns:
        Object type string or None if unknown
    """
    return DATASET_TO_OBJECT_TYPE.get(dataset_name)


def get_color_mode_for_dataset(dataset_name: str) -> Optional[str]:
    """Look up color mode for a known dataset.

    Args:
        dataset_name: Dataset name (case-sensitive)

    Returns:
        'grayscale' or 'per_channel', or None if unknown
    """
    return DATASET_COLOR_MODE.get(dataset_name)


# =============================================================================
# DESCRIPTOR PROPERTIES — What each descriptor captures and when to use it
# =============================================================================
DESCRIPTOR_PROPERTIES = {
    "persistence_image": {
        "captures": "Gaussian-weighted density of persistence points on a 2D birth-persistence grid",
        "input": "PH diagrams",
        "best_when": "Images with spatially distributed features and both H0+H1 content",
        "why": "Preserves both feature location (birth) and importance (persistence) in stable vector",
        "weakness": "Wastes dimensions on empty H1 when only H0 matters; sigma tuning critical",
    },
    "persistence_landscapes": {
        "captures": "Piecewise-linear envelope functions summarizing persistence diagrams",
        "input": "PH diagrams",
        "best_when": "Rich persistence diagrams where statistical stability matters (noisy data)",
        "why": "Stable under noise; multiple landscape layers capture different topological scales",
        "weakness": "High-dimensional (always >2000D, needs XGBoost); overkill for simple topologies",
    },
    "betti_curves": {
        "captures": "Count of topological features alive at each filtration threshold",
        "input": "PH diagrams",
        "best_when": "Discrete structures with clear H0 components and narrow death-time distributions",
        "why": "Bin histogram directly encodes H0 death distribution; peaks reveal important scales",
        "weakness": "Poor on continuous structures (vessels, organs) where PH is noisy",
    },
    "persistence_silhouette": {
        "captures": "Weighted average of persistence landscape functions with power weighting",
        "input": "PH diagrams",
        "best_when": "Need to emphasize high-persistence features over noise (tunable power parameter)",
        "why": "Power weighting suppresses short-lived noise features; compact representation",
        "weakness": "Single weighted summary loses multi-scale information from full landscapes",
    },
    "persistence_entropy": {
        "captures": "Shannon entropy of persistence diagram as a function of filtration threshold",
        "input": "PH diagrams",
        "best_when": "Discriminating regularity vs irregularity in topological structure",
        "why": "High entropy = many similar-lifetime features (regular); low entropy = few dominant features",
        "weakness": "Entropy of noise is high, so noisy PH diagrams give misleading values",
    },
    "persistence_statistics": {
        "captures": "62 statistical summaries of persistence diagram (mean, std, entropy, etc.)",
        "input": "PH diagrams",
        "best_when": "Need low-dimensional interpretable features; fixed 62D regardless of diagram",
        "why": "Fixed dimension, fast, no tuning needed; captures distributional properties",
        "weakness": "Loses spatial/scale information; coarse summary of rich diagrams",
    },
    "tropical_coordinates": {
        "captures": "Coordinates in tropical polynomial framework for persistence diagrams",
        "input": "PH diagrams",
        "best_when": "Moderate-complexity PH diagrams with meaningful geometric structure",
        "why": "Algebraically principled; captures geometric invariants of diagrams",
        "weakness": "Less intuitive than other descriptors; performance varies by object type",
    },
    "template_functions": {
        "captures": "Point distributions in persistence diagrams via tent/Gaussian basis functions",
        "input": "PH diagrams",
        "best_when": "Many discrete objects (cells, nuclei) producing rich H0 point clouds",
        "why": "Universal approximation on diagram space; tent functions fit H0 clusters of isolated objects",
        "weakness": "Fewer templates needed for simple topologies; wastes capacity on continuous structures",
    },
    "minkowski_functionals": {
        "captures": "Area, perimeter, and Euler characteristic at multiple thresholds (no PH needed)",
        "input": "Image directly",
        "best_when": "Image geometry matters more than topology; shape/boundary features diagnostic",
        "why": "Captures geometric properties at multiple scales; complements topological descriptors",
        "weakness": "Does not capture true topological features; less discriminative for complex topology",
    },
    "euler_characteristic_curve": {
        "captures": "Euler characteristic as a function of filtration threshold (no PH needed)",
        "input": "Image directly",
        "best_when": "Global topological structure changes predictably across thresholds",
        "why": "Single curve captures how topology evolves; good for glandular/structured tissue",
        "weakness": "Collapses H0/H1 into single number at each threshold; loses individual feature info",
    },
    "euler_characteristic_transform": {
        "captures": "Directional Euler characteristic curves from multiple viewing angles (no PH needed)",
        "input": "Image directly",
        "best_when": "Shape/orientation matters; need direction-dependent topological information",
        "why": "Provably injective on shapes; captures directional structure missed by scalar summaries",
        "weakness": "High-dimensional (directions x heights); expensive for many directions",
    },
    "edge_histogram": {
        "captures": "Edge orientation distribution across spatial grid cells (no PH needed)",
        "input": "Image directly",
        "best_when": "Shape/silhouette is the key discriminator (organ boundaries, structural outlines)",
        "why": "Directly encodes boundary orientation; 8 orientations x spatial cells = silhouette fingerprint",
        "weakness": "Poor on diffuse features (skin lesions); no topological information",
    },
    "lbp_texture": {
        "captures": "Local texture patterns via multi-scale Local Binary Patterns (no PH needed)",
        "input": "Image directly",
        "best_when": "PH is noisy (thin structures, low contrast) or texture is more diagnostic than topology",
        "why": "Bypasses PH entirely; captures local edge/texture patterns robust to illumination",
        "weakness": "Misses global topological structure; purely local descriptor",
    },
    # --- Training-based descriptors ---
    "ATOL": {
        "captures": "Optimal transport distances from persistence diagram points to learned cluster centers",
        "input": "PH diagrams",
        "best_when": "Dataset has consistent topological patterns that can be captured by clustering; "
                     "especially strong on surface_lesions and glands_lumens",
        "why": "Learns data-adaptive centers via k-means, then computes weighted distances; "
               "extremely compact representation (20-24 centers = 40-48D) that focuses on the most "
               "informative regions of the persistence diagram",
        "weakness": "Requires training data to learn centers; may overfit with very few training samples; "
                    "centers learned on one dataset may not transfer well",
    },
    "persistence_codebook": {
        "captures": "Bag-of-words histogram over learned codewords from persistence diagram points",
        "input": "PH diagrams",
        "best_when": "Rich persistence diagrams with recurring point patterns across samples; "
                     "strong on discrete_cells and glands_lumens",
        "why": "Quantizes persistence points to a learned codebook, producing a histogram that captures "
               "frequency of different topological patterns; robust to outlier points",
        "weakness": "Requires training data for codebook learning; codebook size limits expressiveness; "
                    "may miss rare but important topological features",
    },
}

# =============================================================================
# REASONING CHAINS — Connect image properties to descriptor choice
# =============================================================================
REASONING_CHAINS = {
    "h0_dominant": {
        "condition": "Image contains many discrete, isolated objects (cells, particles, nuclei)",
        "reasoning": (
            "Discrete objects create rich H0 features (each object = one connected component). "
            "H1 is minimal because objects rarely form loops.  The discriminative signal is in "
            "the DISTRIBUTION of H0 birth/death times — how objects appear and merge at different "
            "intensity thresholds."
        ),
        "recommended": ["template_functions", "betti_curves", "persistence_silhouette"],
        "avoid": ["persistence_image (wastes H1 dims)", "edge_histogram (no topology)"],
    },
    "h1_important": {
        "condition": "Image contains enclosed cavities, lumens, ring structures, or loop-like boundaries",
        "reasoning": (
            "Cavities and lumens generate H1 features whose persistence encodes cavity stability. "
            "The key signal is how H1 features are distributed across scales — both the number "
            "of cavities and their persistence range matter for distinguishing tissue types."
        ),
        "recommended": ["persistence_landscapes", "persistence_silhouette", "persistence_image"],
        "avoid": ["betti_curves (H0-biased)", "template_functions (H0-focused)"],
    },
    "noisy_ph": {
        "condition": "Image has thin linear structures, low contrast, or high noise (vessels, angiography)",
        "reasoning": (
            "PH threshold sweep creates many false short-lived components in noisy images.  "
            "Most PH features are noise artifacts rather than genuine topological signal.  "
            "The discriminative information may reside in pixel-level intensity patterns "
            "rather than in the unreliable persistence diagram."
        ),
        "recommended": ["lbp_texture", "edge_histogram"],
        "avoid": ["betti_curves (noise amplified)", "persistence_entropy (entropy of noise)"],
    },
    "shape_silhouette": {
        "condition": "Classification depends on object shape/outline (organs, structural objects)",
        "reasoning": (
            "Organ boundaries have distinctive orientations and silhouettes.  The discriminative "
            "signal is in boundary geometry (curvature, orientation histogram) rather than "
            "in H0/H1 topology, because different organ types may have identical topological "
            "structure (one connected component, zero or few cavities) but very different shapes."
        ),
        "recommended": ["edge_histogram", "persistence_statistics", "lbp_texture"],
        "avoid": ["persistence_image (H1 often empty for solid shapes)"],
    },
    "color_diagnostic": {
        "condition": "RGB image where color channels carry different diagnostic information",
        "reasoning": (
            "Per-channel PH computes separate topology for R, G, B. Different channels capture different "
            "anatomy (R=vasculature, G=melanin, B=deoxygenated blood). Concatenated features provide "
            "3x information with +3-25% accuracy boost depending on object type."
        ),
        "recommended": ["Any descriptor with per_channel color mode"],
        "note": "Largest gains on surface_lesions (+25%) and vessel_trees (+17%)",
    },
}


def build_descriptor_knowledge_text(
    object_type_hint: Optional[str] = None,
    color_mode: str = "grayscale",
    include_rankings: bool = True,
    supported_only: bool = True,
) -> str:
    """Build a formatted knowledge text for LLM descriptor selection.

    Combines descriptor properties, reasoning chains, and benchmark rankings
    into a prompt-ready string that the LLM uses to make informed decisions.

    Args:
        object_type_hint: If known, include specific benchmark rankings
        color_mode: 'grayscale' or 'per_channel'
        include_rankings: Whether to include benchmark rankings
        supported_only: If True (default), exclude training-based descriptors
            (ATOL, persistence_codebook) which require training data and cannot
            work on a single image. Set to False for batch/benchmark mode where
            training data is available.

    Returns:
        Formatted knowledge string for prompt injection
    """
    # Determine which descriptors to show
    training_based = {"ATOL", "persistence_codebook"}
    if supported_only:
        show_descriptors = {k: v for k, v in DESCRIPTOR_PROPERTIES.items()
                           if k not in training_based}
        rankings_source = SUPPORTED_TOP_PERFORMERS
    else:
        show_descriptors = DESCRIPTOR_PROPERTIES
        rankings_source = TOP_PERFORMERS

    sections = []

    # Section 1: Descriptor properties table
    sections.append(f"## Descriptor Knowledge ({len(show_descriptors)} Descriptors)")
    sections.append("")
    ph_based = []
    image_based = []
    for name, props in show_descriptors.items():
        entry = f"**{name}** ({props['input']})"
        entry += f"\n  - Captures: {props['captures']}"
        entry += f"\n  - Best when: {props['best_when']}"
        entry += f"\n  - Why it works: {props['why']}"
        entry += f"\n  - Weakness: {props['weakness']}"
        if props["input"] == "PH diagrams":
            ph_based.append(entry)
        else:
            image_based.append(entry)

    sections.append("### PH-Based Descriptors (require compute_ph first)")
    for entry in ph_based:
        sections.append(entry)
        sections.append("")

    sections.append("### Image-Based Descriptors (no PH needed)")
    for entry in image_based:
        sections.append(entry)
        sections.append("")

    # Section 2: Reasoning chains
    sections.append("## Reasoning Chains: Image Properties -> Descriptor Choice")
    sections.append("")
    for chain_id, chain in REASONING_CHAINS.items():
        sections.append(f"**{chain_id}**: {chain['condition']}")
        sections.append(f"  Reasoning: {chain['reasoning']}")
        recommended = chain.get("recommended", [])
        if recommended:
            sections.append(f"  Recommended: {', '.join(recommended)}")
        avoid = chain.get("avoid", [])
        if avoid:
            sections.append(f"  Avoid: {', '.join(avoid)}")
        note = chain.get("note")
        if note:
            sections.append(f"  Note: {note}")
        sections.append("")

    # Section 3: Benchmark rankings (if object type is known)
    if include_rankings and object_type_hint and object_type_hint in rankings_source:
        sections.append(f"## Benchmark Rankings for {object_type_hint}")
        sections.append("(from 26-dataset Benchmark4 study, 6 classifiers, n=5000, 5-fold stratified CV)")
        sections.append("")
        performers = rankings_source[object_type_hint]
        for i, entry in enumerate(performers, 1):
            desc = entry["descriptor"]
            params = get_optimal_params(desc, object_type_hint)
            dim = params.get("dim", "?")
            if color_mode == "per_channel":
                dim = dim * 3 if isinstance(dim, int) else dim
            sections.append(f"  {i}. {desc} (dim={dim})")
        sections.append("")
        sections.append(
            "The top-ranked descriptor is the empirically best choice. "
            "You may override if your reasoning about the image properties suggests otherwise."
        )
    elif include_rankings:
        sections.append("## Benchmark Rankings")
        sections.append("Object type not yet identified. Rankings available for:")
        for ot in OBJECT_TYPES:
            top1 = rankings_source.get(ot, [{}])[0]
            desc = top1.get("descriptor", "?")
            sections.append(f"  - {ot}: best={desc}")
        sections.append("")

    # Section 4: Color mode note
    if color_mode == "per_channel":
        sections.append("## Color Mode: per_channel (RGB)")
        sections.append(
            "Per-channel mode computes PH on R,G,B separately and concatenates features (3x dimension). "
            "Biggest gains on surface_lesions (+25%) and vessel_trees (+17%)."
        )
        sections.append("")

    return "\n".join(sections)


# =============================================================================
# COMPLEMENTARITY DATA — Pairwise Spearman rank correlations of descriptor
# accuracy vectors across 25 datasets (all 26 minus APTOS2019 which lacks
# persistence_codebook).
#
# Low rho = descriptors capture different information = good fusion candidates.
# High rho = redundant descriptors = fusion unlikely to help.
#
# Source: accuracy_lookup.json (Benchmark4 results, 26 datasets × 15 descriptors)
# =============================================================================

PAIRWISE_SPEARMAN_RHO: Dict[Tuple[str, str], float] = {
    ("persistence_image", "persistence_landscapes"): 0.950,
    ("persistence_image", "betti_curves"): 0.949,
    ("persistence_image", "persistence_silhouette"): 0.951,
    ("persistence_image", "persistence_entropy"): 0.957,
    ("persistence_image", "persistence_statistics"): 0.945,
    ("persistence_image", "tropical_coordinates"): 0.962,
    ("persistence_image", "persistence_codebook"): 0.953,
    ("persistence_image", "ATOL"): 0.921,
    ("persistence_image", "template_functions"): 0.953,
    ("persistence_image", "minkowski_functionals"): 0.938,
    ("persistence_image", "euler_characteristic_curve"): 0.931,
    ("persistence_image", "euler_characteristic_transform"): 0.775,
    ("persistence_image", "edge_histogram"): 0.523,
    ("persistence_image", "lbp_texture"): 0.896,
    ("persistence_landscapes", "betti_curves"): 0.973,
    ("persistence_landscapes", "persistence_silhouette"): 0.983,
    ("persistence_landscapes", "persistence_entropy"): 0.983,
    ("persistence_landscapes", "persistence_statistics"): 0.953,
    ("persistence_landscapes", "tropical_coordinates"): 0.954,
    ("persistence_landscapes", "persistence_codebook"): 0.967,
    ("persistence_landscapes", "ATOL"): 0.950,
    ("persistence_landscapes", "template_functions"): 0.989,
    ("persistence_landscapes", "minkowski_functionals"): 0.948,
    ("persistence_landscapes", "euler_characteristic_curve"): 0.955,
    ("persistence_landscapes", "euler_characteristic_transform"): 0.761,
    ("persistence_landscapes", "edge_histogram"): 0.572,
    ("persistence_landscapes", "lbp_texture"): 0.931,
    ("betti_curves", "persistence_silhouette"): 0.966,
    ("betti_curves", "persistence_entropy"): 0.968,
    ("betti_curves", "persistence_statistics"): 0.965,
    ("betti_curves", "tropical_coordinates"): 0.942,
    ("betti_curves", "persistence_codebook"): 0.990,
    ("betti_curves", "ATOL"): 0.976,
    ("betti_curves", "template_functions"): 0.988,
    ("betti_curves", "minkowski_functionals"): 0.973,
    ("betti_curves", "euler_characteristic_curve"): 0.983,
    ("betti_curves", "euler_characteristic_transform"): 0.735,
    ("betti_curves", "edge_histogram"): 0.545,
    ("betti_curves", "lbp_texture"): 0.942,
    ("persistence_silhouette", "persistence_entropy"): 0.990,
    ("persistence_silhouette", "persistence_statistics"): 0.968,
    ("persistence_silhouette", "tropical_coordinates"): 0.972,
    ("persistence_silhouette", "persistence_codebook"): 0.973,
    ("persistence_silhouette", "ATOL"): 0.947,
    ("persistence_silhouette", "template_functions"): 0.989,
    ("persistence_silhouette", "minkowski_functionals"): 0.945,
    ("persistence_silhouette", "euler_characteristic_curve"): 0.958,
    ("persistence_silhouette", "euler_characteristic_transform"): 0.772,
    ("persistence_silhouette", "edge_histogram"): 0.567,
    ("persistence_silhouette", "lbp_texture"): 0.925,
    ("persistence_entropy", "persistence_statistics"): 0.958,
    ("persistence_entropy", "tropical_coordinates"): 0.973,
    ("persistence_entropy", "persistence_codebook"): 0.968,
    ("persistence_entropy", "ATOL"): 0.938,
    ("persistence_entropy", "template_functions"): 0.986,
    ("persistence_entropy", "minkowski_functionals"): 0.942,
    ("persistence_entropy", "euler_characteristic_curve"): 0.955,
    ("persistence_entropy", "euler_characteristic_transform"): 0.792,
    ("persistence_entropy", "edge_histogram"): 0.579,
    ("persistence_entropy", "lbp_texture"): 0.920,
    ("persistence_statistics", "tropical_coordinates"): 0.965,
    ("persistence_statistics", "persistence_codebook"): 0.981,
    ("persistence_statistics", "ATOL"): 0.973,
    ("persistence_statistics", "template_functions"): 0.974,
    ("persistence_statistics", "minkowski_functionals"): 0.970,
    ("persistence_statistics", "euler_characteristic_curve"): 0.968,
    ("persistence_statistics", "euler_characteristic_transform"): 0.750,
    ("persistence_statistics", "edge_histogram"): 0.482,
    ("persistence_statistics", "lbp_texture"): 0.948,
    ("tropical_coordinates", "persistence_codebook"): 0.954,
    ("tropical_coordinates", "ATOL"): 0.939,
    ("tropical_coordinates", "template_functions"): 0.965,
    ("tropical_coordinates", "minkowski_functionals"): 0.948,
    ("tropical_coordinates", "euler_characteristic_curve"): 0.955,
    ("tropical_coordinates", "euler_characteristic_transform"): 0.808,
    ("tropical_coordinates", "edge_histogram"): 0.564,
    ("tropical_coordinates", "lbp_texture"): 0.916,
    ("persistence_codebook", "ATOL"): 0.983,
    ("persistence_codebook", "template_functions"): 0.985,
    ("persistence_codebook", "minkowski_functionals"): 0.977,
    ("persistence_codebook", "euler_characteristic_curve"): 0.983,
    ("persistence_codebook", "euler_characteristic_transform"): 0.723,
    ("persistence_codebook", "edge_histogram"): 0.509,
    ("persistence_codebook", "lbp_texture"): 0.948,
    ("ATOL", "template_functions"): 0.966,
    ("ATOL", "minkowski_functionals"): 0.990,
    ("ATOL", "euler_characteristic_curve"): 0.992,
    ("ATOL", "euler_characteristic_transform"): 0.720,
    ("ATOL", "edge_histogram"): 0.518,
    ("ATOL", "lbp_texture"): 0.964,
    ("template_functions", "minkowski_functionals"): 0.962,
    ("template_functions", "euler_characteristic_curve"): 0.972,
    ("template_functions", "euler_characteristic_transform"): 0.755,
    ("template_functions", "edge_histogram"): 0.559,
    ("template_functions", "lbp_texture"): 0.940,
    ("minkowski_functionals", "euler_characteristic_curve"): 0.991,
    ("minkowski_functionals", "euler_characteristic_transform"): 0.735,
    ("minkowski_functionals", "edge_histogram"): 0.534,
    ("minkowski_functionals", "lbp_texture"): 0.949,
    ("euler_characteristic_curve", "euler_characteristic_transform"): 0.728,
    ("euler_characteristic_curve", "edge_histogram"): 0.535,
    ("euler_characteristic_curve", "lbp_texture"): 0.946,
    ("euler_characteristic_transform", "edge_histogram"): 0.826,
    ("euler_characteristic_transform", "lbp_texture"): 0.795,
    ("edge_histogram", "lbp_texture"): 0.568,
}

# Most complementary pairs: edge_histogram is the universal complement (rho ≈ 0.48-0.58)
# ECT is the second-best complement (rho ≈ 0.72-0.83)
# lbp_texture + edge_histogram is also low (rho ≈ 0.57)
MOST_COMPLEMENTARY_PAIRS = [
    ("persistence_statistics", "edge_histogram", 0.482),
    ("persistence_codebook", "edge_histogram", 0.509),
    ("ATOL", "edge_histogram", 0.518),
    ("persistence_image", "edge_histogram", 0.523),
    ("minkowski_functionals", "edge_histogram", 0.534),
    ("euler_characteristic_curve", "edge_histogram", 0.535),
    ("betti_curves", "edge_histogram", 0.545),
    ("template_functions", "edge_histogram", 0.559),
    ("tropical_coordinates", "edge_histogram", 0.564),
    ("persistence_silhouette", "edge_histogram", 0.567),
    ("edge_histogram", "lbp_texture", 0.568),
    ("persistence_landscapes", "edge_histogram", 0.572),
    ("persistence_entropy", "edge_histogram", 0.579),
    ("ATOL", "euler_characteristic_transform", 0.720),
    ("persistence_codebook", "euler_characteristic_transform", 0.723),
]

# =============================================================================
# SUPPORTED TOP PERFORMERS — Filtered for agent use (13 descriptors)
# Excludes ATOL and persistence_codebook which require training data.
# Source: benchmark4 (26 datasets, 6 classifiers)
# =============================================================================
SUPPORTED_TOP_PERFORMERS: Dict[str, List[Dict[str, Any]]] = {
    "discrete_cells": [
        {"descriptor": "minkowski_functionals", "accuracy": 0.7605},
        {"descriptor": "template_functions", "accuracy": 0.7536},
        {"descriptor": "persistence_statistics", "accuracy": 0.7522},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.7504},
        {"descriptor": "betti_curves", "accuracy": 0.7459},
        {"descriptor": "persistence_landscapes", "accuracy": 0.7447},
        {"descriptor": "persistence_silhouette", "accuracy": 0.7282},
        {"descriptor": "persistence_entropy", "accuracy": 0.7125},
        {"descriptor": "lbp_texture", "accuracy": 0.7060},
        {"descriptor": "tropical_coordinates", "accuracy": 0.7023},
        {"descriptor": "persistence_image", "accuracy": 0.6949},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.5187},
        {"descriptor": "edge_histogram", "accuracy": 0.4887},
    ],
    "glands_lumens": [
        {"descriptor": "persistence_statistics", "accuracy": 0.9002},
        {"descriptor": "minkowski_functionals", "accuracy": 0.8857},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.8778},
        {"descriptor": "template_functions", "accuracy": 0.8718},
        {"descriptor": "betti_curves", "accuracy": 0.8656},
        {"descriptor": "lbp_texture", "accuracy": 0.8606},
        {"descriptor": "persistence_landscapes", "accuracy": 0.8526},
        {"descriptor": "persistence_silhouette", "accuracy": 0.8479},
        {"descriptor": "persistence_entropy", "accuracy": 0.8419},
        {"descriptor": "tropical_coordinates", "accuracy": 0.8068},
        {"descriptor": "persistence_image", "accuracy": 0.8053},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.5411},
        {"descriptor": "edge_histogram", "accuracy": 0.5182},
    ],
    "vessel_trees": [
        {"descriptor": "lbp_texture", "accuracy": 0.5225},
        {"descriptor": "persistence_statistics", "accuracy": 0.5134},
        {"descriptor": "template_functions", "accuracy": 0.4655},
        {"descriptor": "tropical_coordinates", "accuracy": 0.4600},
        {"descriptor": "persistence_landscapes", "accuracy": 0.4524},
        {"descriptor": "minkowski_functionals", "accuracy": 0.4338},
        {"descriptor": "persistence_entropy", "accuracy": 0.4316},
        {"descriptor": "betti_curves", "accuracy": 0.4292},
        {"descriptor": "persistence_silhouette", "accuracy": 0.4267},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.4212},
        {"descriptor": "edge_histogram", "accuracy": 0.4200},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.3666},
        {"descriptor": "persistence_image", "accuracy": 0.3260},
    ],
    "surface_lesions": [
        {"descriptor": "persistence_statistics", "accuracy": 0.6301},
        {"descriptor": "minkowski_functionals", "accuracy": 0.6072},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.5805},
        {"descriptor": "template_functions", "accuracy": 0.5659},
        {"descriptor": "betti_curves", "accuracy": 0.5601},
        {"descriptor": "lbp_texture", "accuracy": 0.5513},
        {"descriptor": "tropical_coordinates", "accuracy": 0.5257},
        {"descriptor": "persistence_silhouette", "accuracy": 0.5218},
        {"descriptor": "persistence_landscapes", "accuracy": 0.5199},
        {"descriptor": "persistence_entropy", "accuracy": 0.5040},
        {"descriptor": "persistence_image", "accuracy": 0.4713},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.4080},
        {"descriptor": "edge_histogram", "accuracy": 0.3671},
    ],
    "organ_shape": [
        {"descriptor": "minkowski_functionals", "accuracy": 0.7731},
        {"descriptor": "persistence_statistics", "accuracy": 0.7621},
        {"descriptor": "lbp_texture", "accuracy": 0.7611},
        {"descriptor": "edge_histogram", "accuracy": 0.7399},
        {"descriptor": "template_functions", "accuracy": 0.7359},
        {"descriptor": "persistence_landscapes", "accuracy": 0.7286},
        {"descriptor": "euler_characteristic_curve", "accuracy": 0.7215},
        {"descriptor": "betti_curves", "accuracy": 0.6990},
        {"descriptor": "persistence_silhouette", "accuracy": 0.6687},
        {"descriptor": "tropical_coordinates", "accuracy": 0.6558},
        {"descriptor": "persistence_entropy", "accuracy": 0.6490},
        {"descriptor": "persistence_image", "accuracy": 0.6054},
        {"descriptor": "euler_characteristic_transform", "accuracy": 0.5911},
    ],
}


# =============================================================================
# PH PROFILE SIGNALS — Data-driven signals from PH statistics
# that can override the experience-based default descriptor choice.
# Thresholds calibrated on 26-dataset PH profiles.
# =============================================================================

# =============================================================================
# DESCRIPTOR EXPECTED QUALITY — Empirically-derived ranges from benchmark data
# Used by REFLECT phase to determine if feature quality is mediocre
# Sparsity = % of zero-valued features; Variance = variance of feature vector
# =============================================================================
DESCRIPTOR_EXPECTED_QUALITY = {
    "persistence_image": {
        "expected_sparsity_range": (20, 70),
        "expected_variance_range": (0.01, 100),
        "warning_sparsity": 75,
    },
    "persistence_landscapes": {
        "expected_sparsity_range": (30, 80),
        "expected_variance_range": (0.001, 50),
        "warning_sparsity": 85,
    },
    "betti_curves": {
        "expected_sparsity_range": (5, 40),
        "expected_variance_range": (1, 5_000),
        "warning_sparsity": 50,
    },
    "persistence_silhouette": {
        "expected_sparsity_range": (10, 50),
        "expected_variance_range": (0.01, 1_000),
        "warning_sparsity": 60,
    },
    "persistence_entropy": {
        "expected_sparsity_range": (5, 40),
        "expected_variance_range": (0.001, 100),
        "warning_sparsity": 50,
    },
    "persistence_statistics": {
        "expected_sparsity_range": (0, 10),
        "expected_variance_range": (100, 5_000_000),
        "warning_sparsity": 15,
    },
    "tropical_coordinates": {
        "expected_sparsity_range": (10, 50),
        "expected_variance_range": (0.01, 10_000),
        "warning_sparsity": 60,
    },
    "template_functions": {
        "expected_sparsity_range": (20, 65),
        "expected_variance_range": (10, 10_000),
        "warning_sparsity": 70,
    },
    "ATOL": {
        "expected_sparsity_range": (0, 20),
        "expected_variance_range": (0.01, 10_000),
        "warning_sparsity": 30,
    },
    "persistence_codebook": {
        "expected_sparsity_range": (5, 40),
        "expected_variance_range": (1, 50_000),
        "warning_sparsity": 50,
    },
    "minkowski_functionals": {
        "expected_sparsity_range": (0, 15),
        "expected_variance_range": (100, 1_000_000_000),
        "warning_sparsity": 20,
    },
    "euler_characteristic_curve": {
        "expected_sparsity_range": (0, 15),
        "expected_variance_range": (100, 5_000_000),
        "warning_sparsity": 20,
    },
    "euler_characteristic_transform": {
        "expected_sparsity_range": (0, 20),
        "expected_variance_range": (10, 1_000_000),
        "warning_sparsity": 30,
    },
    "edge_histogram": {
        "expected_sparsity_range": (0, 40),
        "expected_variance_range": (0.01, 50_000),
        "warning_sparsity": 50,
    },
    "lbp_texture": {
        "expected_sparsity_range": (0, 15),
        "expected_variance_range": (0.001, 100_000),
        "warning_sparsity": 20,
    },
}


def get_expected_quality(descriptor: str) -> Dict[str, Any]:
    """Get expected quality ranges for a descriptor.

    Args:
        descriptor: Descriptor name

    Returns:
        Dict with expected_sparsity_range, expected_variance_range, warning_sparsity.
        Returns permissive defaults if descriptor unknown.
    """
    return DESCRIPTOR_EXPECTED_QUALITY.get(descriptor, {
        "expected_sparsity_range": (0, 95),
        "expected_variance_range": (0, 1e10),
        "warning_sparsity": 95,
    })


def build_quality_assessment_text(
    descriptor: str,
    sparsity: float,
    variance: float,
) -> str:
    """Build quality assessment text comparing actual vs expected ranges.

    Args:
        descriptor: Descriptor name
        sparsity: Actual sparsity percentage
        variance: Actual feature variance

    Returns:
        Formatted assessment string for REFLECT prompt
    """
    expected = get_expected_quality(descriptor)
    sp_lo, sp_hi = expected["expected_sparsity_range"]
    var_lo, var_hi = expected["expected_variance_range"]
    warn_sp = expected["warning_sparsity"]

    lines = []
    lines.append(f"## Expected Quality for {descriptor}")
    lines.append(f"- Typical sparsity: {sp_lo}-{sp_hi}%")
    lines.append(f"- Typical variance: {var_lo}-{var_hi}")
    lines.append(f"- Warning threshold: sparsity > {warn_sp}% suggests poor fit")
    lines.append(f"- NOTE: Low sparsity (below typical range) means dense/rich features — this is GOOD, not bad.")
    lines.append("")

    # Sparsity assessment
    if sparsity > warn_sp:
        lines.append(f"- Sparsity: {sparsity:.1f}% ⚠ ABOVE WARNING THRESHOLD ({warn_sp}%) — many near-zero features, poor signal")
    elif sparsity > sp_hi:
        lines.append(f"- Sparsity: {sparsity:.1f}% ⚠ ABOVE TYPICAL RANGE ({sp_lo}-{sp_hi}%) — may indicate weak signal")
    elif sparsity < sp_lo:
        lines.append(f"- Sparsity: {sparsity:.1f}% ✓ BELOW typical range — denser than usual, which is FINE")
    else:
        lines.append(f"- Sparsity: {sparsity:.1f}% ✓ within expected range")

    # Variance assessment
    if variance < var_lo:
        lines.append(f"- Variance: {variance:.6f} ⚠ BELOW EXPECTED RANGE ({var_lo}-{var_hi}) — features may lack discriminative power")
    elif variance > var_hi:
        lines.append(f"- Variance: {variance:.6f} ⚠ ABOVE EXPECTED RANGE ({var_lo}-{var_hi}) — possible numerical instability")
    else:
        lines.append(f"- Variance: {variance:.6f} ✓ within expected range")

    # Overall assessment — only flag for genuinely bad indicators
    # High sparsity or very low variance are bad; low sparsity is not
    is_mediocre = sparsity > warn_sp or (variance < var_lo and var_lo > 0)
    if is_mediocre:
        lines.append("")
        lines.append("Quality appears MEDIOCRE for this descriptor. Consider RETRY_EXTRACT with a different descriptor.")

    return "\n".join(lines)


PH_SIGNAL_DEFINITIONS = {
    "h1_dominant": {
        "description": "H1 features significantly outnumber H0 — loop/cavity structure dominates",
        "condition": "H1/H0 ratio > 1.8 AND H1_count > 500",
        "recommends": ["persistence_landscapes", "persistence_silhouette"],
        "reasoning": (
            "When H1 features greatly exceed H0, the image has rich loop/cavity "
            "structure.  The discriminative signal is concentrated in multi-level "
            "H1 information — descriptors that can represent layered or hierarchical "
            "H1 structure will preserve this signal better than single-summary statistics."
        ),
    },
    "rich_topology": {
        "description": "Very many PH features with H1-heavy profile — multi-scale structure",
        "condition": "total features > 20,000 AND H1/H0 > 1.4",
        "recommends": ["persistence_landscapes", "betti_curves"],
        "reasoning": (
            "Extremely rich PH diagrams (>20K features) with H1 emphasis contain "
            "multi-scale topological information.  Descriptors that decompose the "
            "diagram into multiple scales or resolution levels will capture this "
            "richness better than fixed-resolution representations."
        ),
    },
    "low_persistence_signal": {
        "description": "Very low average persistence in both H0 and H1 — PH is noisy",
        "condition": "H0_avg_persistence < 0.01 AND H1_avg_persistence < 0.01",
        "recommends": ["lbp_texture", "edge_histogram"],
        "reasoning": (
            "When average persistence is very low in both dimensions, most PH "
            "features are noise (short-lived).  The persistent homology signal is "
            "unreliable, so descriptors that extract information directly from pixel "
            "intensity patterns may capture more discriminative structure."
        ),
    },
    "sparse_distinctive": {
        "description": "Few but persistent features — compact representations preferred",
        "condition": "total features < 800 AND H0_avg_persistence > 0.04",
        "recommends": ["tropical_coordinates", "persistence_image"],
        "reasoning": (
            "Sparse PH diagrams with high average persistence contain few but "
            "meaningful features.  Descriptors that faithfully preserve the exact "
            "geometric layout of points in the birth-persistence plane will retain "
            "more signal than those that aggregate or summarize broadly."
        ),
    },
}


def compute_ph_signals(
    h0_count: int,
    h1_count: int,
    h0_avg_persistence: float,
    h1_avg_persistence: float,
    h0_max_persistence: float = 1.0,
    h1_max_persistence: float = 1.0,
) -> List[Dict[str, Any]]:
    """Compute PH-based signals from persistence homology statistics.

    Returns a list of triggered signals, each with:
      - name: signal identifier
      - description: human-readable explanation
      - recommends: list of recommended descriptors
      - reasoning: why this signal recommends those descriptors
      - metrics: the specific PH values that triggered the signal

    Args:
        h0_count: Number of H0 features
        h1_count: Number of H1 features
        h0_avg_persistence: Average persistence of H0 features
        h1_avg_persistence: Average persistence of H1 features
        h0_max_persistence: Maximum persistence of H0 features
        h1_max_persistence: Maximum persistence of H1 features

    Returns:
        List of triggered signal dicts (may be empty)
    """
    signals = []
    total = h0_count + h1_count
    ratio = h1_count / h0_count if h0_count > 0 else 0.0

    # Signal 1: h1_dominant
    if ratio > 1.8 and h1_count > 500:
        signals.append({
            "name": "h1_dominant",
            "description": PH_SIGNAL_DEFINITIONS["h1_dominant"]["description"],
            "recommends": PH_SIGNAL_DEFINITIONS["h1_dominant"]["recommends"],
            "reasoning": PH_SIGNAL_DEFINITIONS["h1_dominant"]["reasoning"],
            "metrics": {"h1_h0_ratio": round(ratio, 2), "h1_count": h1_count},
        })

    # Signal 2: rich_topology
    if total > 20000 and ratio > 1.4:
        signals.append({
            "name": "rich_topology",
            "description": PH_SIGNAL_DEFINITIONS["rich_topology"]["description"],
            "recommends": PH_SIGNAL_DEFINITIONS["rich_topology"]["recommends"],
            "reasoning": PH_SIGNAL_DEFINITIONS["rich_topology"]["reasoning"],
            "metrics": {"total_features": total, "h1_h0_ratio": round(ratio, 2)},
        })

    # Signal 3: low_persistence_signal
    if h0_avg_persistence < 0.01 and h1_avg_persistence < 0.01:
        signals.append({
            "name": "low_persistence_signal",
            "description": PH_SIGNAL_DEFINITIONS["low_persistence_signal"]["description"],
            "recommends": PH_SIGNAL_DEFINITIONS["low_persistence_signal"]["recommends"],
            "reasoning": PH_SIGNAL_DEFINITIONS["low_persistence_signal"]["reasoning"],
            "metrics": {
                "h0_avg_persistence": round(h0_avg_persistence, 6),
                "h1_avg_persistence": round(h1_avg_persistence, 6),
            },
        })

    # Signal 4: sparse_distinctive
    if total < 800 and h0_avg_persistence > 0.04:
        signals.append({
            "name": "sparse_distinctive",
            "description": PH_SIGNAL_DEFINITIONS["sparse_distinctive"]["description"],
            "recommends": PH_SIGNAL_DEFINITIONS["sparse_distinctive"]["recommends"],
            "reasoning": PH_SIGNAL_DEFINITIONS["sparse_distinctive"]["reasoning"],
            "metrics": {
                "total_features": total,
                "h0_avg_persistence": round(h0_avg_persistence, 6),
            },
        })

    return signals


def build_ph_signals_text(signals: List[Dict[str, Any]], default_descriptor: str) -> str:
    """Build formatted text of PH signals for the ACT prompt.

    Includes conflict detection when signals recommend a different descriptor
    than the experience-based default.

    Args:
        signals: List of triggered signal dicts from compute_ph_signals()
        default_descriptor: The experience-based default descriptor

    Returns:
        Formatted string for prompt injection
    """
    if not signals:
        return (
            "No PH-based signals triggered. Your PH profile is within normal ranges.\n"
            "The experience-based default is a safe choice."
        )

    lines = ["**Triggered PH Signals:**\n"]
    conflict_descriptors = set()

    for sig in signals:
        lines.append(f"- **{sig['name']}**: {sig['description']}")
        metrics_str = ", ".join(f"{k}={v}" for k, v in sig["metrics"].items())
        lines.append(f"  Evidence: {metrics_str}")
        lines.append(f"  Suggests: {', '.join(sig['recommends'])}")
        lines.append(f"  Reasoning: {sig['reasoning']}")
        lines.append("")
        for rec in sig["recommends"]:
            if rec != default_descriptor:
                conflict_descriptors.add(rec)

    if conflict_descriptors:
        lines.append("**CONFLICT DETECTED:**")
        lines.append(
            f"Your PH signals suggest trying {' or '.join(sorted(conflict_descriptors))}, "
            f"but the experience-based default is {default_descriptor}."
        )
        lines.append(
            "You must weigh the PH evidence against benchmark experience. "
            "If the PH signal is strong and specific to THIS image, "
            "DEVIATE from the default. If unsure, FOLLOW the default."
        )
    else:
        lines.append(
            "PH signals are consistent with the experience-based default."
        )

    return "\n".join(lines)


def build_color_mode_advisory(n_channels: int) -> str:
    """Build advisory text about color mode for the OBSERVE prompt.

    Args:
        n_channels: Number of image channels (1 or 3)

    Returns:
        Advisory string for color mode decision
    """
    if n_channels == 1:
        return (
            "This image is grayscale (1 channel). Use color_mode='grayscale'."
        )
    # 3-channel RGB
    lines = [
        "This image has 3 channels (RGB). Per-channel mode computes PH on R,G,B",
        "separately and concatenates features (3x dimension). Accuracy gains by type:",
    ]
    for obj_type, benefit in COLOR_RULES["per_channel"]["benefit"].items():
        lines.append(f"  - {obj_type}: {benefit}")
    lines.append(
        "Per-channel is recommended for RGB images where color carries diagnostic info "
        "(dermoscopy, pathology, fundoscopy). Use grayscale if color is not diagnostically "
        "meaningful (X-ray, CT, MRI)."
    )
    return "\n".join(lines)


def build_parameter_table(object_type: str) -> str:
    """Build a formatted parameter table for all 13 descriptors for a given object_type.

    Args:
        object_type: One of 5 object types

    Returns:
        Formatted table string showing optimal params per descriptor
    """
    lines = []
    for desc in SUPPORTED_DESCRIPTORS:
        params = get_optimal_params(desc, object_type)
        param_strs = []
        for k, v in sorted(params.items()):
            if k == "dim":
                continue  # Show dim separately
            param_strs.append(f"{k}={v}")
        dim = params.get("dim", "?")
        param_str = ", ".join(param_strs) if param_strs else "defaults"
        lines.append(f"  {desc}: dim={dim}, {param_str}")
    return "\n".join(lines)


def get_top_recommendation(object_type: str) -> Tuple[str, float]:
    """Get the top-ranked supported descriptor for an object type.

    Args:
        object_type: One of 5 object types

    Returns:
        Tuple of (descriptor_name, accuracy)
    """
    performers = SUPPORTED_TOP_PERFORMERS.get(object_type, [])
    if performers:
        return performers[0]["descriptor"], performers[0]["accuracy"]
    return "persistence_statistics", 0.0


def get_pairwise_rho(desc1: str, desc2: str) -> Optional[float]:
    """Get Spearman rank correlation between two descriptors.

    Returns None if the pair is not found (same descriptor or unknown).
    """
    if desc1 == desc2:
        return 1.0
    pair = (desc1, desc2)
    if pair in PAIRWISE_SPEARMAN_RHO:
        return PAIRWISE_SPEARMAN_RHO[pair]
    pair_rev = (desc2, desc1)
    if pair_rev in PAIRWISE_SPEARMAN_RHO:
        return PAIRWISE_SPEARMAN_RHO[pair_rev]
    return None


def get_complementary_descriptors(
    primary: str,
    n: int = 5,
) -> List[Tuple[str, float]]:
    """Get the most complementary descriptors for a given primary.

    Returns list of (descriptor, rho) sorted by ascending rho (most complementary first).
    """
    pairs = []
    for (d1, d2), rho in PAIRWISE_SPEARMAN_RHO.items():
        if d1 == primary:
            pairs.append((d2, rho))
        elif d2 == primary:
            pairs.append((d1, rho))
    pairs.sort(key=lambda x: x[1])
    return pairs[:n]


def build_complementarity_text(
    primary: Optional[str] = None,
    object_type: Optional[str] = None,
    top_n: int = 8,
) -> str:
    """Build a formatted complementarity text for LLM prompt injection.

    Shows which descriptors are most complementary (low correlation = capture
    different information) for fusion portfolio selection.

    Args:
        primary: If given, show complementary descriptors for this primary.
        object_type: If given, show top performers for this type.
        top_n: Number of complementary pairs to show.

    Returns:
        Formatted string for prompt injection.
    """
    sections = []

    sections.append("## Descriptor Complementarity (Spearman Rank Correlations)")
    sections.append("")
    sections.append(
        "Low rho means descriptors capture DIFFERENT information — good fusion candidates.\n"
        "High rho means descriptors are redundant — fusion unlikely to help."
    )
    sections.append("")

    if primary:
        # Show complementary descriptors for the primary
        complements = get_complementary_descriptors(primary, n=top_n)
        sections.append(f"### Most complementary to {primary}:")
        for desc, rho in complements:
            tag = "VERY complementary" if rho < 0.6 else "somewhat complementary" if rho < 0.8 else "redundant"
            sections.append(f"  - {desc}: rho={rho:.3f} ({tag})")
        sections.append("")

    # Global most complementary pairs
    sections.append("### Most complementary pairs (global):")
    for d1, d2, rho in MOST_COMPLEMENTARY_PAIRS[:top_n]:
        sections.append(f"  - {d1} + {d2}: rho={rho:.3f}")
    sections.append("")

    sections.append(
        "Key insight: edge_histogram captures boundary/orientation information that is\n"
        "complementary to ALL topology-based descriptors (rho ≈ 0.48-0.58).\n"
        "euler_characteristic_transform is the second-most complementary (rho ≈ 0.72-0.83)."
    )
    sections.append("")

    # Object-type specific top performers (if given)
    if object_type and object_type in TOP_PERFORMERS:
        performers = TOP_PERFORMERS[object_type]
        sections.append(f"### Top-5 descriptors for {object_type} (benchmark accuracy):")
        for entry in performers[:5]:
            sections.append(f"  - {entry['descriptor']}: {entry['accuracy']:.3f}")
        sections.append("")

        # Suggest complementary pairs from top performers
        top_descs = [e["descriptor"] for e in performers[:5]]
        sections.append(f"### Suggested fusion pairs for {object_type}:")
        fusion_pairs = []
        for i, d1 in enumerate(top_descs):
            for d2 in top_descs[i + 1:]:
                rho = get_pairwise_rho(d1, d2)
                if rho is not None:
                    fusion_pairs.append((d1, d2, rho))
        fusion_pairs.sort(key=lambda x: x[2])
        for d1, d2, rho in fusion_pairs[:5]:
            sections.append(f"  - {d1} + {d2}: rho={rho:.3f}")

        # Also check edge_histogram as complement to top-1
        top1 = top_descs[0]
        eh_rho = get_pairwise_rho(top1, "edge_histogram")
        if eh_rho is not None and "edge_histogram" not in top_descs:
            sections.append(f"  - {top1} + edge_histogram: rho={eh_rho:.3f} (cross-type complement)")
        sections.append("")

    return "\n".join(sections)


# =============================================================================
# V9 DATA STRUCTURES AND HELPER FUNCTIONS
# =============================================================================

# =============================================================================
# OBJECT TYPE TAXONOMY — Quantitative PH signatures for blind object-type
# inference.  Shown to the LLM in INTERPRET so it can guess the object type
# from PH data alone (without knowing the dataset name).
# =============================================================================
OBJECT_TYPE_TAXONOMY: Dict[str, Dict[str, str]] = {
    "discrete_cells": {
        "ph_signature": "H0 >> H1, many short-lived H0 components from isolated objects",
        "typical_metrics": "H0_count: 200-2000, H1_count: 50-500, H0/H1 ratio > 2",
        "image_clues": "Small repeated structures, moderate contrast, often RGB microscopy",
    },
    "glands_lumens": {
        "ph_signature": "Both H0 and H1 significant, H1 from enclosed cavities/lumens",
        "typical_metrics": "H0_count: 100-1000, H1_count: 100-800, H0/H1 ratio ~1-2",
        "image_clues": "Tissue structures with enclosed spaces, varied texture, often RGB",
    },
    "vessel_trees": {
        "ph_signature": "H1 often dominant (loops from vessel branches), low persistence",
        "typical_metrics": "H0_count: 50-500, H1_count: 100-2000, low avg persistence",
        "image_clues": "Thin branching structures, low contrast, often RGB fundoscopy",
    },
    "surface_lesions": {
        "ph_signature": "Moderate H0+H1, irregular boundary patterns in H1",
        "typical_metrics": "H0_count: 50-300, H1_count: 100-500",
        "image_clues": "Color-rich surface features, moderate contrast, RGB dermoscopy",
    },
    "organ_shape": {
        "ph_signature": "H0 dominant (organ outline), few H1 (no internal cavities)",
        "typical_metrics": "H0_count: 50-500, H1_count: 10-100, high H0 persistence",
        "image_clues": "Large solid shapes, often grayscale (X-ray, CT, ultrasound)",
    },
}

# =============================================================================
# PARAMETER REASONING GUIDE — TDA domain knowledge for data-driven parameter
# derivation.  Maps descriptor -> param -> {principle, from_data rules}.
# Used by the LLM in ANALYZE to reason about parameter choices from PH data.
# =============================================================================
PARAMETER_REASONING_GUIDE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "persistence_image": {
        "sigma": {
            "principle": (
                "Gaussian bandwidth controlling how much each persistence point "
                "spreads on the birth-persistence grid.  Too low -> sparse/noisy image; "
                "too high -> over-smoothed, losing fine structure."
            ),
            "from_data": [
                "If avg persistence is very small (<0.01), use higher sigma (0.5-0.6) "
                "to smooth the many short-lived noisy features into coherent density.",
                "If avg persistence is moderate (0.02-0.10), sigma 0.1-0.3 preserves "
                "fine structure while smoothing noise.",
                "Default 0.5 works well across most medical datasets; sigma < 0.1 "
                "almost always hurts.",
            ],
        },
        "resolution": {
            "principle": (
                "Grid resolution (NxN) controlling the number of pixels in the "
                "persistence image.  Higher resolution captures finer detail but "
                "increases dimensionality quadratically (dim = resolution^2 * 2)."
            ),
            "from_data": [
                "If PH diagram is sparse (total < 800 features), low resolution "
                "(10-14) suffices — too many bins waste dimensions on empty regions.",
                "If PH diagram is rich (total > 5000 features), higher resolution "
                "(20-30) captures the richer point distribution.",
                "If H0 features cluster tightly (H0_count > 500 with persistence "
                "variance < 0.02), lower resolution (10-14) avoids wasting bins on "
                "empty space.  If H1 is also significant (H1_count > 500), higher "
                "resolution (20-26) captures the richer combined structure.",
            ],
        },
    },
    "persistence_landscapes": {
        "n_layers": {
            "principle": (
                "Number of landscape envelope layers.  Layer k captures the k-th "
                "largest persistence at each filtration value.  More layers = more "
                "topological scale information but higher dimensionality."
            ),
            "from_data": [
                "If H0+H1 count is large (>5000), many layers (28-38) capture "
                "the rich multi-scale structure.",
                "If H0+H1 count is small (<2000), fewer layers (10-15) suffice "
                "since higher layers are mostly zero.",
                "If avg persistence is very low (<0.01), even fewer layers (8-10) "
                "suffice since noisy short-lived features produce uninformative "
                "higher layers.",
            ],
        },
        "n_bins": {
            "principle": (
                "Number of filtration threshold bins per landscape layer.  More "
                "bins = finer resolution along the filtration axis."
            ),
            "from_data": [
                "Default n_bins=50 works well across most topological profiles.",
                "If the persistence range (max_persistence - min_persistence) is "
                "wide, use n_bins=75 to capture finer boundary detail.",
                "Dimension = n_layers * n_bins, so always use XGBoost (>2000D typical).",
            ],
        },
    },
    "betti_curves": {
        "n_bins": {
            "principle": (
                "Number of filtration threshold bins for the Betti number curve.  "
                "Each bin counts features alive at that threshold.  "
                "Dimension = n_bins * 2 (H0 + H1 concatenated)."
            ),
            "from_data": [
                "If death-time distribution is narrow (low persistence variance), "
                "fewer bins (50-100) suffice since most action happens in a small range.",
                "If death-time distribution is wide, more bins (140-200) capture "
                "the full filtration range.",
                "If H0 dominates with few H1 features (simple topology), 50-80 bins "
                "suffice.  If both H0 and H1 are rich, 120-200 bins capture the "
                "full multi-dimensional profile.",
            ],
        },
    },
    "persistence_silhouette": {
        "n_bins": {
            "principle": (
                "Number of filtration threshold bins for the silhouette function.  "
                "Similar to betti_curves but with power-weighted averaging.  "
                "Dimension = n_bins * 2."
            ),
            "from_data": [
                "If the PH profile has irregular, varied persistence patterns "
                "(high persistence variance), use more bins (160-180) to capture "
                "fine filtration resolution.",
                "If the topology is simple (H0 dominant, few H1, low variance), "
                "fewer bins (80-120) suffice.",
                "Default 140 bins works well for moderate topological complexity.",
            ],
        },
        "power": {
            "principle": (
                "Weighting exponent: power < 1 emphasizes short-lived features, "
                "power > 1 emphasizes long-lived features.  power=1 gives uniform "
                "weighting (identical to mean landscape)."
            ),
            "from_data": [
                "If many short-lived noise features are present (low avg persistence "
                "< 0.01), use power > 1 (e.g. 2.0) to suppress noise and emphasize "
                "persistent features.",
                "If short-lived features are diagnostically relevant (e.g. many "
                "small H0 components from discrete objects), use power < 1 (e.g. 0.5) "
                "to preserve them.",
                "If avg_persistence is very low AND H1 count is high (noisy H1-heavy "
                "profile), power=2.0 helps suppress the noisy short-lived features.",
            ],
        },
    },
    "template_functions": {
        "n_templates": {
            "principle": (
                "Number of tent/Gaussian basis functions placed on the persistence "
                "diagram.  Each template evaluates how many persistence points fall "
                "near its center.  Dimension = n_templates * 2."
            ),
            "from_data": [
                "If PH diagram has many clusters of points (H0_count > 1000 with "
                "moderate spread), use many templates (81-121) to tile the "
                "birth-persistence plane and capture cluster structure.",
                "If PH diagram is sparse (total < 800) or concentrated in a small "
                "region, fewer templates (16-25) suffice — excess templates produce "
                "zero-valued features.",
                "If persistence values span a wide range (max_persistence > 0.1), "
                "more templates capture the spread; if concentrated (max < 0.03), "
                "fewer templates suffice.",
            ],
        },
    },
    "minkowski_functionals": {
        "n_thresholds": {
            "principle": (
                "Number of binarization thresholds at which area, perimeter, and "
                "Euler characteristic are computed.  Dimension = n_thresholds * 3.  "
                "More thresholds = finer geometric profile of the image."
            ),
            "from_data": [
                "If image has high dynamic range / many intensity levels, use more "
                "thresholds (45-60) to capture the full geometric evolution.",
                "If image has low contrast or few distinct intensity levels, fewer "
                "thresholds (12-16) suffice.",
                "If many small objects appear at different intensity levels "
                "(high H0_count > 1000 with varied birth times), use 45-60 "
                "thresholds.  If the topology is simple with few components, "
                "12-20 thresholds suffice.",
            ],
        },
    },
}


def build_descriptor_properties_only() -> str:
    """Format DESCRIPTOR_PROPERTIES for ANALYZE prompt -- properties WITHOUT
    rankings or accuracy data.

    Shows captures, best_when, why, weakness for each descriptor grouped by
    PH-based vs Image-based.  The LLM uses this to reason about which
    descriptor fits the image, without being anchored by benchmark numbers.

    Returns:
        Formatted multi-line string with descriptor properties.
    """
    ph_based = []
    image_based = []
    for name, props in DESCRIPTOR_PROPERTIES.items():
        entry = f"**{name}** ({props['input']})"
        entry += f"\n  - Captures: {props['captures']}"
        entry += f"\n  - Best when: {props['best_when']}"
        entry += f"\n  - Why it works: {props['why']}"
        entry += f"\n  - Weakness: {props['weakness']}"
        if props["input"] == "PH diagrams":
            ph_based.append(entry)
        else:
            image_based.append(entry)

    sections = []
    sections.append("## Descriptor Properties (15 Descriptors)")
    sections.append("")
    sections.append("### PH-Based Descriptors (require compute_ph first)")
    for entry in ph_based:
        sections.append(entry)
        sections.append("")
    sections.append("### Image-Based Descriptors (no PH needed)")
    for entry in image_based:
        sections.append(entry)
        sections.append("")
    return "\n".join(sections)


def build_ph_signal_observations(signals: List[Dict[str, Any]]) -> str:
    """Format PH signals as topological observations WITHOUT the ``recommends``
    field.

    Shows signal name, description, triggering condition, metrics, and
    reasoning -- but strips out descriptor recommendations.  This is for the
    ANALYZE prompt where the LLM should reason about descriptors itself using
    DESCRIPTOR_PROPERTIES.

    Args:
        signals: List of triggered signal dicts from :func:`compute_ph_signals`.

    Returns:
        Formatted multi-line string.  Returns a "no signals" message when the
        list is empty.
    """
    if not signals:
        return (
            "No PH-based signals triggered. Your PH profile is within normal "
            "ranges for all four signal checks."
        )

    lines = ["## Topological Observations from PH Data\n"]
    for sig in signals:
        sig_def = PH_SIGNAL_DEFINITIONS.get(sig["name"], {})
        condition = sig_def.get("condition", "")
        lines.append(f"**{sig['name']}**: {sig['description']}")
        if condition:
            lines.append(f"  Condition: {condition}")
        metrics_str = ", ".join(f"{k}={v}" for k, v in sig["metrics"].items())
        lines.append(f"  Measured: {metrics_str}")
        lines.append(f"  Reasoning: {sig['reasoning']}")
        lines.append("")
    return "\n".join(lines)


def build_reasoning_principles() -> str:
    """Format REASONING_CHAINS with ``condition`` + ``reasoning`` text, but
    WITHOUT ``recommended`` and ``avoid`` lists.

    The LLM in ANALYZE must connect these principles to descriptors itself
    using DESCRIPTOR_PROPERTIES.  This prevents the LLM from short-circuiting
    reasoning by copying a recommendation list.

    Returns:
        Formatted multi-line string with reasoning principles.
    """
    sections = []
    sections.append("## Reasoning Principles: Image Properties -> Topological Implications")
    sections.append("")
    for chain_id, chain in REASONING_CHAINS.items():
        sections.append(f"**{chain_id}**: {chain['condition']}")
        sections.append(f"  Reasoning: {chain['reasoning']}")
        note = chain.get("note")
        if note:
            sections.append(f"  Note: {note}")
        sections.append("")
    return "\n".join(sections)


def build_parameter_reasoning_text(descriptor: str) -> str:
    """Format PARAMETER_REASONING_GUIDE for a specific descriptor.

    Shows the TDA principle and data-driven derivation rules for each tunable
    parameter of the given descriptor.  Used by the LLM to reason about
    parameter values from PH data characteristics.

    Args:
        descriptor: One of the 15 supported descriptors.

    Returns:
        Formatted multi-line string.  Returns a short message if the
        descriptor has no entries in the guide.
    """
    guide = PARAMETER_REASONING_GUIDE.get(descriptor)
    if not guide:
        return (
            f"No parameter reasoning guide available for {descriptor}. "
            "Use benchmark-optimal defaults from get_optimal_params()."
        )

    sections = []
    sections.append(f"## Parameter Reasoning for {descriptor}")
    sections.append("")
    for param_name, info in guide.items():
        sections.append(f"### {param_name}")
        sections.append(f"Principle: {info['principle']}")
        sections.append("Data-driven derivation:")
        for rule in info["from_data"]:
            sections.append(f"  - {rule}")
        sections.append("")
    return "\n".join(sections)


def build_benchmark_advisory(object_type: str) -> str:
    """Build the full benchmark advisory for the ACT prompt.

    Includes:
      - Ranked descriptor list with accuracy % (from SUPPORTED_TOP_PERFORMERS)
      - Optimal parameters from :func:`get_optimal_params` for each descriptor
      - Full PH signal recommendations (WITH ``recommends``) from
        PH_SIGNAL_DEFINITIONS
      - Full reasoning chain recommendations (WITH ``recommended``/``avoid``)
        from REASONING_CHAINS
      - A statement of the benchmark top pick

    Args:
        object_type: One of 5 object types.

    Returns:
        Formatted multi-line string for prompt injection.
    """
    sections = []

    # --- Section 1: Ranked descriptor list with accuracy and params ---
    performers = SUPPORTED_TOP_PERFORMERS.get(object_type, [])
    if performers:
        top_desc = performers[0]["descriptor"]
        top_acc = performers[0]["accuracy"]
        sections.append(f"## Benchmark Advisory for {object_type}")
        sections.append("")
        sections.append("### Descriptor Rankings (balanced accuracy, Benchmark4)")
        for i, entry in enumerate(performers, 1):
            desc = entry["descriptor"]
            acc = entry["accuracy"]
            params = get_optimal_params(desc, object_type)
            param_strs = []
            for k, v in sorted(params.items()):
                if k == "dim":
                    continue
                param_strs.append(f"{k}={v}")
            dim = params.get("dim", "?")
            param_str = ", ".join(param_strs) if param_strs else "defaults"
            sections.append(f"  {i}. {desc}: {acc:.1%} (dim={dim}, {param_str})")
        sections.append("")
    else:
        top_desc = "persistence_statistics"
        top_acc = 0.0
        sections.append(f"## Benchmark Advisory for {object_type}")
        sections.append("")
        sections.append("No benchmark rankings available for this object type.")
        sections.append("")

    # --- Section 2: PH signal definitions with recommendations ---
    sections.append("### PH Signal Definitions (with recommendations)")
    for sig_name, sig_def in PH_SIGNAL_DEFINITIONS.items():
        sections.append(f"- **{sig_name}**: {sig_def['description']}")
        sections.append(f"  Condition: {sig_def['condition']}")
        sections.append(f"  Recommends: {', '.join(sig_def['recommends'])}")
        sections.append(f"  Reasoning: {sig_def['reasoning']}")
    sections.append("")

    # --- Section 3: Reasoning chains with recommended/avoid ---
    sections.append("### Reasoning Chains (with recommendations)")
    for chain_id, chain in REASONING_CHAINS.items():
        sections.append(f"- **{chain_id}**: {chain['condition']}")
        sections.append(f"  Reasoning: {chain['reasoning']}")
        recommended = chain.get("recommended", [])
        if recommended:
            sections.append(f"  Recommended: {', '.join(recommended)}")
        avoid = chain.get("avoid", [])
        if avoid:
            sections.append(f"  Avoid: {', '.join(avoid)}")
        note = chain.get("note")
        if note:
            sections.append(f"  Note: {note}")
    sections.append("")

    # --- Section 4: Top pick summary ---
    sections.append(
        f"The benchmark top pick is **{top_desc}** ({top_acc:.1%})."
    )

    return "\n".join(sections)


def build_reference_quality_ranges(descriptor: str) -> str:
    """Format expected quality ranges as a neutral reference for REFLECT.

    Shows typical sparsity range, typical variance range, and warning
    threshold -- but NO emoji indicators, no pre-computed verdict text, and
    no GOOD/BAD labels.  Just the numbers as reference so the LLM can form
    its own assessment.

    Args:
        descriptor: One of the 15 supported descriptors.

    Returns:
        Formatted multi-line string with reference ranges.
    """
    expected = get_expected_quality(descriptor)
    sp_lo, sp_hi = expected["expected_sparsity_range"]
    var_lo, var_hi = expected["expected_variance_range"]
    warn_sp = expected["warning_sparsity"]

    lines = []
    lines.append(f"## Reference Quality Ranges for {descriptor}")
    lines.append(f"- Typical sparsity: {sp_lo}% - {sp_hi}%")
    lines.append(f"- Typical variance: {var_lo} - {var_hi}")
    lines.append(f"- Warning threshold: sparsity > {warn_sp}%")
    lines.append("")
    lines.append(
        "Note: sparsity below the typical range indicates denser-than-usual "
        "features (not necessarily a problem).  Sparsity above the warning "
        "threshold suggests the descriptor may be a poor fit for this image."
    )
    return "\n".join(lines)


# =============================================================================
# TIERED BENCHMARK ADVISORY — Groups descriptors by performance tier instead
# of showing exact accuracy %.  Prevents "accuracy gap anchoring" where the
# LLM always defers to the descriptor with the highest number.
# =============================================================================

def build_tiered_benchmark_advisory(object_type: str) -> str:
    """Build a tiered benchmark advisory WITHOUT exact accuracy percentages.

    Groups descriptors into 4 tiers based on proximity to the best performer.
    Includes PH signal definitions and reasoning chains (same as
    build_benchmark_advisory) but replaces exact accuracy with tier placement.

    Tier thresholds (absolute gap from best):
      Tier 1 (Highest):  within 2% of best
      Tier 2 (Strong):   within 5% of best
      Tier 3 (Moderate): within 10% of best
      Tier 4 (Weak):     >10% below best

    Args:
        object_type: One of the 5 object types.

    Returns:
        Formatted multi-line string for prompt injection.
    """
    sections: list = []

    performers = SUPPORTED_TOP_PERFORMERS.get(object_type, [])
    if not performers:
        sections.append(f"## Benchmark Advisory for {object_type} (Tiered)")
        sections.append("")
        sections.append("No benchmark rankings available for this object type.")
        return "\n".join(sections)

    best_acc = performers[0]["accuracy"]

    # Assign tiers
    tiers: Dict[int, list] = {1: [], 2: [], 3: [], 4: []}
    for entry in performers:
        gap = best_acc - entry["accuracy"]
        desc = entry["descriptor"]
        params = get_optimal_params(desc, object_type)
        param_strs = []
        for k, v in sorted(params.items()):
            if k == "dim":
                continue
            param_strs.append(f"{k}={v}")
        dim = params.get("dim", "?")
        param_str = ", ".join(param_strs) if param_strs else "defaults"
        info = f"{desc} (dim={dim}, {param_str})"

        if gap <= 0.02:
            tiers[1].append(info)
        elif gap <= 0.05:
            tiers[2].append(info)
        elif gap <= 0.10:
            tiers[3].append(info)
        else:
            tiers[4].append(info)

    tier_labels = {
        1: "Tier 1 (Highest Performing)",
        2: "Tier 2 (Strong)",
        3: "Tier 3 (Moderate)",
        4: "Tier 4 (Weak)",
    }

    sections.append(f"## Benchmark Advisory for {object_type} (Tiered — from 26-dataset study)")
    sections.append("")
    for tier_num in (1, 2, 3, 4):
        if tiers[tier_num]:
            sections.append(f"### {tier_labels[tier_num]}")
            for info in tiers[tier_num]:
                sections.append(f"- {info}")
            sections.append("")

    sections.append(
        "Tier placement reflects average empirical performance across datasets of "
        "this object type. It does NOT account for the specific topology of YOUR image."
    )
    sections.append("")

    # --- PH signal definitions (same as build_benchmark_advisory) ---
    sections.append("### PH Signal Definitions (with recommendations)")
    for sig_name, sig_def in PH_SIGNAL_DEFINITIONS.items():
        sections.append(f"- **{sig_name}**: {sig_def['description']}")
        sections.append(f"  Condition: {sig_def['condition']}")
        sections.append(f"  Recommends: {', '.join(sig_def['recommends'])}")
        sections.append(f"  Reasoning: {sig_def['reasoning']}")
    sections.append("")

    # --- Reasoning chains (same as build_benchmark_advisory) ---
    sections.append("### Reasoning Chains (with recommendations)")
    for chain_id, chain in REASONING_CHAINS.items():
        sections.append(f"- **{chain_id}**: {chain['condition']}")
        sections.append(f"  Reasoning: {chain['reasoning']}")
        recommended = chain.get("recommended", [])
        if recommended:
            sections.append(f"  Recommended: {', '.join(recommended)}")
        avoid = chain.get("avoid", [])
        if avoid:
            sections.append(f"  Avoid: {', '.join(avoid)}")
        note = chain.get("note")
        if note:
            sections.append(f"  Note: {note}")
    sections.append("")

    return "\n".join(sections)


# =============================================================================
# DOMAIN CONTEXT EXTRACTION — Strips dataset names from query while keeping
# domain-relevant information (domain, object description, class count).
# =============================================================================

def extract_domain_context(query: str) -> str:
    """Extract domain context from query, stripping dataset names.

    Removes all known dataset name patterns so the LLM sees domain info
    (e.g., "Hematology", "microscopy images of blood cells") but not
    identifiers like "BloodMNIST".

    Args:
        query: The original query string (may contain dataset context).

    Returns:
        Cleaned domain context string, or empty string if none found.
    """
    import re

    # Extract the "Dataset context:" section if present
    match = re.search(r"Dataset context:\s*(.+?)(?:\n\n|\Z)", query, re.DOTALL)
    if not match:
        return ""

    context = match.group(1).strip()

    # Strip all known dataset names (case-insensitive)
    dataset_names = list(DATASET_TO_OBJECT_TYPE.keys())
    # Also strip common abbreviations / alternate forms
    dataset_names += [
        "MedMNIST", "medmnist", "MNIST",
        "CRC-HE", "NCT-CRC", "AML_Cyto", "GasHis",
    ]
    for name in dataset_names:
        context = re.sub(re.escape(name), "[dataset]", context, flags=re.IGNORECASE)

    # Clean up residual patterns like "Dataset: [dataset];" or "[dataset] dataset"
    context = re.sub(r"\[dataset\]\s*dataset", "[dataset]", context, flags=re.IGNORECASE)

    return context
