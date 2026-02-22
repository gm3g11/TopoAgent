"""TopoBenchmark Configuration.

Rich dataset descriptions for agent context, protocol parameters,
and object type / color mode mappings.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import benchmark4 config for dataset metadata
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))
from config import (
    DATASETS as B4_DATASETS,
    get_object_type,
    get_color_mode,
    get_effective_n_samples,
)
sys.path.pop(0)


# =============================================================================
# DATASET DESCRIPTIONS — Rich context for agent queries
# =============================================================================
DATASET_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "BloodMNIST": {
        "domain": "Hematology",
        "description": (
            "Microscopy images of individual blood cells stained with Wright-Giemsa. "
            "Each image shows a single cell against a light background. Classes correspond "
            "to 8 cell types: basophil, eosinophil, erythroblast, immature granulocyte, "
            "lymphocyte, monocyte, neutrophil, and platelet."
        ),
        "what_matters": "Cell shape and internal granularity distinguish cell types. "
                        "Cells are isolated objects — many distinct connected components.",
        "n_classes": 8,
        "image_size": 224,
        "color_mode": "per_channel",
        "object_type": "discrete_cells",
    },
    "TissueMNIST": {
        "domain": "Histology",
        "description": (
            "Grayscale kidney cortex microscopy images showing cellular textures. "
            "8 tissue types based on cellular composition and structure."
        ),
        "what_matters": "Texture and density of cellular structures. "
                        "Patterns of nuclei and cell boundaries.",
        "n_classes": 8,
        "image_size": 224,
        "color_mode": "grayscale",
        "object_type": "discrete_cells",
    },
    "PathMNIST": {
        "domain": "Histopathology",
        "description": (
            "Colorectal cancer histopathology images with 9 tissue types including "
            "adipose, background, debris, lymphocytes, mucus, smooth muscle, "
            "normal colon mucosa, cancer-associated stroma, and tumor epithelium."
        ),
        "what_matters": "Glandular structure, lumen presence, and tissue architecture. "
                        "H&E staining reveals cellular and structural patterns.",
        "n_classes": 9,
        "image_size": 224,
        "color_mode": "per_channel",
        "object_type": "glands_lumens",
    },
    "OCTMNIST": {
        "domain": "Ophthalmology",
        "description": (
            "Retinal OCT (Optical Coherence Tomography) cross-section images. "
            "4 classes: choroidal neovascularization, diabetic macular edema, "
            "drusen, and normal retina."
        ),
        "what_matters": "Layer structure and deformations in retinal cross-sections. "
                        "Shape of fluid accumulations and layer boundaries.",
        "n_classes": 4,
        "image_size": 224,
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "OrganAMNIST": {
        "domain": "Radiology",
        "description": (
            "Axial-view CT organ images. 11 organ classes including bladder, "
            "femur, heart, kidney, liver, lung, pancreas, spleen, and others."
        ),
        "what_matters": "Overall organ shape and silhouette in axial plane. "
                        "Boundary contour is the primary discriminative feature.",
        "n_classes": 11,
        "image_size": 224,
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "RetinaMNIST": {
        "domain": "Ophthalmology",
        "description": (
            "Retinal fundus images graded for diabetic retinopathy severity. "
            "5 levels: no DR, mild, moderate, severe, and proliferative DR."
        ),
        "what_matters": "Branching vessel patterns and presence of microaneurysms, "
                        "hemorrhages, and neovascularization.",
        "n_classes": 5,
        "image_size": 224,
        "color_mode": "per_channel",
        "object_type": "vessel_trees",
    },
    "PneumoniaMNIST": {
        "domain": "Radiology",
        "description": (
            "Pediatric chest X-ray images classified as normal or pneumonia."
        ),
        "what_matters": "Lung field opacity patterns and overall shape. "
                        "Binary classification based on infiltrate presence.",
        "n_classes": 2,
        "image_size": 224,
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "BreastMNIST": {
        "domain": "Radiology",
        "description": (
            "Breast ultrasound images classified as benign or malignant."
        ),
        "what_matters": "Lesion shape and boundary regularity. "
                        "Malignant lesions tend to have irregular boundaries.",
        "n_classes": 2,
        "image_size": 224,
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "DermaMNIST": {
        "domain": "Dermatology",
        "description": (
            "Dermoscopic images of skin lesions from the HAM10000 dataset. "
            "7 classes: melanoma, melanocytic nevus, basal cell carcinoma, "
            "actinic keratosis, benign keratosis, dermatofibroma, vascular lesion."
        ),
        "what_matters": "Surface texture, color patterns, and lesion boundary shape. "
                        "Color carries significant diagnostic information.",
        "n_classes": 7,
        "image_size": 224,
        "color_mode": "per_channel",
        "object_type": "surface_lesions",
    },
    "OrganCMNIST": {
        "domain": "Radiology",
        "description": (
            "Coronal-view CT organ images. 11 organ classes, same as OrganAMNIST "
            "but viewed in the coronal plane."
        ),
        "what_matters": "Organ shape and silhouette in the coronal viewing plane.",
        "n_classes": 11,
        "image_size": 224,
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "OrganSMNIST": {
        "domain": "Radiology",
        "description": (
            "Sagittal-view CT organ images. 11 organ classes, same as OrganAMNIST "
            "but viewed in the sagittal plane."
        ),
        "what_matters": "Organ shape and silhouette in the sagittal viewing plane.",
        "n_classes": 11,
        "image_size": 224,
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "ISIC2019": {
        "domain": "Dermatology",
        "description": (
            "Large-scale dermoscopy images from the ISIC 2019 challenge. "
            "8 skin lesion types. High-resolution clinical images."
        ),
        "what_matters": "Lesion texture, color distribution, and boundary patterns. "
                        "RGB color is highly diagnostic.",
        "n_classes": 8,
        "image_size": 1022,
        "color_mode": "per_channel",
        "object_type": "surface_lesions",
    },
    "Kvasir": {
        "domain": "Gastroenterology",
        "description": (
            "Endoscopic images of the gastrointestinal tract. 8 classes including "
            "normal landmarks, polyps, and pathological findings."
        ),
        "what_matters": "Mucosal surface texture, glandular patterns, and polyp shapes. "
                        "Tissue architecture visible through endoscope.",
        "n_classes": 8,
        "image_size": 720,
        "color_mode": "per_channel",
        "object_type": "glands_lumens",
    },
    "BrainTumorMRI": {
        "domain": "Neuroradiology",
        "description": (
            "Brain MRI images with 4 classes: glioma, meningioma, pituitary tumor, "
            "and no tumor."
        ),
        "what_matters": "Tumor shape, location, and boundary regularity in brain scans.",
        "n_classes": 4,
        "image_size": 512,
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "MURA": {
        "domain": "Orthopedic Radiology",
        "description": (
            "Musculoskeletal X-ray images from Stanford. 14 body region classes "
            "including elbow, finger, forearm, hand, humerus, shoulder, wrist."
        ),
        "what_matters": "Bone shape and silhouette. Skeletal structure outlines.",
        "n_classes": 14,
        "image_size": "variable",
        "color_mode": "grayscale",
        "object_type": "organ_shape",
    },
    "BreakHis": {
        "domain": "Histopathology",
        "description": (
            "Breast cancer histopathology images at 40x-400x magnification. "
            "8 classes: 4 benign and 4 malignant subtypes."
        ),
        "what_matters": "Glandular structure, nuclear morphology, and tissue architecture. "
                        "H&E staining with gland/lumen patterns.",
        "n_classes": 8,
        "image_size": 700,
        "color_mode": "per_channel",
        "object_type": "glands_lumens",
    },
    "NCT_CRC_HE": {
        "domain": "Histopathology",
        "description": (
            "Colorectal cancer tissue patches stained with H&E. 9 tissue types "
            "similar to PathMNIST but higher resolution."
        ),
        "what_matters": "Tissue architecture, glandular structures, and cellular patterns.",
        "n_classes": 9,
        "image_size": 224,
        "color_mode": "per_channel",
        "object_type": "glands_lumens",
    },
    "MalariaCell": {
        "domain": "Parasitology",
        "description": (
            "Thin blood smear images for malaria detection. "
            "2 classes: parasitized and uninfected red blood cells."
        ),
        "what_matters": "Cell shape and presence of intracellular parasites. "
                        "Individual cells are isolated objects.",
        "n_classes": 2,
        "image_size": 142,
        "color_mode": "per_channel",
        "object_type": "discrete_cells",
    },
    "IDRiD": {
        "domain": "Ophthalmology",
        "description": (
            "High-resolution retinal fundus images from Indian Diabetic Retinopathy "
            "Image Dataset. 5 severity grades."
        ),
        "what_matters": "Retinal vessel branching, microaneurysms, and hemorrhage patterns. "
                        "Very high resolution (4288px original).",
        "n_classes": 5,
        "image_size": 4288,
        "color_mode": "per_channel",
        "object_type": "vessel_trees",
    },
    "PCam": {
        "domain": "Histopathology",
        "description": (
            "PatchCamelyon: small tissue patches from lymph node sections. "
            "Binary: metastatic tissue vs normal."
        ),
        "what_matters": "Cellular density and distribution patterns. "
                        "Small patches with discrete cellular structures.",
        "n_classes": 2,
        "image_size": 96,
        "color_mode": "per_channel",
        "object_type": "discrete_cells",
    },
    "LC25000": {
        "domain": "Histopathology",
        "description": (
            "Lung and colon cancer histopathology images. 5 classes: "
            "lung adenocarcinoma, lung squamous cell carcinoma, lung benign, "
            "colon adenocarcinoma, colon benign."
        ),
        "what_matters": "Glandular architecture and tissue organization patterns.",
        "n_classes": 5,
        "image_size": 768,
        "color_mode": "per_channel",
        "object_type": "glands_lumens",
    },
    "SIPaKMeD": {
        "domain": "Cytology",
        "description": (
            "Pap smear cell images. 5 cell types for cervical cancer screening."
        ),
        "what_matters": "Individual cell shape, nucleus-to-cytoplasm ratio. "
                        "Cells are isolated discrete objects.",
        "n_classes": 5,
        "image_size": "variable",
        "color_mode": "per_channel",
        "object_type": "discrete_cells",
    },
    "AML_Cytomorphology": {
        "domain": "Hematology",
        "description": (
            "Acute Myeloid Leukemia cytomorphology images. 15 cell types "
            "including blasts, promyelocytes, and mature cell types."
        ),
        "what_matters": "Cell morphology and nuclear characteristics. "
                        "Many discrete cell types with subtle differences.",
        "n_classes": 15,
        "image_size": "variable",
        "color_mode": "per_channel",
        "object_type": "discrete_cells",
    },
    "APTOS2019": {
        "domain": "Ophthalmology",
        "description": (
            "APTOS 2019 Blindness Detection: retinal fundus images with "
            "5 diabetic retinopathy severity grades."
        ),
        "what_matters": "Retinal vessel patterns, exudates, hemorrhages, "
                        "and neovascularization severity.",
        "n_classes": 5,
        "image_size": 1050,
        "color_mode": "per_channel",
        "object_type": "vessel_trees",
    },
    "GasHisSDB": {
        "domain": "Histopathology",
        "description": (
            "Gastric histopathology images. Binary: abnormal vs normal tissue."
        ),
        "what_matters": "Tissue texture and surface patterns. "
                        "Color and texture features are diagnostic.",
        "n_classes": 2,
        "image_size": 160,
        "color_mode": "per_channel",
        "object_type": "surface_lesions",
    },
    "Chaoyang": {
        "domain": "Histopathology",
        "description": (
            "Pathology images from Chaoyang Hospital. 4 classes: "
            "normal, serrated, adenocarcinoma, and adenoma."
        ),
        "what_matters": "Glandular architecture and cellular organization. "
                        "H&E staining with tissue-level patterns.",
        "n_classes": 4,
        "image_size": 512,
        "color_mode": "per_channel",
        "object_type": "glands_lumens",
    },
}


# =============================================================================
# PROTOCOL CONFIGS
# =============================================================================
PROTOCOL1_CONFIG = {
    "name": "per_dataset_strategic_selection",
    "description": "Agent selects ONE descriptor per dataset; accuracy looked up from benchmark4",
    "datasets": list(DATASET_DESCRIPTIONS.keys()),
    "n_datasets": len(DATASET_DESCRIPTIONS),
}

PROTOCOL2_CONFIG = {
    "name": "per_image_end_to_end",
    "description": "Agent processes individual images through full pipeline",
    "datasets": ["BloodMNIST", "Kvasir", "DermaMNIST"],
    "n_samples": 200,  # default; overridden by frozen config when available
    "n_folds": 5,
}


def get_protocol2_n(dataset: str) -> int:
    """Get sample size for Protocol 2 evaluation.

    Uses frozen config if available, otherwise falls back to PROTOCOL2_CONFIG default.
    """
    frozen_n = get_frozen_n(dataset)
    if frozen_n is not None:
        return frozen_n
    return PROTOCOL2_CONFIG['n_samples']


# =============================================================================
# ABLATION CONFIGS
# =============================================================================
ABLATION_CONFIGS = {
    "full": {
        "skills_mode": True,
        "enable_reflection": True,
        "enable_short_memory": True,
        "enable_long_memory": True,
    },
    "no_skills": {
        "skills_mode": False,
        "enable_reflection": True,
        "enable_short_memory": True,
        "enable_long_memory": True,
    },
    "no_reflection": {
        "skills_mode": True,
        "enable_reflection": False,
        "enable_short_memory": True,
        "enable_long_memory": True,
    },
    "no_short_memory": {
        "skills_mode": True,
        "enable_reflection": True,
        "enable_short_memory": False,
        "enable_long_memory": True,
    },
    "no_long_memory": {
        "skills_mode": True,
        "enable_reflection": True,
        "enable_short_memory": True,
        "enable_long_memory": False,
    },
}


# =============================================================================
# FROZEN DATASET CONFIG
# =============================================================================
FROZEN_CONFIG_PATH = Path(__file__).parent / "frozen_dataset_config.json"

_frozen_config_cache = None

def _load_frozen_config() -> Optional[Dict]:
    """Load frozen dataset config (cached)."""
    global _frozen_config_cache
    if _frozen_config_cache is not None:
        return _frozen_config_cache
    if FROZEN_CONFIG_PATH.exists():
        with open(FROZEN_CONFIG_PATH) as f:
            _frozen_config_cache = json.load(f)
        return _frozen_config_cache
    return None


def get_frozen_n(dataset: str) -> Optional[int]:
    """Get the frozen sample size for a dataset.

    Returns None if frozen config doesn't exist or dataset not found.
    """
    config = _load_frozen_config()
    if config is None:
        return None
    ds_cfg = config.get('datasets', {}).get(dataset)
    if ds_cfg is None:
        return None
    return ds_cfg['n']


def get_frozen_config() -> Optional[Dict]:
    """Get the full frozen dataset configuration."""
    return _load_frozen_config()


def build_agent_query(dataset: str) -> str:
    """Build the query string for the agent given a dataset name.

    Returns a natural-language query that includes dataset description,
    domain context, and what to look for — but does NOT reveal the
    ground truth best descriptor.
    """
    desc = DATASET_DESCRIPTIONS[dataset]
    query = (
        f"Classify images from the {dataset} dataset.\n\n"
        f"Domain: {desc['domain']}\n"
        f"Description: {desc['description']}\n"
        f"What matters: {desc['what_matters']}\n"
        f"Number of classes: {desc['n_classes']}\n"
        f"Image size: {desc['image_size']}\n"
        f"Color mode: {desc['color_mode']}\n"
        f"Object type: {desc['object_type']}\n\n"
        f"Select the best TDA descriptor for this dataset and explain your reasoning."
    )
    return query
