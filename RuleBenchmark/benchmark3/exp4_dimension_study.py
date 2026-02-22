#!/usr/bin/env python
"""
Experiment 4.1: Dimension Selection Study

Tests small/medium/large dimension configurations for all 15 descriptors
across different object types to find optimal dimensions.

Usage:
    python benchmarks/benchmark3/exp4_dimension_study.py --trial
    python benchmarks/benchmark3/exp4_dimension_study.py --object-type discrete_cells
    python benchmarks/benchmark3/exp4_dimension_study.py --full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from RuleBenchmark.benchmark3.exp4_config import (
    DIMENSION_CONFIGS, EXPECTED_DIMS, PARAM_GRIDS,
    OBJECT_TYPE_DATASETS, TRIAL_DATASETS,
    ALL_DESCRIPTORS, PH_BASED_DESCRIPTORS, IMAGE_BASED_DESCRIPTORS,
    LEARNED_DESCRIPTORS, EVALUATION, RESULTS_PATH,
    get_dimension, get_params,
)
from RuleBenchmark.benchmark3.config import (
    MEDMNIST_PATH, DATASETS, get_classifier,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name: str, n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset with stratified sampling.

    Uses data_loader.py for proper loading of all datasets (MedMNIST + External).
    Applies adaptive resize: >2048 -> 1024, else keep original.
    """
    from RuleBenchmark.benchmark3.data_loader import load_dataset as _load_dataset_full

    # Get native image size from config
    dataset_info = DATASETS.get(dataset_name, {})
    native_size = dataset_info.get('image_size', 224)

    # Adaptive resize: >2048 -> 1024, else keep original
    if isinstance(native_size, str):
        target_size = 1024
    elif native_size > 2048:
        target_size = 1024
        print(f"    Image size {native_size} > 2048, resizing to 1024")
    else:
        target_size = native_size

    # Load using proper data_loader
    images, labels, class_names = _load_dataset_full(
        dataset_name, n_samples=n_samples, seed=seed, image_size=target_size
    )

    return images, labels


def load_medmnist(dataset_name: str, n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load MedMNIST dataset."""

    # Map to file names
    name_map = {
        'BloodMNIST': 'bloodmnist',
        'TissueMNIST': 'tissuemnist',
        'PathMNIST': 'pathmnist',
        'DermaMNIST': 'dermamnist',
        'OCTMNIST': 'octmnist',
        'OrganAMNIST': 'organamnist',
        'RetinaMNIST': 'retinamnist',
        'PneumoniaMNIST': 'pneumoniamnist',
        'BreastMNIST': 'breastmnist',
        'OrganCMNIST': 'organcmnist',
        'OrganSMNIST': 'organsmnist',
    }

    file_name = name_map.get(dataset_name, dataset_name.lower())

    # Try 224 resolution first, then 28
    for res in [224, 28]:
        npz_path = MEDMNIST_PATH / f"{file_name}_{res}.npz"
        if npz_path.exists():
            break
        npz_path = MEDMNIST_PATH / f"{file_name}.npz"
        if npz_path.exists():
            break

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name} at {MEDMNIST_PATH}")

    print(f"  Loading {dataset_name} from {npz_path.name}...")
    data = np.load(npz_path, mmap_mode='r')

    # Use train split
    all_images = data['train_images']
    all_labels = data['train_labels'].flatten()

    # Stratified sampling
    n_total = len(all_labels)
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    np.random.seed(seed)
    indices = []
    for i, label in enumerate(sorted(unique_labels)):
        class_idx = np.where(all_labels == label)[0]
        n = per_class + (1 if i < remainder else 0)
        n = min(n, len(class_idx))
        chosen = np.random.choice(class_idx, n, replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    images = np.array(all_images[indices])
    labels = all_labels[indices].copy()

    del data, all_images, all_labels
    gc.collect()

    # Adaptive resize: keep original if ≤1024, else resize to 1024×1024
    images = adaptive_resize(images, max_size=1024)

    # Convert to grayscale float32 [0, 1]
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

    print(f"    Loaded {len(images)} images, shape={images.shape}, classes={n_classes}")
    return images, labels


def adaptive_resize(images: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Adaptive resize: keep original if ≤max_size, else resize to max_size×max_size.

    Uses INTER_AREA (pixel area relation) for downsampling which is best for
    preserving topological features like connected components and holes.
    """
    import cv2

    h, w = images.shape[1], images.shape[2]

    # No resize needed if within limit
    if h <= max_size and w <= max_size:
        print(f"    Image size {h}×{w} ≤ {max_size}, keeping original")
        return images

    # Calculate target size maintaining aspect ratio
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    print(f"    Resizing {h}×{w} → {new_h}×{new_w} (scale={scale:.3f})")

    # Resize each image using INTER_AREA (best for downsampling, preserves topology)
    resized = []
    for img in images:
        if img.ndim == 2:
            # Grayscale
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Color (H, W, C)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized.append(resized_img)

    return np.array(resized, dtype=images.dtype)


# =============================================================================
# PH COMPUTATION
# =============================================================================

def compute_ph_gudhi(images: np.ndarray, n_jobs: int = -1) -> List[Dict]:
    """Compute PH using GUDHI CubicalComplex + joblib."""
    import gudhi
    from joblib import Parallel, delayed

    def _ph_single(img):
        cc = gudhi.CubicalComplex(top_dimensional_cells=img)
        cc.persistence()
        pd = {}
        for dim in [0, 1]:
            intervals = cc.persistence_intervals_in_dimension(dim)
            pd[f'H{dim}'] = [
                {'birth': float(b), 'death': float(d)}
                for b, d in intervals
                if np.isfinite(d) and d > b
            ]
        return pd

    print(f"  Computing PH with GUDHI (n_jobs={n_jobs})...")
    start = time.time()
    diagrams = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_ph_single)(img) for img in images
    )
    elapsed = time.time() - start
    print(f"    PH: {elapsed:.1f}s ({elapsed/len(images)*1000:.1f}ms/image)")

    return diagrams


# =============================================================================
# DESCRIPTOR EXTRACTION
# =============================================================================

def extract_descriptor(
    desc_name: str,
    images: np.ndarray,
    diagrams: List[Dict],
    params: Dict,
    expected_dim: int,
) -> np.ndarray:
    """Extract features for one descriptor with given params."""

    if desc_name in IMAGE_BASED_DESCRIPTORS:
        return _extract_image_descriptor(desc_name, images, params, expected_dim)
    else:
        return _extract_ph_descriptor(desc_name, diagrams, params, expected_dim)


def _extract_image_descriptor(
    desc_name: str,
    images: np.ndarray,
    params: Dict,
    expected_dim: int,
) -> np.ndarray:
    """Extract image-based descriptors."""
    from topoagent.tools.descriptors import (
        EulerCharacteristicCurveTool, EulerCharacteristicTransformTool,
        LBPTextureTool, EdgeHistogramTool,
    )
    from topoagent.tools.morphology import MinkowskiFunctionalsTool

    if desc_name == 'minkowski_functionals':
        return _extract_minkowski(images, params, expected_dim)

    if desc_name == 'lbp_texture':
        return _extract_lbp(images, params, expected_dim)

    tool_map = {
        'euler_characteristic_curve': EulerCharacteristicCurveTool,
        'euler_characteristic_transform': EulerCharacteristicTransformTool,
        'edge_histogram': EdgeHistogramTool,
    }

    tool = tool_map[desc_name]()
    feat_list = []

    run_params = {}
    if desc_name == 'euler_characteristic_curve':
        run_params = {'resolution': params.get('resolution', 100)}
    elif desc_name == 'euler_characteristic_transform':
        run_params = {
            'n_directions': params.get('n_directions', 32),
            'n_heights': params.get('n_heights', 20),
        }
    elif desc_name == 'edge_histogram':
        run_params = {
            'n_orientation_bins': params.get('n_orientation_bins', 8),
            'n_spatial_cells': params.get('n_spatial_cells', 10),
        }

    for img in images:
        result = tool._run(image_array=img.tolist(), **run_params)
        if result.get('success', False):
            vec = np.array(result['combined_vector'], dtype=np.float32)
        else:
            vec = np.zeros(expected_dim, dtype=np.float32)

        # Pad/truncate to expected dimension
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


def _extract_minkowski(images: np.ndarray, params: Dict, expected_dim: int) -> np.ndarray:
    """Extract Minkowski functionals at multiple thresholds."""
    from topoagent.tools.morphology import MinkowskiFunctionalsTool

    n_thresholds = params.get('n_thresholds', 10)
    adaptive = params.get('adaptive', True)
    tool = MinkowskiFunctionalsTool()
    percentiles = np.linspace(10, 90, n_thresholds)

    feat_list = []
    for img in images:
        if adaptive:
            thresholds = np.percentile(img, percentiles)
        else:
            thresholds = np.linspace(0.05, 0.95, n_thresholds)

        curve = []
        for t in thresholds:
            result = tool._run(image_array=img.tolist(), threshold=float(t))
            if result.get('success', False):
                mf = result.get('minkowski_functionals', {})
                curve.extend([
                    mf.get('area', 0.0),
                    mf.get('perimeter', 0.0),
                    mf.get('euler_characteristic', 0.0),
                ])
            else:
                curve.extend([0.0, 0.0, 0.0])

        vec = np.array(curve, dtype=np.float32)
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


def _extract_lbp(images: np.ndarray, params: Dict, expected_dim: int) -> np.ndarray:
    """Extract multi-scale LBP."""
    from topoagent.tools.descriptors import LBPTextureTool

    scales = params.get('scales', [(8, 1.0), (16, 2.0), (24, 3.0)])
    tool = LBPTextureTool()

    feat_list = []
    for img in images:
        multi_scale_vec = []
        for P, R in scales:
            result = tool._run(image_array=img.tolist(), P=P, R=R, method='uniform')
            if result.get('success', False):
                multi_scale_vec.extend(result['combined_vector'])
            else:
                multi_scale_vec.extend([0.0] * (P + 2))

        vec = np.array(multi_scale_vec, dtype=np.float32)
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


def _extract_ph_descriptor(
    desc_name: str,
    diagrams: List[Dict],
    params: Dict,
    expected_dim: int,
) -> np.ndarray:
    """Extract PH-based descriptors."""
    from RuleBenchmark.benchmark3.descriptors import (
        PersistenceStatisticsTool, PersistenceImageTool,
        BettiCurvesTool, TropicalCoordinatesTool, ATOLTool,
    )
    from topoagent.tools.descriptors import (
        PersistenceEntropyTool, PersistenceCodebookTool,
        TemplateFunctionsTool,
    )
    from topoagent.tools.vectorization import (
        PersistenceLandscapesTool, PersistenceSilhouetteTool,
    )

    tool_map = {
        'persistence_statistics': PersistenceStatisticsTool,
        'persistence_image': PersistenceImageTool,
        'persistence_landscapes': PersistenceLandscapesTool,
        'persistence_silhouette': PersistenceSilhouetteTool,
        'betti_curves': BettiCurvesTool,
        'persistence_entropy': PersistenceEntropyTool,
        'persistence_codebook': PersistenceCodebookTool,
        'tropical_coordinates': TropicalCoordinatesTool,
        'ATOL': ATOLTool,
        'template_functions': TemplateFunctionsTool,
    }

    tool = tool_map[desc_name]()

    # Map params to tool params
    run_params = _map_params_to_tool(desc_name, params)

    feat_list = []
    for diag in diagrams:
        result = tool._run(persistence_data=diag, **run_params)
        if result.get('success', False):
            vec = np.array(result['combined_vector'], dtype=np.float32)
        else:
            vec = np.zeros(expected_dim, dtype=np.float32)

        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


def _map_params_to_tool(desc_name: str, params: Dict) -> Dict:
    """Map config params to tool-specific params."""
    if desc_name == 'persistence_image':
        return {
            'resolution': params.get('resolution', 20),
            'sigma': params.get('sigma', 0.1),
            'weight_function': params.get('weight_function', 'linear'),
        }
    elif desc_name == 'persistence_landscapes':
        return {
            'n_layers': params.get('n_layers', 4),
            'n_bins': params.get('n_bins', 100),
            'combine_dims': params.get('combine_dims', True),
        }
    elif desc_name == 'persistence_silhouette':
        return {
            'n_bins': params.get('n_bins', 100),
            'power': params.get('power', 1.0),
        }
    elif desc_name == 'betti_curves':
        return {
            'n_bins': params.get('n_bins', 100),
            'normalize': params.get('normalize', False),
        }
    elif desc_name == 'persistence_entropy':
        return {
            'mode': params.get('mode', 'vector'),
            'n_bins': params.get('n_bins', 100),
        }
    elif desc_name == 'persistence_codebook':
        return {'codebook_size': params.get('codebook_size', 32)}
    elif desc_name == 'ATOL':
        return {'n_centers': params.get('n_centers', 16)}
    elif desc_name == 'tropical_coordinates':
        return {'max_terms': params.get('max_terms', 5)}
    elif desc_name == 'template_functions':
        return {
            'n_templates': params.get('n_templates', 25),
            'template_type': params.get('template_type', 'tent'),
        }
    elif desc_name == 'persistence_statistics':
        return {'subset': params.get('subset', 'basic')}
    else:
        return params


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_with_cv(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    n_classes: int = None,
    dataset_name: str = None,
    classifier_name: str = 'TabPFN',
) -> Dict[str, float]:
    """Evaluate features with cross-validation using specified classifier."""

    if n_classes is None:
        n_classes = len(np.unique(labels))

    clf = get_classifier(classifier_name, seed=seed, dataset_name=dataset_name, n_classes=n_classes)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for train_idx, val_idx in skf.split(features, labels):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            clf_copy = get_classifier(classifier_name, seed=seed, dataset_name=dataset_name, n_classes=n_classes)
            clf_copy.fit(X_train, y_train)
            y_pred = clf_copy.predict(X_val)
            acc = balanced_accuracy_score(y_val, y_pred)
            fold_accs.append(acc)
        except Exception as e:
            # Try to clear CUDA memory and fallback to RandomForest
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

            if 'CUDA' in str(e) or 'out of memory' in str(e):
                print(f"      OOM at this dimension - stopping")
                # Return special marker indicating OOM
                return {
                    'mean': -1.0,  # Special marker for OOM
                    'std': 0.0,
                    'folds': [],
                    'oom': True,
                    'oom_message': str(e),
                }
            else:
                print(f"      CV error: {e}")
                fold_accs.append(0.0)

    return {
        'mean': np.mean(fold_accs),
        'std': np.std(fold_accs),
        'folds': fold_accs,
    }


def evaluate_learned_descriptor_cv(
    desc_name: str,
    diagrams: List[Dict],
    labels: np.ndarray,
    params: Dict,
    expected_dim: int,
    n_folds: int = 5,
    seed: int = 42,
    dataset_name: str = None,
    classifier_name: str = 'TabPFN',
) -> Dict[str, float]:
    """Evaluate learned descriptor with CV-scoped fitting (no data leakage)."""
    from RuleBenchmark.benchmark3.descriptors import ATOLTool
    from topoagent.tools.descriptors import PersistenceCodebookTool

    n_classes = len(np.unique(labels))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_diagrams = [diagrams[i] for i in train_idx]
        val_diagrams = [diagrams[i] for i in val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Create fresh tool and fit on train only
        if desc_name == 'ATOL':
            tool = ATOLTool()
            n_centers = params.get('n_centers', 16)
            tool.fit(train_diagrams, n_centers=n_centers)
            run_params = {'n_centers': n_centers}
        else:  # persistence_codebook
            tool = PersistenceCodebookTool()
            codebook_size = params.get('codebook_size', 32)
            tool.fit(train_diagrams, codebook_size=codebook_size)
            run_params = {'codebook_size': codebook_size}

        # Extract features
        def extract_batch(diags):
            feats = []
            for diag in diags:
                result = tool._run(persistence_data=diag, **run_params)
                if result.get('success', False):
                    vec = np.array(result['combined_vector'], dtype=np.float32)
                else:
                    vec = np.zeros(expected_dim, dtype=np.float32)
                if len(vec) < expected_dim:
                    vec = np.pad(vec, (0, expected_dim - len(vec)))
                elif len(vec) > expected_dim:
                    vec = vec[:expected_dim]
                feats.append(vec)
            return np.array(feats, dtype=np.float32)

        X_train = extract_batch(train_diagrams)
        X_val = extract_batch(val_diagrams)

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            clf = get_classifier(classifier_name, seed=seed, dataset_name=dataset_name, n_classes=n_classes)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            acc = balanced_accuracy_score(y_val, y_pred)
            fold_accs.append(acc)
        except Exception as e:
            # Try to clear CUDA memory and fallback to RandomForest
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

            if 'CUDA' in str(e) or 'out of memory' in str(e):
                print(f"      OOM at this dimension - stopping")
                # Return special marker indicating OOM
                return {
                    'mean': -1.0,  # Special marker for OOM
                    'std': 0.0,
                    'folds': [],
                    'oom': True,
                    'oom_message': str(e),
                }
            else:
                print(f"      CV error: {e}")
                fold_accs.append(0.0)

    return {
        'mean': np.mean(fold_accs),
        'std': np.std(fold_accs),
        'folds': fold_accs,
    }


# =============================================================================
# MAIN DIMENSION STUDY
# =============================================================================

def run_dimension_study(
    object_type: str,
    datasets: List[str],
    n_samples: int = 50,
    n_folds: int = 3,
    seed: int = 42,
    n_jobs: int = 4,
    descriptors: List[str] = None,
    classifier_name: str = 'TabPFN',
) -> Dict:
    """Run dimension study for one object type."""

    if descriptors is None:
        descriptors = ALL_DESCRIPTORS

    results = {
        'object_type': object_type,
        'datasets': datasets,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'seed': seed,
        'results': {},
    }

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset} (object_type: {object_type})")
        print(f"{'='*60}")

        # Load data
        images, labels = load_dataset(dataset, n_samples=n_samples, seed=seed)

        # Compute PH (for PH-based descriptors)
        diagrams = compute_ph_gudhi(images, n_jobs=n_jobs)

        results['results'][dataset] = {}

        for desc_name in descriptors:
            print(f"\n  Descriptor: {desc_name}")
            results['results'][dataset][desc_name] = {}

            for level in ['small', 'medium', 'large']:
                params = get_params(desc_name, level)
                expected_dim = get_dimension(desc_name, level)

                print(f"    {level} ({expected_dim}D)...", end=' ')
                start = time.time()

                try:
                    if desc_name in LEARNED_DESCRIPTORS:
                        # CV-scoped fitting for learned descriptors
                        cv_result = evaluate_learned_descriptor_cv(
                            desc_name, diagrams, labels, params, expected_dim,
                            n_folds=n_folds, seed=seed, dataset_name=dataset,
                            classifier_name=classifier_name,
                        )
                    else:
                        # Regular extraction + CV
                        features = extract_descriptor(
                            desc_name, images, diagrams, params, expected_dim,
                        )
                        cv_result = evaluate_with_cv(
                            features, labels, n_folds=n_folds, seed=seed,
                            dataset_name=dataset, classifier_name=classifier_name,
                        )

                    elapsed = time.time() - start
                    acc_mean = cv_result['mean']
                    acc_std = cv_result['std']

                    results['results'][dataset][desc_name][level] = {
                        'accuracy': acc_mean,
                        'std': acc_std,
                        'dim': expected_dim,
                        'time': elapsed,
                        'params': params,
                    }

                    print(f"{acc_mean:.3f}±{acc_std:.2f} ({elapsed:.1f}s)")

                except Exception as e:
                    print(f"ERROR: {e}")
                    results['results'][dataset][desc_name][level] = {
                        'accuracy': 0.0,
                        'std': 0.0,
                        'dim': expected_dim,
                        'error': str(e),
                    }

    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze dimension study results to find best dimensions."""

    analysis = {
        'best_per_descriptor': {},
        'best_per_dataset': {},
        'summary': {},
    }

    # Aggregate across datasets
    desc_level_scores = {}

    for dataset, desc_results in results['results'].items():
        analysis['best_per_dataset'][dataset] = {}

        for desc_name, level_results in desc_results.items():
            if desc_name not in desc_level_scores:
                desc_level_scores[desc_name] = {'small': [], 'medium': [], 'large': []}

            best_level = None
            best_acc = -1

            for level, metrics in level_results.items():
                acc = metrics.get('accuracy', 0.0)
                desc_level_scores[desc_name][level].append(acc)

                if acc > best_acc:
                    best_acc = acc
                    best_level = level

            analysis['best_per_dataset'][dataset][desc_name] = {
                'best_level': best_level,
                'accuracy': best_acc,
            }

    # Find best level per descriptor (average across datasets)
    for desc_name, level_scores in desc_level_scores.items():
        level_means = {
            level: np.mean(scores) if scores else 0.0
            for level, scores in level_scores.items()
        }
        best_level = max(level_means, key=level_means.get)

        analysis['best_per_descriptor'][desc_name] = {
            'best_level': best_level,
            'mean_accuracy': level_means[best_level],
            'all_means': level_means,
        }

    # Summary
    analysis['summary'] = {
        'object_type': results['object_type'],
        'n_datasets': len(results['datasets']),
        'n_descriptors': len(desc_level_scores),
        'recommendations': {
            desc: info['best_level']
            for desc, info in analysis['best_per_descriptor'].items()
        },
    }

    return analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 4.1: Dimension Selection Study")
    parser.add_argument('--trial', action='store_true', help='Quick trial with 50 samples')
    parser.add_argument('--full', action='store_true', help='Full run with 5000 samples')
    parser.add_argument('--object-type', type=str, default=None,
                       help='Run for specific object type')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Run for specific dataset')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples per dataset')
    parser.add_argument('--n-folds', type=int, default=3,
                       help='Number of CV folds')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of parallel jobs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classifier', type=str, default='TabPFN',
                       choices=['TabPFN', 'XGBoost', 'RandomForest'],
                       help='Classifier to use (default: TabPFN)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')

    args = parser.parse_args()

    # Determine n_samples
    if args.full:
        n_samples = 5000
        n_folds = 5
    elif args.trial:
        n_samples = 50
        n_folds = 3
    else:
        n_samples = args.n_samples
        n_folds = args.n_folds

    print("=" * 70)
    print("  EXPERIMENT 4.1: DIMENSION SELECTION STUDY")
    print("=" * 70)
    print(f"  n_samples: {n_samples}")
    print(f"  n_folds: {n_folds}")
    print(f"  classifier: {args.classifier}")
    print(f"  seed: {args.seed}")
    print("=" * 70)

    all_results = {}

    # Determine which object types to run
    if args.dataset:
        # Single dataset
        from RuleBenchmark.benchmark3.exp4_config import DATASET_OBJECT_TYPE
        obj_type = DATASET_OBJECT_TYPE.get(args.dataset, 'unknown')
        datasets_to_run = {obj_type: [args.dataset]}
    elif args.object_type:
        # Single object type
        datasets_to_run = {args.object_type: TRIAL_DATASETS.get(args.object_type, [])}
    else:
        # All trial datasets
        datasets_to_run = TRIAL_DATASETS

    for obj_type, datasets in datasets_to_run.items():
        print(f"\n\n{'#'*70}")
        print(f"  OBJECT TYPE: {obj_type}")
        print(f"{'#'*70}")

        results = run_dimension_study(
            object_type=obj_type,
            datasets=datasets,
            n_samples=n_samples,
            n_folds=n_folds,
            seed=args.seed,
            n_jobs=args.n_jobs,
            classifier_name=args.classifier,
        )

        analysis = analyze_results(results)
        results['analysis'] = analysis
        all_results[obj_type] = results

        # Print recommendations
        print(f"\n  RECOMMENDATIONS for {obj_type}:")
        for desc, level in analysis['summary']['recommendations'].items():
            acc = analysis['best_per_descriptor'][desc]['mean_accuracy']
            print(f"    {desc}: {level} ({acc:.3f})")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_PATH / f"exp4_dimension_study_n{n_samples}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
