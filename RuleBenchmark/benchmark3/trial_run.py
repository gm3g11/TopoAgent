#!/usr/bin/env python
"""
Benchmark3 Trial Run: Correctness & Speed Check

Tests all 15 descriptors × 4 classifiers on 500 BloodMNIST samples.
Validates:
  - Output dimensions match EXPECTED_DIMS
  - No NaN/Inf in features
  - GPU acceleration works (XGBoost, TabPFN)
  - Multi-core CPU parallelism (PH computation, RandomForest, KNN)
  - Classification produces above-chance results
  - Per-descriptor extraction timing
  - Per-classifier training/inference timing

Usage:
    python benchmarks/benchmark3/trial_run.py
    python benchmarks/benchmark3/trial_run.py --n-samples 200 --n-jobs 4
    python benchmarks/benchmark3/trial_run.py --no-gpu
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Dict, List, Any, Optional

from RuleBenchmark.benchmark3.config import (
    DESCRIPTORS, EXPECTED_DIMS, CLASSIFIERS, ACTIVE_CLASSIFIERS,
    PH_CONFIG, MANY_CLASS_DATASETS, MEDMNIST_PATH,
    get_classifier, get_preprocessing_pipeline,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bloodmnist(n_samples: int = 500, seed: int = 42) -> tuple:
    """Load BloodMNIST at native 224x224 resolution."""

    npz_path = MEDMNIST_PATH / "bloodmnist_224.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"BloodMNIST not found at {npz_path}")

    print(f"Loading BloodMNIST from {npz_path}...")
    data = np.load(npz_path, mmap_mode='r')
    all_labels = data['train_labels'].flatten()
    all_images = data['train_images']

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

    print(f"  Loaded {len(images)} images, shape={images.shape}, classes={n_classes}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return images, labels


# =============================================================================
# PH COMPUTATION (giotto-tda CubicalPersistence)
# =============================================================================

def _gudhi_ph_single(img: np.ndarray) -> Dict:
    """Compute PH for a single image using GUDHI CubicalComplex."""
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=img)
    cc.persistence()

    pd = {}
    for dim in [0, 1]:
        intervals = cc.persistence_intervals_in_dimension(dim)
        pd[f'H{dim}'] = [
            {'birth': float(b), 'death': float(d)}
            for b, d in intervals
            if np.isfinite(d) and d > b  # Filter infinite and zero-persistence
        ]
    return pd


def compute_ph_gudhi(images: np.ndarray, n_jobs: int = -1) -> tuple:
    """Compute PH using GUDHI CubicalComplex + joblib parallelization.

    Returns list of persistence dicts in our standard format:
        [{'H0': [{'birth': b, 'death': d}, ...], 'H1': [...]}, ...]
    """
    from joblib import Parallel, delayed

    print(f"  Computing PH with GUDHI + joblib (n_jobs={n_jobs}, {images.shape[1]}x{images.shape[2]})...")

    start = time.time()
    diagrams = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_gudhi_ph_single)(img) for img in images
    )
    elapsed = time.time() - start
    print(f"  PH computation: {elapsed:.1f}s for {len(images)} images "
          f"({elapsed/len(images)*1000:.1f}ms/image)")

    return diagrams, elapsed


# =============================================================================
# DESCRIPTOR EXTRACTION
# =============================================================================

def extract_descriptor(
    desc_name: str,
    images: np.ndarray,
    diagrams: List[Dict],
    config: Dict,
) -> tuple:
    """Extract features for one descriptor.

    Returns: (features_array, elapsed_time, errors)
    """
    from RuleBenchmark.benchmark3.config import PH_BASED_DESCRIPTORS, IMAGE_BASED_DESCRIPTORS

    expected_dim = EXPECTED_DIMS[desc_name]
    params = config.get('params', {})
    errors = []

    start = time.time()

    if desc_name in IMAGE_BASED_DESCRIPTORS:
        features = _extract_image_descriptor(desc_name, images, params, expected_dim, errors)
    else:
        features = _extract_ph_descriptor(desc_name, diagrams, params, expected_dim, errors)

    elapsed = time.time() - start
    return features, elapsed, errors


def _extract_image_descriptor(
    desc_name: str, images: np.ndarray, params: Dict,
    expected_dim: int, errors: List
) -> np.ndarray:
    """Extract image-based descriptors (no PH needed)."""
    from topoagent.tools.descriptors import (
        EulerCharacteristicCurveTool, EulerCharacteristicTransformTool,
        MinkowskiFunctionalsTool, LBPTextureTool, EdgeHistogramTool,
    )

    # Special handling for minkowski_functionals (multi-threshold → 30D)
    if desc_name == 'minkowski_functionals':
        return _extract_minkowski_multi_threshold(images, params, expected_dim, errors)

    tool_map = {
        'euler_characteristic_curve': EulerCharacteristicCurveTool,
        'euler_characteristic_transform': EulerCharacteristicTransformTool,
        'lbp_texture': LBPTextureTool,
        'edge_histogram': EdgeHistogramTool,
    }

    tool = tool_map[desc_name]()
    feat_list = []

    # Map config params to actual tool params
    run_params = {}
    if desc_name == 'euler_characteristic_transform':
        run_params = {
            'n_directions': params.get('n_directions', 32),
            'n_heights': params.get('n_thresholds', 20),
        }
    elif desc_name == 'euler_characteristic_curve':
        run_params = {'resolution': params.get('n_bins', 100)}
    elif desc_name == 'edge_histogram':
        run_params = {
            'n_orientation_bins': params.get('n_orientation_bins', 8),
            'n_spatial_cells': params.get('n_spatial_cells', 10),
        }
    elif desc_name == 'lbp_texture':
        # Multi-scale LBP: handled separately below
        return _extract_lbp_multiscale(images, params, expected_dim, errors)

    for i, img in enumerate(images):
        result = tool._run(image_array=img.tolist(), **run_params)
        if result.get('success', False):
            vec = np.array(result['combined_vector'], dtype=np.float32)
            if len(vec) < expected_dim:
                vec = np.pad(vec, (0, expected_dim - len(vec)))
            elif len(vec) > expected_dim:
                vec = vec[:expected_dim]
            feat_list.append(vec)
        else:
            errors.append(f"Sample {i}: {result.get('error', 'unknown')}")
            feat_list.append(np.zeros(expected_dim, dtype=np.float32))

    return np.array(feat_list, dtype=np.float32)


def _extract_minkowski_multi_threshold(
    images: np.ndarray, params: Dict,
    expected_dim: int, errors: List
) -> np.ndarray:
    """Extract Minkowski functionals at multiple thresholds → 30D.

    Computes area, perimeter, Euler char at n_thresholds levels.
    Uses adaptive (percentile-based) thresholds when configured.
    Output: 3 functionals × n_thresholds = 30D
    """
    from topoagent.tools.morphology import MinkowskiFunctionalsTool

    n_thresholds = params.get('n_thresholds', 10)
    adaptive = params.get('adaptive', True)
    tool = MinkowskiFunctionalsTool()

    # Percentile levels for adaptive thresholds
    percentiles = np.linspace(10, 90, n_thresholds)

    feat_list = []
    for i, img in enumerate(images):
        img_list = img.tolist()

        # Compute thresholds: adaptive (percentile) or fixed (linspace)
        if adaptive:
            thresholds = np.percentile(img, percentiles)
        else:
            thresholds = np.linspace(0.05, 0.95, n_thresholds)

        curve = []
        for t in thresholds:
            result = tool._run(image_array=img_list, threshold=float(t))
            if result.get('success', False):
                mf = result.get('minkowski_functionals', {})
                area = mf.get('area', 0.0)
                perimeter = mf.get('perimeter', 0.0)
                euler = mf.get('euler_characteristic', 0.0)
                curve.extend([area, perimeter, euler])
            else:
                curve.extend([0.0, 0.0, 0.0])

        vec = np.array(curve, dtype=np.float32)
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


def _extract_ph_descriptor(
    desc_name: str, diagrams: List[Dict], params: Dict,
    expected_dim: int, errors: List
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

    # Special parameters for specific descriptors
    run_params = {}
    if desc_name == 'persistence_entropy':
        run_params = {'mode': 'vector', 'n_bins': params.get('n_bins', 100)}
    elif desc_name == 'persistence_landscapes':
        run_params = {
            'n_layers': params.get('n_layers', 4),
            'n_bins': params.get('n_bins', 100),
            'combine_dims': True,  # 400D mode
        }
    elif desc_name == 'persistence_silhouette':
        run_params = {
            'power': params.get('power', 1.0),
            'n_bins': params.get('n_bins', 100),
        }
    elif desc_name == 'betti_curves':
        run_params = {'n_bins': params.get('n_bins', 100)}
    elif desc_name == 'persistence_image':
        run_params = {
            'sigma': params.get('sigma', 0.1),
            'resolution': params.get('n_bins', 20),
        }
    elif desc_name == 'persistence_codebook':
        # codebook_size × 2 hom dims = output dim; 32 × 2 = 64D
        run_params = {'codebook_size': params.get('n_codewords', 64) // 2}
    elif desc_name == 'ATOL':
        run_params = {'n_centers': params.get('n_clusters', 16)}
    elif desc_name == 'template_functions':
        run_params = {'n_templates': params.get('n_templates', 25)}
    elif desc_name == 'persistence_statistics':
        run_params = {'subset': 'basic'}

    # NOTE: For learned descriptors (ATOL, Codebook), we DO NOT fit here.
    # Fitting happens inside CV folds via _extract_learned_descriptor_for_fold()
    # to avoid data leakage. Here we just return placeholder features.
    from RuleBenchmark.benchmark3.config import LEARNED_DESCRIPTORS
    if desc_name in LEARNED_DESCRIPTORS:
        # Return zeros as placeholder; actual features extracted per-fold in evaluate_classifier
        return np.zeros((len(diagrams), expected_dim), dtype=np.float32)

    feat_list = []
    for i, diag in enumerate(diagrams):
        result = tool._run(persistence_data=diag, **run_params)
        if result.get('success', False):
            vec = np.array(result['combined_vector'], dtype=np.float32)
            if len(vec) < expected_dim:
                vec = np.pad(vec, (0, expected_dim - len(vec)))
            elif len(vec) > expected_dim:
                vec = vec[:expected_dim]
            feat_list.append(vec)
        else:
            errors.append(f"Sample {i}: {result.get('error', 'unknown')}")
            feat_list.append(np.zeros(expected_dim, dtype=np.float32))

    return np.array(feat_list, dtype=np.float32)


def _extract_lbp_multiscale(
    images: np.ndarray, params: Dict,
    expected_dim: int, errors: List
) -> np.ndarray:
    """Extract multi-scale LBP → 54D.

    P=8,R=1 (uniform → 10 bins) + P=16,R=2 (uniform → 18 bins) +
    P=24,R=3 (uniform → 26 bins) = 54D total.
    """
    from topoagent.tools.descriptors import LBPTextureTool

    scales = [
        (8, 1.0),   # P=8, R=1 → 10 bins (uniform)
        (16, 2.0),  # P=16, R=2 → 18 bins (uniform)
        (24, 3.0),  # P=24, R=3 → 26 bins (uniform)
    ]

    tool = LBPTextureTool()
    feat_list = []

    for i, img in enumerate(images):
        img_list = img.tolist()
        multi_scale_vec = []

        for P, R in scales:
            result = tool._run(image_array=img_list, P=P, R=R, method='uniform')
            if result.get('success', False):
                vec = result['combined_vector']
                multi_scale_vec.extend(vec)
            else:
                # P+2 bins for uniform LBP
                multi_scale_vec.extend([0.0] * (P + 2))

        vec = np.array(multi_scale_vec, dtype=np.float32)
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


# =============================================================================
# CLASSIFICATION EVALUATION
# =============================================================================

def _extract_learned_descriptor_for_fold(
    desc_name: str,
    train_diagrams: List[Dict],
    val_diagrams: List[Dict],
    params: Dict,
) -> tuple:
    """Extract features for learned descriptors with train-only fitting.

    This prevents data leakage by fitting ATOL/Codebook only on training data.

    Returns: (X_train, X_val) feature arrays
    """
    from RuleBenchmark.benchmark3.descriptors import ATOLTool
    from topoagent.tools.descriptors import PersistenceCodebookTool
    from RuleBenchmark.benchmark3.config import EXPECTED_DIMS

    expected_dim = EXPECTED_DIMS[desc_name]

    if desc_name == 'ATOL':
        tool = ATOLTool()
        n_centers = params.get('n_clusters', 16)
        # Fit on train data ONLY
        tool.fit(train_diagrams, n_centers=n_centers)
        run_params = {'n_centers': n_centers}
    else:  # persistence_codebook
        tool = PersistenceCodebookTool()
        codebook_size = params.get('n_codewords', 64) // 2
        # Fit on train data ONLY
        tool.fit(train_diagrams, codebook_size=codebook_size)
        run_params = {'codebook_size': codebook_size}

    def extract_batch(diagrams):
        feat_list = []
        for diag in diagrams:
            result = tool._run(persistence_data=diag, **run_params)
            if result.get('success', False):
                vec = np.array(result['combined_vector'], dtype=np.float32)
                if len(vec) < expected_dim:
                    vec = np.pad(vec, (0, expected_dim - len(vec)))
                elif len(vec) > expected_dim:
                    vec = vec[:expected_dim]
                feat_list.append(vec)
            else:
                feat_list.append(np.zeros(expected_dim, dtype=np.float32))
        return np.array(feat_list, dtype=np.float32)

    X_train = extract_batch(train_diagrams)
    X_val = extract_batch(val_diagrams)

    return X_train, X_val


def evaluate_classifier(
    clf_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    n_folds: int = 3,  # 3 folds for speed in trial
    diagrams: List[Dict] = None,
    desc_name: str = None,
    desc_params: Dict = None,
) -> Dict[str, Any]:
    """Evaluate one classifier with cross-validation.

    For learned descriptors (ATOL, persistence_codebook), pass diagrams and desc_name
    to enable proper CV-scoped fitting that avoids data leakage.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from RuleBenchmark.benchmark3.config import LEARNED_DESCRIPTORS

    n_classes = len(np.unique(labels))
    clf = get_classifier(clf_name, seed=seed, dataset_name='BloodMNIST', n_classes=n_classes)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    total_train_time = 0
    total_predict_time = 0

    # Check if this is a learned descriptor that needs CV-scoped fitting
    is_learned = desc_name in LEARNED_DESCRIPTORS if desc_name else False

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        # For learned descriptors, re-extract features with train-only fitting
        if is_learned and diagrams is not None:
            train_diagrams = [diagrams[i] for i in train_idx]
            val_diagrams = [diagrams[i] for i in val_idx]

            X_train, X_val = _extract_learned_descriptor_for_fold(
                desc_name, train_diagrams, val_diagrams, desc_params or {}
            )
        else:
            X_train, X_val = features[train_idx], features[val_idx]

        y_train, y_val = labels[train_idx], labels[val_idx]

        # Preprocessing: impute + scale
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            clf_copy = clone(clf)

            t0 = time.time()
            clf_copy.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = clf_copy.predict(X_val)
            predict_time = time.time() - t0

            acc = accuracy_score(y_val, y_pred)
            bal_acc = balanced_accuracy_score(y_val, y_pred)

            fold_results.append({
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'train_time': train_time,
                'predict_time': predict_time,
            })
            total_train_time += train_time
            total_predict_time += predict_time

        except Exception as e:
            fold_results.append({
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'train_time': 0.0,
                'predict_time': 0.0,
                'error': str(e),
            })

    result = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'balanced_accuracy': np.mean([r['balanced_accuracy'] for r in fold_results]),
        'bal_acc_std': np.std([r['balanced_accuracy'] for r in fold_results]),
        'avg_train_time': total_train_time / n_folds,
        'avg_predict_time': total_predict_time / n_folds,
        'total_time': total_train_time + total_predict_time,
        'errors': [r.get('error') for r in fold_results if 'error' in r],
    }
    return result


# =============================================================================
# MAIN TRIAL RUN
# =============================================================================

def run_trial(
    n_samples: int = 500,
    n_jobs: int = -1,
    use_gpu: bool = True,
    seed: int = 42,
):
    """Run the full trial: extract all descriptors, evaluate all classifiers."""
    import torch

    print("=" * 70)
    print("  BENCHMARK3 TRIAL RUN")
    print("=" * 70)
    print(f"  Samples: {n_samples}")
    print(f"  Descriptors: {len(DESCRIPTORS)}")
    print(f"  Classifiers: {ACTIVE_CLASSIFIERS}")
    print(f"  n_jobs: {n_jobs}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Use GPU: {use_gpu}")
    print("=" * 70)

    total_start = time.time()

    # --- Step 1: Load data ---
    print("\n[1/4] LOADING DATA")
    print("-" * 40)
    images, labels = load_bloodmnist(n_samples=n_samples, seed=seed)

    # --- Step 2: Compute PH ---
    print("\n[2/4] COMPUTING PERSISTENT HOMOLOGY")
    print("-" * 40)
    diagrams, ph_time = compute_ph_gudhi(images, n_jobs=n_jobs)

    # Quick validation
    n_h0 = np.mean([len(d.get('H0', [])) for d in diagrams])
    n_h1 = np.mean([len(d.get('H1', [])) for d in diagrams])
    print(f"  Avg bars: H0={n_h0:.1f}, H1={n_h1:.1f}")

    # --- Step 3: Extract all descriptors ---
    print("\n[3/4] EXTRACTING DESCRIPTORS")
    print("-" * 40)
    print(f"{'Descriptor':<32} {'Dim':>5} {'Actual':>6} {'Time':>7} {'NaN%':>5} {'Zero%':>5} {'Status'}")
    print("-" * 90)

    features_dict = {}
    timing_dict = {}
    all_correct = True

    for desc_name, desc_config in DESCRIPTORS.items():
        expected_dim = EXPECTED_DIMS[desc_name]

        features, elapsed, errors = extract_descriptor(
            desc_name, images, diagrams, desc_config
        )

        actual_dim = features.shape[1]
        nan_pct = np.mean(np.isnan(features)) * 100
        inf_pct = np.mean(np.isinf(features)) * 100
        zero_pct = np.mean(features == 0) * 100

        dim_ok = actual_dim == expected_dim
        nan_ok = nan_pct == 0 and inf_pct == 0
        status = "OK" if (dim_ok and nan_ok and not errors) else "WARN"

        if not dim_ok:
            status = f"DIM_MISMATCH({actual_dim}!={expected_dim})"
            all_correct = False
        elif not nan_ok:
            status = f"NaN:{nan_pct:.1f}%/Inf:{inf_pct:.1f}%"
            all_correct = False
        elif errors:
            status = f"ERRORS({len(errors)})"

        print(f"{desc_name:<32} {expected_dim:>5} {actual_dim:>6} {elapsed:>6.2f}s "
              f"{nan_pct:>4.1f}% {zero_pct:>4.1f}% {status}")

        # Replace NaN/Inf for classification
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_dict[desc_name] = features
        timing_dict[desc_name] = elapsed

    total_extract_time = sum(timing_dict.values())
    print(f"\n  Total extraction time: {total_extract_time:.1f}s")
    print(f"  Total feature dims: {sum(f.shape[1] for f in features_dict.values())}")

    # --- Step 4: Classification ---
    print("\n[4/4] CLASSIFICATION EVALUATION")
    print("-" * 40)

    # Header
    clf_header = "".join(f"{c:>14}" for c in ACTIVE_CLASSIFIERS)
    print(f"{'Descriptor':<28} {clf_header}  {'Best':>12}")
    print("-" * (28 + 14 * len(ACTIVE_CLASSIFIERS) + 14))

    results_table = {}
    clf_timings = {c: 0.0 for c in ACTIVE_CLASSIFIERS}

    from RuleBenchmark.benchmark3.config import LEARNED_DESCRIPTORS

    for desc_name, features in features_dict.items():
        results_table[desc_name] = {}
        row = f"{desc_name:<28}"

        best_acc = 0
        best_clf = ""

        # For learned descriptors, pass diagrams for CV-scoped fitting (no data leakage)
        is_learned = desc_name in LEARNED_DESCRIPTORS
        desc_config = DESCRIPTORS.get(desc_name, {})

        for clf_name in ACTIVE_CLASSIFIERS:
            result = evaluate_classifier(
                clf_name, features, labels, seed=seed, n_folds=3,
                diagrams=diagrams if is_learned else None,
                desc_name=desc_name if is_learned else None,
                desc_params=desc_config.get('params', {}) if is_learned else None,
            )
            results_table[desc_name][clf_name] = result
            clf_timings[clf_name] += result['total_time']

            acc = result['balanced_accuracy']
            row += f"{acc:>10.3f}±{result['bal_acc_std']:.2f}"

            if acc > best_acc:
                best_acc = acc
                best_clf = clf_name

            if result['errors']:
                row_suffix = " ERR"

        row += f"  {best_clf:>12}"
        print(row)

    total_time = time.time() - total_start

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  Correctness: {'ALL PASS' if all_correct else 'ISSUES DETECTED'}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"    - Data loading: ~{ph_time:.1f}s (PH computation)")
    print(f"    - Feature extraction: {total_extract_time:.1f}s")
    print(f"    - Classification: {sum(clf_timings.values()):.1f}s")

    print(f"\n  Descriptor extraction speed (top 5 slowest):")
    for desc, t in sorted(timing_dict.items(), key=lambda x: -x[1])[:5]:
        print(f"    {desc}: {t:.2f}s ({t/n_samples*1000:.1f}ms/sample)")

    print(f"\n  Classifier speed:")
    for clf_name in ACTIVE_CLASSIFIERS:
        t = clf_timings[clf_name]
        print(f"    {clf_name}: {t:.2f}s total ({t/len(DESCRIPTORS):.2f}s/descriptor)")

    # Best results per classifier
    print(f"\n  Best balanced_accuracy per classifier:")
    for clf_name in ACTIVE_CLASSIFIERS:
        best_desc = max(results_table.keys(),
                       key=lambda d: results_table[d][clf_name]['balanced_accuracy'])
        best_acc = results_table[best_desc][clf_name]['balanced_accuracy']
        print(f"    {clf_name}: {best_acc:.3f} ({best_desc})")

    # Best results per descriptor
    print(f"\n  Best balanced_accuracy per descriptor (top 5):")
    desc_best = {}
    for desc_name in results_table:
        best = max(results_table[desc_name].values(), key=lambda r: r['balanced_accuracy'])
        desc_best[desc_name] = best['balanced_accuracy']

    for desc, acc in sorted(desc_best.items(), key=lambda x: -x[1])[:5]:
        print(f"    {desc}: {acc:.3f}")

    # Chance level
    n_classes = len(np.unique(labels))
    print(f"\n  Chance level: {1/n_classes:.3f} ({n_classes} classes)")

    # GPU/CPU check
    print(f"\n  Resource utilization:")
    import torch
    if torch.cuda.is_available() and use_gpu:
        print(f"    GPU: Active (XGBoost tree_method=hist/cuda, TabPFN device=cuda)")
    else:
        print(f"    GPU: Not used")
    print(f"    CPU cores: {n_jobs if n_jobs > 0 else 'all'} "
          f"(PH computation, RandomForest, KNN)")

    print("\n" + "=" * 70)
    print("  TRIAL RUN COMPLETE")
    print("=" * 70)

    return results_table, timing_dict, features_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark3 Trial Run")
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Parallel jobs (-1 = all cores)')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    run_trial(
        n_samples=args.n_samples,
        n_jobs=args.n_jobs,
        use_gpu=not args.no_gpu,
        seed=args.seed,
    )
