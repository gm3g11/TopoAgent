#!/usr/bin/env python
"""
Experiment 4.3: Color Handling Experiment

Test different color conversion methods for each object type to derive rules:
"For object type X, use color method Y"

Uses 4 representative descriptors instead of all 15 for efficiency.

Usage:
    python benchmarks/benchmark3/exp4_color_experiment.py --trial
    python benchmarks/benchmark3/exp4_color_experiment.py --object-type discrete_cells
    python benchmarks/benchmark3/exp4_color_experiment.py --full
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
import cv2
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from RuleBenchmark.benchmark3.exp4_config import (
    COLOR_HANDLING_RULES, COLOR_METHODS, COLOR_EXPERIMENT_DESCRIPTORS,
    OBJECT_TYPE_DATASETS, TRIAL_DATASETS, RESULTS_PATH,
    MEMORY_SAFE_DIMENSION_SEARCH, LEARNED_DESCRIPTORS,
    PH_BASED_DESCRIPTORS,
)
from RuleBenchmark.benchmark3.exp4_dimension_study import (
    load_dataset, compute_ph_gudhi, extract_descriptor,
    evaluate_with_cv, evaluate_learned_descriptor_cv,
)
from RuleBenchmark.benchmark3.config import MEDMNIST_PATH, DATASETS


# =============================================================================
# COLOR CONVERSION METHODS
# =============================================================================

def convert_color(images: np.ndarray, method: str) -> np.ndarray:
    """
    Convert images using specified color method.

    Args:
        images: (N, H, W, 3) RGB images or (N, H, W) grayscale
        method: Color conversion method name

    Returns:
        Converted images ready for PH computation
    """
    # If already grayscale, only grayscale method makes sense
    if images.ndim == 3 or (images.ndim == 4 and images.shape[3] == 1):
        if images.ndim == 4:
            images = images[:, :, :, 0]
        if method != 'grayscale':
            print(f"    Warning: Images already grayscale, using grayscale method")
        return images.astype(np.float32) / 255.0 if images.max() > 1 else images.astype(np.float32)

    # RGB images
    if method == 'grayscale':
        # Luminosity formula
        converted = (0.299 * images[:, :, :, 0] +
                    0.587 * images[:, :, :, 1] +
                    0.114 * images[:, :, :, 2])

    elif method == 'red_channel':
        converted = images[:, :, :, 0].astype(np.float32)

    elif method == 'green_channel':
        converted = images[:, :, :, 1].astype(np.float32)

    elif method == 'blue_channel':
        converted = images[:, :, :, 2].astype(np.float32)

    elif method == 'hsv_value':
        # V channel from HSV (intensity)
        converted = []
        for img in images:
            hsv = cv2.cvtColor(img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8),
                              cv2.COLOR_RGB2HSV)
            converted.append(hsv[:, :, 2])  # V channel
        converted = np.array(converted, dtype=np.float32)

    elif method == 'hsv_saturation':
        # S channel from HSV (color purity)
        converted = []
        for img in images:
            hsv = cv2.cvtColor(img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8),
                              cv2.COLOR_RGB2HSV)
            converted.append(hsv[:, :, 1])  # S channel
        converted = np.array(converted, dtype=np.float32)

    elif method == 'per_channel':
        # Return as-is, will compute PH on each channel separately
        return images.astype(np.float32) / 255.0 if images.max() > 1 else images.astype(np.float32)

    else:
        raise ValueError(f"Unknown color method: {method}")

    # Normalize to [0, 1]
    converted = converted.astype(np.float32)
    if converted.max() > 1.0:
        converted = converted / 255.0

    return converted


def compute_ph_per_channel(images: np.ndarray, n_jobs: int = 4) -> List[Dict]:
    """
    Compute PH on each RGB channel separately and combine.

    Returns diagrams with H0_R, H0_G, H0_B, H1_R, H1_G, H1_B
    """
    import gudhi
    from joblib import Parallel, delayed

    def _ph_single_channel(img):
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

    def _ph_rgb(img_rgb):
        """Compute PH on each channel and combine."""
        combined = {}
        channel_names = ['R', 'G', 'B']

        for c, name in enumerate(channel_names):
            channel_img = img_rgb[:, :, c]
            pd = _ph_single_channel(channel_img)
            for key, intervals in pd.items():
                combined[f'{key}_{name}'] = intervals

        # Also add combined H0/H1 for compatibility
        combined['H0'] = combined.get('H0_R', []) + combined.get('H0_G', []) + combined.get('H0_B', [])
        combined['H1'] = combined.get('H1_R', []) + combined.get('H1_G', []) + combined.get('H1_B', [])

        return combined

    print(f"  Computing PH per channel (RGB) with n_jobs={n_jobs}...")
    start = time.time()
    diagrams = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_ph_rgb)(img) for img in images
    )
    elapsed = time.time() - start
    print(f"    PH (per-channel): {elapsed:.1f}s ({elapsed/len(images)*1000:.1f}ms/image)")

    return diagrams


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_color_method(
    descriptor: str,
    images_rgb: np.ndarray,
    labels: np.ndarray,
    color_method: str,
    n_folds: int = 5,
    seed: int = 42,
    n_jobs: int = 4,
    dataset_name: str = None,
) -> Dict:
    """Evaluate a descriptor with a specific color method."""

    print(f"\n    Color method: {color_method}")

    # Get descriptor config
    config = MEMORY_SAFE_DIMENSION_SEARCH.get(descriptor, {})
    # Use medium dimension for fair comparison
    if config:
        values = config['values']
        dimensions = config['dimensions']
        mid_idx = len(values) // 2
        params = config.get('fixed', {}).copy()
        params[config['control_param']] = values[mid_idx]
        expected_dim = dimensions[mid_idx]
    else:
        params = {}
        expected_dim = 100

    start = time.time()

    try:
        if color_method == 'per_channel':
            # Compute PH on each channel
            diagrams = compute_ph_per_channel(images_rgb, n_jobs=n_jobs)
            # For per_channel, features are 3x larger
            expected_dim_total = expected_dim * 3 if descriptor in PH_BASED_DESCRIPTORS else expected_dim
        else:
            # Convert to single channel
            images_gray = convert_color(images_rgb, color_method)

            if descriptor in PH_BASED_DESCRIPTORS:
                diagrams = compute_ph_gudhi(images_gray, n_jobs=n_jobs)
            else:
                diagrams = None
            expected_dim_total = expected_dim

        # Extract features and evaluate
        if descriptor in LEARNED_DESCRIPTORS:
            cv_result = evaluate_learned_descriptor_cv(
                descriptor, diagrams, labels, params, expected_dim_total,
                n_folds=n_folds, seed=seed, dataset_name=dataset_name,
            )
        else:
            if color_method == 'per_channel' and descriptor in PH_BASED_DESCRIPTORS:
                # For per_channel, we need to handle the combined diagrams
                features = extract_descriptor(
                    descriptor, None, diagrams, params, expected_dim_total,
                )
            else:
                images_for_extract = images_gray if color_method != 'per_channel' else images_rgb
                features = extract_descriptor(
                    descriptor, images_for_extract, diagrams, params, expected_dim_total,
                )
            cv_result = evaluate_with_cv(
                features, labels, n_folds=n_folds, seed=seed,
                dataset_name=dataset_name,
            )

        # Check for OOM
        if cv_result.get('oom', False):
            return {
                'accuracy': -1.0,
                'std': 0.0,
                'oom': True,
                'time': time.time() - start,
            }

        elapsed = time.time() - start
        print(f"      {color_method}: {cv_result['mean']:.3f} ± {cv_result['std']:.3f} ({elapsed:.1f}s)")

        return {
            'accuracy': cv_result['mean'],
            'std': cv_result['std'],
            'time': elapsed,
            'params': params,
            'dim': expected_dim_total,
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"      {color_method}: ERROR - {e}")
        return {
            'accuracy': 0.0,
            'std': 0.0,
            'error': str(e),
            'time': elapsed,
        }


def run_color_experiment(
    object_type: str,
    datasets: List[str],
    descriptors: List[str] = None,
    color_methods: List[str] = None,
    n_samples: int = 1000,
    n_folds: int = 5,
    seed: int = 42,
    n_jobs: int = 4,
) -> Dict:
    """Run color experiment for one object type."""

    if descriptors is None:
        descriptors = COLOR_EXPERIMENT_DESCRIPTORS

    if color_methods is None:
        # Simplified: grayscale (baseline) vs per_channel (full color)
        # hsv_saturation optional - uncomment if color purity matters
        color_methods = ['grayscale', 'per_channel']  # , 'hsv_saturation']

    results = {
        'object_type': object_type,
        'datasets': datasets,
        'descriptors': descriptors,
        'color_methods': color_methods,
        'n_samples': n_samples,
        'results': {},
    }

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset} (object_type: {object_type})")
        print(f"{'='*60}")

        # Load dataset (keep RGB)
        images_rgb, labels = load_dataset_rgb(dataset, n_samples=n_samples, seed=seed)

        # Check if images are actually RGB
        is_rgb = images_rgb.ndim == 4 and images_rgb.shape[3] == 3
        if not is_rgb:
            print(f"  Dataset is grayscale - only testing grayscale method")
            color_methods_to_test = ['grayscale']
        else:
            color_methods_to_test = color_methods
            print(f"  Dataset is RGB - testing {len(color_methods_to_test)} color methods")

        results['results'][dataset] = {
            'is_rgb': is_rgb,
            'image_shape': images_rgb.shape,
            'descriptors': {},
        }

        for descriptor in descriptors:
            print(f"\n  Descriptor: {descriptor}")
            results['results'][dataset]['descriptors'][descriptor] = {}

            for color_method in color_methods_to_test:
                result = evaluate_color_method(
                    descriptor=descriptor,
                    images_rgb=images_rgb,
                    labels=labels,
                    color_method=color_method,
                    n_folds=n_folds,
                    seed=seed,
                    n_jobs=n_jobs,
                    dataset_name=dataset,
                )
                results['results'][dataset]['descriptors'][descriptor][color_method] = result

        # Memory cleanup
        del images_rgb
        gc.collect()

    return results


def _load_external_rgb(dataset_name: str, n_samples: int, seed: int, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load external dataset keeping RGB channels."""
    from PIL import Image
    import os

    # Import paths from config
    from RuleBenchmark.benchmark3.config import (
        ISIC_PATH, KVASIR_PATH, BRAIN_TUMOR_PATH, BREAKHIS_PATH,
        NCT_CRC_PATH, MALARIA_PATH, GASHISSDB_PATH, APTOS_PATH,
    )

    # Dataset path mapping
    path_map = {
        'ISIC2019': ISIC_PATH,
        'Kvasir': KVASIR_PATH,
        'BrainTumorMRI': BRAIN_TUMOR_PATH,
        'BreakHis': BREAKHIS_PATH,
        'NCT_CRC_HE': NCT_CRC_PATH,
        'MalariaCell': MALARIA_PATH,
        'GasHisSDB': GASHISSDB_PATH / '160',
        'APTOS2019': APTOS_PATH,
    }

    root_path = path_map.get(dataset_name)
    if root_path is None or not root_path.exists():
        print(f"  Warning: External dataset {dataset_name} path not found, using placeholder")
        np.random.seed(seed)
        n_classes = DATASETS.get(dataset_name, {}).get('n_classes', 2)
        images = np.random.rand(n_samples, target_size, target_size, 3).astype(np.float32)
        labels = np.random.randint(0, n_classes, n_samples)
        return images, labels

    # Find all images with class labels (folder-based structure)
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    all_paths = []
    all_labels = []
    class_names = []

    def find_images_recursive(path, depth=0, max_depth=4):
        if depth > max_depth:
            return
        try:
            for item in sorted(os.listdir(path)):
                item_path = path / item
                if item_path.is_file() and item.lower().endswith(valid_exts):
                    # Infer class from parent folder
                    class_name = path.name
                    if class_name not in class_names:
                        class_names.append(class_name)
                    all_paths.append(str(item_path))
                    all_labels.append(class_names.index(class_name))
                elif item_path.is_dir() and not item.startswith('.'):
                    find_images_recursive(item_path, depth + 1, max_depth)
        except Exception:
            pass

    find_images_recursive(root_path)

    if len(all_paths) == 0:
        print(f"  Warning: No images found in {root_path}, using placeholder")
        np.random.seed(seed)
        n_classes = DATASETS.get(dataset_name, {}).get('n_classes', 2)
        images = np.random.rand(n_samples, target_size, target_size, 3).astype(np.float32)
        labels = np.random.randint(0, n_classes, n_samples)
        return images, labels

    all_labels = np.array(all_labels)

    # Stratified sampling
    np.random.seed(seed)
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    indices = []
    for i, label in enumerate(sorted(unique_labels)):
        class_idx = np.where(all_labels == label)[0]
        n = per_class + (1 if i < remainder else 0)
        n = min(n, len(class_idx))
        chosen = np.random.choice(class_idx, n, replace=False)
        indices.extend(chosen)

    # Load images keeping RGB
    images = []
    labels = []
    for idx in indices:
        try:
            img = Image.open(all_paths[idx])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if img.size[0] != target_size or img.size[1] != target_size:
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(all_labels[idx])
        except Exception as e:
            print(f"    Warning: Failed to load {all_paths[idx]}: {e}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    print(f"    Loaded {len(images)} RGB images, shape={images.shape}, classes={n_classes}")
    return images, labels


def load_dataset_rgb(dataset_name: str, n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset keeping RGB channels (no grayscale conversion).

    For MedMNIST: loads from NPZ files (RGB stored natively)
    For External: loads images and keeps RGB channels
    """
    from PIL import Image
    import os

    dataset_info = DATASETS.get(dataset_name, {})
    source = dataset_info.get('source', 'medmnist')
    native_size = dataset_info.get('image_size', 224)

    # Adaptive resize: >2048 -> 1024, else keep original
    if isinstance(native_size, str):
        target_size = 224  # Default for variable
    elif native_size > 2048:
        target_size = 1024
    else:
        target_size = native_size

    if source != 'medmnist':
        # Load external dataset with RGB using data_loader infrastructure
        return _load_external_rgb(dataset_name, n_samples, seed, target_size)

    # MedMNIST loading
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
    }

    file_name = name_map.get(dataset_name, dataset_name.lower())

    # Try different resolutions
    for res in [224, 28]:
        npz_path = MEDMNIST_PATH / f"{file_name}_{res}.npz"
        if npz_path.exists():
            break
        npz_path = MEDMNIST_PATH / f"{file_name}.npz"
        if npz_path.exists():
            break

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")

    print(f"  Loading {dataset_name} from {npz_path.name} (keeping RGB)...")
    data = np.load(npz_path, mmap_mode='r')

    all_images = data['train_images']
    all_labels = data['train_labels'].flatten()

    # Stratified sampling
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

    # Keep as float32 but don't convert to grayscale
    images = images.astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0

    print(f"    Loaded {len(images)} images, shape={images.shape}, classes={n_classes}")
    return images, labels


def analyze_color_results(results: Dict) -> Dict:
    """Analyze color experiment results to derive rules."""

    analysis = {
        'object_type': results['object_type'],
        'best_methods': {},
        'method_rankings': {},
        'recommendations': {},
    }

    # Aggregate accuracies by color method across datasets and descriptors
    method_accuracies = {}

    for dataset, ds_results in results['results'].items():
        for descriptor, desc_results in ds_results.get('descriptors', {}).items():
            for method, metrics in desc_results.items():
                acc = metrics.get('accuracy', 0)
                if acc >= 0:  # Ignore OOM/errors
                    if method not in method_accuracies:
                        method_accuracies[method] = []
                    method_accuracies[method].append(acc)

    # Compute mean accuracy per method
    method_means = {
        method: np.mean(accs) if accs else 0
        for method, accs in method_accuracies.items()
    }

    # Rank methods
    sorted_methods = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
    analysis['method_rankings'] = {m: rank+1 for rank, (m, _) in enumerate(sorted_methods)}

    # Best method
    if sorted_methods:
        best_method, best_acc = sorted_methods[0]
        analysis['best_methods'] = {
            'method': best_method,
            'mean_accuracy': best_acc,
            'all_means': method_means,
        }

        # Compare to grayscale
        grayscale_acc = method_means.get('grayscale', 0)
        improvement = best_acc - grayscale_acc

        analysis['recommendations'] = {
            'best_method': best_method,
            'improvement_vs_grayscale': improvement,
            'recommendation': best_method if improvement > 0.02 else 'grayscale',
            'reason': f"+{improvement:.1%} vs grayscale" if improvement > 0.02 else "No significant improvement",
        }

    return analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 4.3: Color Handling Experiment")
    parser.add_argument('--trial', action='store_true', help='Quick trial')
    parser.add_argument('--full', action='store_true', help='Full experiment')
    parser.add_argument('--object-type', type=str, default=None,
                       help='Specific object type')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Samples per dataset')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    print("=" * 70)
    print("  EXPERIMENT 4.3: COLOR HANDLING EXPERIMENT")
    print("=" * 70)
    print(f"  n_samples: {args.n_samples}")
    print(f"  n_folds: {args.n_folds}")
    print(f"  Descriptors: {COLOR_EXPERIMENT_DESCRIPTORS}")
    print("=" * 70)

    all_results = {}

    # Determine what to run
    if args.dataset:
        # Single dataset
        from RuleBenchmark.benchmark3.exp4_config import DATASET_OBJECT_TYPE
        obj_type = DATASET_OBJECT_TYPE.get(args.dataset, 'unknown')
        object_types_to_run = {obj_type: [args.dataset]}
    elif args.object_type:
        # Single object type
        object_types_to_run = {args.object_type: TRIAL_DATASETS.get(args.object_type, [])}
    elif args.trial:
        # Trial: one dataset per object type
        object_types_to_run = TRIAL_DATASETS
    else:
        # Full: all datasets
        object_types_to_run = OBJECT_TYPE_DATASETS

    for obj_type, datasets in object_types_to_run.items():
        print(f"\n\n{'#'*70}")
        print(f"  OBJECT TYPE: {obj_type}")
        print(f"  Expected best: {COLOR_HANDLING_RULES.get(obj_type, {}).get('recommended', 'unknown')}")
        print(f"{'#'*70}")

        results = run_color_experiment(
            object_type=obj_type,
            datasets=datasets,
            n_samples=args.n_samples,
            n_folds=args.n_folds,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )

        analysis = analyze_color_results(results)
        results['analysis'] = analysis
        all_results[obj_type] = results

        # Print summary
        print(f"\n  COLOR RECOMMENDATION for {obj_type}:")
        print(f"    Best method: {analysis['recommendations'].get('best_method', 'N/A')}")
        print(f"    vs grayscale: {analysis['recommendations'].get('improvement_vs_grayscale', 0):+.1%}")

    # Save results
    # When running a single object type, include it in filename to avoid overwriting
    if args.output:
        output_path = Path(args.output)
    elif args.object_type:
        output_path = RESULTS_PATH / f"exp4_color_{args.object_type}_n{args.n_samples}.json"
    else:
        output_path = RESULTS_PATH / f"exp4_color_experiment_n{args.n_samples}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file before overwriting
    if output_path.exists():
        import shutil
        import time as _time
        backup_path = output_path.with_suffix(f'.backup_{int(_time.time())}.json')
        shutil.copy2(output_path, backup_path)
        print(f"\n  Backed up existing file to: {backup_path}")

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    # Final summary
    print(f"\n\n{'='*70}")
    print("  FINAL COLOR RULES")
    print(f"{'='*70}")
    for obj_type, results in all_results.items():
        rec = results.get('analysis', {}).get('recommendations', {})
        print(f"  {obj_type}: {rec.get('recommendation', 'N/A')} ({rec.get('reason', '')})")


if __name__ == '__main__':
    main()
