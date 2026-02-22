"""
Convergence Analysis for TopoBenchmark.

Determines the minimum sample size per dataset where balanced accuracy
stabilizes, then creates evidence for a frozen evaluation dataset.

Pipeline: image → compute_ph → descriptor(optimal_params) → TabPFN → balanced_accuracy
Classifier: TabPFN everywhere. PCA(500) if dim > 2000. No XGBoost confound.

Addresses MICCAI reviewer concerns:
  1. Ranking stability (Spearman rho of top-5 descriptor rankings at n vs n_max)
  2. Minimum n per class: n_min = max(50, 10 × n_classes)
  3. Seed sensitivity: 3 seeds per n-value, report mean ± std
  4. Tests top-3 descriptors per dataset (from benchmark4 ground truth)
  5. Convergence criterion: |acc(n) - acc(n_max)| < 1% for ALL top-3 AND std < 2%
  6. Class-balance check: flag if any class has < 5 samples

Usage:
    python TopoBenchmark/convergence_analysis.py \
        --dataset BloodMNIST --descriptors top3 --n-seeds 3
"""

import sys
import json
import time
import pickle
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from RuleBenchmark.benchmark4.data_loader import load_dataset, rgb_to_channels
from RuleBenchmark.benchmark4.descriptor_runner import (
    extract_features, extract_features_per_channel, extract_learned_features,
    PH_BASED, IMAGE_BASED, LEARNED,
)
from RuleBenchmark.benchmark4.classifier_wrapper import get_classifier
from RuleBenchmark.benchmark4.optimal_rules import get_rules
from RuleBenchmark.benchmark4.config import (
    DATASETS, ALL_DESCRIPTORS, PH_CACHE_PATH, get_object_type, get_color_mode,
)
from TopoBenchmark.ground_truth import load_ground_truth

# Suppress warnings from TabPFN/sklearn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# Output paths
# =============================================================================
CONVERGENCE_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "convergence"
CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)

STANDARD_N_VALUES = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000]


# =============================================================================
# Helper functions
# =============================================================================

def get_n_values(dataset: str, n_max_override: Optional[int] = None) -> List[int]:
    """Get valid n-values for a dataset.

    n_min = max(50, 10 × n_classes)
    n_max = n_max_override or DATASETS[dataset]['n_samples']
    Filter standard grid to [n_min, n_max], always include n_max.
    """
    cfg = DATASETS[dataset]
    n_classes = cfg['n_classes']
    n_max = n_max_override if n_max_override else cfg['n_samples']
    n_min = max(50, 10 * n_classes)

    values = [n for n in STANDARD_N_VALUES if n_min <= n <= n_max]

    # Always include n_max as final point
    if n_max not in values:
        values.append(n_max)
    values.sort()

    return values


def get_top_k_descriptors(dataset: str, k: int = 3) -> List[str]:
    """Get top-k descriptors for a dataset from benchmark4 ground truth."""
    gt = load_ground_truth()
    return gt.get_top_n_descriptors(dataset, n=k)


def stratified_subsample(
    images: np.ndarray,
    labels: np.ndarray,
    n: int,
    seed: int,
    diagrams_gray: Optional[List] = None,
    diagrams_R: Optional[List] = None,
    diagrams_G: Optional[List] = None,
    diagrams_B: Optional[List] = None,
) -> Tuple:
    """Stratified subsample of n images from the full dataset.

    Maintains class proportions. Returns subsampled versions of all
    provided arrays/lists.
    """
    rng = np.random.RandomState(seed)
    n_total = len(labels)

    if n >= n_total:
        # Return everything
        return (images, labels, diagrams_gray, diagrams_R, diagrams_G, diagrams_B,
                list(range(n_total)))

    # Stratified sampling: sample proportionally from each class
    classes, class_counts = np.unique(labels, return_counts=True)
    class_proportions = class_counts / n_total

    indices = []
    for cls, prop in zip(classes, class_proportions):
        cls_indices = np.where(labels == cls)[0]
        n_cls = max(1, int(round(prop * n)))  # at least 1 per class
        if n_cls > len(cls_indices):
            n_cls = len(cls_indices)
        chosen = rng.choice(cls_indices, n_cls, replace=False)
        indices.extend(chosen)

    # Adjust if we overshot or undershot
    indices = np.array(indices)
    if len(indices) > n:
        indices = rng.choice(indices, n, replace=False)
    elif len(indices) < n:
        remaining = np.setdiff1d(np.arange(n_total), indices)
        extra = rng.choice(remaining, n - len(indices), replace=False)
        indices = np.concatenate([indices, extra])

    indices = np.sort(indices)

    sub_images = images[indices]
    sub_labels = labels[indices]
    sub_diags_gray = [diagrams_gray[i] for i in indices] if diagrams_gray is not None else None
    sub_diags_R = [diagrams_R[i] for i in indices] if diagrams_R is not None else None
    sub_diags_G = [diagrams_G[i] for i in indices] if diagrams_G is not None else None
    sub_diags_B = [diagrams_B[i] for i in indices] if diagrams_B is not None else None

    return (sub_images, sub_labels, sub_diags_gray, sub_diags_R, sub_diags_G,
            sub_diags_B, indices)


def check_class_balance(labels: np.ndarray, n: int, min_per_class: int = 5) -> List[str]:
    """Check class balance and return warning flags."""
    flags = []
    counts = Counter(labels.tolist())
    for cls, count in sorted(counts.items()):
        if count < min_per_class:
            flags.append(f"n={n}: class {cls} has only {count} samples")
    return flags


def load_ph_cache(dataset: str, n_max_override: Optional[int] = None) -> Optional[Dict]:
    """Load precomputed PH cache from benchmark4.

    Tries n_max_override first, falls back to default n_max.
    """
    n_default = DATASETS[dataset]['n_samples']
    # Try override first, then default
    candidates = []
    if n_max_override and n_max_override != n_default:
        candidates.append(n_max_override)
    candidates.append(n_default)

    for n in candidates:
        cache_path = PH_CACHE_PATH / f"{dataset}_n{n}.pkl"
        if cache_path.exists():
            print(f"  Loading PH cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    return None


def compute_ph_fresh(images_gray: np.ndarray) -> List[Dict]:
    """Compute PH from scratch using best available library."""
    from RuleBenchmark.benchmark4.precompute_ph import compute_ph
    diagrams, lib = compute_ph(images_gray)
    print(f"  PH computed with {lib} for {len(images_gray)} images")
    return diagrams


# =============================================================================
# Core evaluation
# =============================================================================

def run_single_point(
    dataset: str,
    descriptor: str,
    n: int,
    seed: int,
    n_folds: int = 5,
    # Pre-loaded data (avoid redundant I/O)
    images: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    diagrams_gray: Optional[List] = None,
    diagrams_R: Optional[List] = None,
    diagrams_G: Optional[List] = None,
    diagrams_B: Optional[List] = None,
    # Pre-extracted features for non-learned descriptors (n_max × D)
    precomputed_features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run a single convergence evaluation point.

    If precomputed_features is provided (for non-learned descriptors),
    subsample the feature matrix by indices instead of re-extracting.

    Returns: {mean_bal_acc, std_bal_acc, fold_scores, n_actual, feature_dim,
              class_counts, class_balance_flags}
    """
    cfg = DATASETS[dataset]
    color_mode = cfg['color_mode']
    object_type = cfg['object_type']
    n_classes = cfg['n_classes']

    # Subsample
    (sub_images, sub_labels, sub_diags_gray, sub_diags_R, sub_diags_G,
     sub_diags_B, indices) = stratified_subsample(
        images, labels, n, seed,
        diagrams_gray, diagrams_R, diagrams_G, diagrams_B,
    )

    n_actual = len(sub_labels)
    class_counts = dict(Counter(sub_labels.tolist()))
    balance_flags = check_class_balance(sub_labels, n)

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)  # fit on FULL labels to ensure all classes present
    sub_labels_enc = le.transform(sub_labels)

    # Pre-slice features if available (non-learned descriptors)
    sub_features = None
    if precomputed_features is not None:
        sub_features = precomputed_features[indices]

    # Get optimal params (needed for learned descriptors)
    is_learned = descriptor in LEARNED
    if is_learned or sub_features is None:
        rules = get_rules()
        desc_cfg = rules.get_descriptor_config(descriptor, object_type, color_mode)
        params = desc_cfg['params']
        dim_per_channel = desc_cfg['dim_per_channel']
        total_dim = desc_cfg['total_dim']

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(sub_images, sub_labels_enc)):
        train_labels = sub_labels_enc[train_idx]
        test_labels = sub_labels_enc[test_idx]

        # Extract features
        if sub_features is not None:
            # Pre-extracted: just slice by fold indices
            X_train = sub_features[train_idx]
            X_test = sub_features[test_idx]
        elif is_learned:
            # Learned descriptors need train/test split at extraction
            if color_mode == 'per_channel' and sub_diags_R is not None:
                X_train_parts, X_test_parts = [], []
                for ch_diags in [sub_diags_R, sub_diags_G, sub_diags_B]:
                    train_diags = [ch_diags[i] for i in train_idx]
                    test_diags = [ch_diags[i] for i in test_idx]
                    Xtr, Xte = extract_learned_features(
                        descriptor, train_diags, test_diags,
                        params, dim_per_channel,
                    )
                    X_train_parts.append(Xtr)
                    X_test_parts.append(Xte)
                X_train = np.hstack(X_train_parts)
                X_test = np.hstack(X_test_parts)
            else:
                train_diags = [sub_diags_gray[i] for i in train_idx]
                test_diags = [sub_diags_gray[i] for i in test_idx]
                X_train, X_test = extract_learned_features(
                    descriptor, train_diags, test_diags,
                    params, total_dim,
                )
        else:
            # Fallback: extract from raw data (shouldn't normally reach here)
            if descriptor in IMAGE_BASED:
                if color_mode == 'per_channel':
                    features = extract_features_per_channel(
                        descriptor,
                        images_R=sub_images[:, :, :, 0] if sub_images.ndim == 4 else sub_images,
                        images_G=sub_images[:, :, :, 1] if sub_images.ndim == 4 else sub_images,
                        images_B=sub_images[:, :, :, 2] if sub_images.ndim == 4 else sub_images,
                        params=params,
                        expected_dim_per_channel=dim_per_channel,
                    )
                else:
                    features = extract_features(
                        descriptor, images=sub_images,
                        params=params, expected_dim=total_dim,
                    )
            elif descriptor in PH_BASED:
                if color_mode == 'per_channel' and sub_diags_R is not None:
                    features = extract_features_per_channel(
                        descriptor,
                        diags_R=sub_diags_R, diags_G=sub_diags_G, diags_B=sub_diags_B,
                        params=params,
                        expected_dim_per_channel=dim_per_channel,
                    )
                else:
                    features = extract_features(
                        descriptor, diagrams=sub_diags_gray,
                        params=params, expected_dim=total_dim,
                    )
            else:
                raise ValueError(f"Unknown descriptor type: {descriptor}")
            X_train = features[train_idx]
            X_test = features[test_idx]

        actual_dim = X_train.shape[1]

        # Classifier: TabPFN everywhere. PCA-bagging handles dim > 2000.
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clf = get_classifier(
            'TabPFN',
            n_features=actual_dim,
            n_classes=n_classes,
            seed=seed + fold_idx,
            device=device,
        )

        clf.fit(X_train, train_labels)
        preds = clf.predict(X_test)
        bal_acc = balanced_accuracy_score(test_labels, preds)
        fold_scores.append(float(bal_acc))

    return {
        'mean_bal_acc': float(np.mean(fold_scores)),
        'std_bal_acc': float(np.std(fold_scores)),
        'fold_scores': fold_scores,
        'n_actual': n_actual,
        'feature_dim': actual_dim,
        'class_counts': {str(k): int(v) for k, v in class_counts.items()},
        'class_balance_flags': balance_flags,
    }


# =============================================================================
# Full convergence run
# =============================================================================

def run_convergence(
    dataset: str,
    descriptors: List[str],
    n_values: List[int],
    n_seeds: int = 3,
    n_folds: int = 5,
    output_path: Optional[Path] = None,
    n_max_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full convergence analysis for one dataset.

    Memory-efficient: loads PH from cache first (small), only loads full images
    when needed for image-based descriptors. Frees images after each descriptor.

    Args:
        n_max_override: If set, use this instead of DATASETS[dataset]['n_samples'].
            Requires a PH cache at this n (e.g., {dataset}_n10000.pkl).
    """
    import gc

    cfg = DATASETS[dataset]
    n_max = n_max_override if n_max_override else cfg['n_samples']
    n_classes = cfg['n_classes']
    color_mode = cfg['color_mode']

    if output_path is None:
        output_path = CONVERGENCE_DIR / f"{dataset}_convergence.json"

    print(f"\n{'='*60}")
    print(f"  Convergence Analysis: {dataset}")
    print(f"  n_max={n_max}, n_classes={n_classes}, color={color_mode}")
    print(f"  descriptors: {descriptors}")
    print(f"  n_values: {n_values}")
    print(f"  n_seeds={n_seeds}, n_folds={n_folds}")
    print(f"{'='*60}")

    # Separate descriptors by type
    ph_descriptors = [d for d in descriptors if d in PH_BASED or d in LEARNED]
    img_descriptors = [d for d in descriptors if d in IMAGE_BASED]
    learned_ph = [d for d in ph_descriptors if d in LEARNED]
    non_learned_ph = [d for d in ph_descriptors if d not in LEARNED]

    object_type = cfg['object_type']

    # Step 1: Load PH cache (compact — diagrams are small lists of numpy arrays)
    diagrams_gray = None
    diagrams_R = None
    diagrams_G = None
    diagrams_B = None
    labels = None
    class_names = None

    if ph_descriptors:
        cache = load_ph_cache(dataset, n_max_override=n_max_override)
        if cache is not None:
            diagrams_gray = cache.get('diagrams_gray_sublevel')
            labels = np.array(cache.get('labels'))
            class_names = cache.get('class_names')
            if color_mode == 'per_channel':
                diagrams_R = cache.get('diagrams_R_sublevel')
                diagrams_G = cache.get('diagrams_G_sublevel')
                diagrams_B = cache.get('diagrams_B_sublevel')
            print(f"  PH loaded from cache ({len(diagrams_gray)} diagrams)")
            # Free the raw cache dict but keep the extracted data
            del cache
            gc.collect()
        else:
            print(f"  WARNING: No PH cache found for {dataset}. "
                  f"Loading images to compute PH...")
            t0 = time.time()
            images_tmp, labels, class_names = load_dataset(
                dataset, n_samples=n_max, seed=42)
            print(f"  Loaded {len(labels)} images in {time.time()-t0:.1f}s")
            if color_mode == 'per_channel' and images_tmp.ndim == 4:
                imgs_R, imgs_G, imgs_B = rgb_to_channels(images_tmp)
                imgs_gray = np.mean(images_tmp.astype(np.float32) / 255.0, axis=-1)
                del images_tmp; gc.collect()
                diagrams_gray = compute_ph_fresh(imgs_gray)
                del imgs_gray; gc.collect()
                diagrams_R = compute_ph_fresh(imgs_R.astype(np.float32) / 255.0)
                del imgs_R; gc.collect()
                diagrams_G = compute_ph_fresh(imgs_G.astype(np.float32) / 255.0)
                del imgs_G; gc.collect()
                diagrams_B = compute_ph_fresh(imgs_B.astype(np.float32) / 255.0)
                del imgs_B; gc.collect()
            else:
                imgs_gray = images_tmp.astype(np.float32) / 255.0
                del images_tmp; gc.collect()
                if imgs_gray.ndim == 4:
                    imgs_gray = np.mean(imgs_gray, axis=-1)
                diagrams_gray = compute_ph_fresh(imgs_gray)
                del imgs_gray; gc.collect()
            labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    # If we only have image-based descriptors, load labels from dataset
    if labels is None:
        # Lightweight: just load labels without full images
        # load_dataset with n_samples=0 isn't supported, so load small then expand
        _, labels, class_names = load_dataset(dataset, n_samples=n_max, seed=42)
        print(f"  Loaded labels for {len(labels)} samples")

    # =========================================================================
    # Pre-extract features for non-learned descriptors (extract once, reuse)
    # This avoids redundant extraction across (n, seed, fold) combinations.
    # Learned descriptors (ATOL, persistence_codebook) still need per-fold
    # extraction because they fit on training data.
    # =========================================================================
    precomputed = {}
    rules = get_rules()

    if non_learned_ph and diagrams_gray is not None:
        print(f"\n  Pre-extracting features for {len(non_learned_ph)} "
              f"non-learned PH descriptors...")
        for desc in non_learned_ph:
            t0 = time.time()
            desc_cfg = rules.get_descriptor_config(desc, object_type, color_mode)
            if color_mode == 'per_channel' and diagrams_R is not None:
                features = extract_features_per_channel(
                    desc,
                    diags_R=diagrams_R, diags_G=diagrams_G, diags_B=diagrams_B,
                    params=desc_cfg['params'],
                    expected_dim_per_channel=desc_cfg['dim_per_channel'],
                )
            else:
                features = extract_features(
                    desc, diagrams=diagrams_gray,
                    params=desc_cfg['params'], expected_dim=desc_cfg['total_dim'],
                )
            precomputed[desc] = features
            print(f"    {desc}: {features.shape} in {time.time()-t0:.1f}s")

    # Initialize results structure
    results = {
        'dataset': dataset,
        'n_max': n_max,
        'n_classes': n_classes,
        'color_mode': color_mode,
        'class_names': class_names if isinstance(class_names, list) else list(class_names),
        'descriptors_tested': descriptors,
        'n_values': n_values,
        'n_seeds': n_seeds,
        'n_folds': n_folds,
        'classifier': 'TabPFN',
        'results': {},
        'class_balance_flags': [],
        'timing': {},
    }

    total_evals = len(descriptors) * len(n_values) * n_seeds
    eval_count = 0
    dummy_images = np.zeros((len(labels), 1, 1), dtype=np.float32)

    # Process PH-based descriptors first (no images needed)
    for desc in ph_descriptors:
        is_precomputed = desc in precomputed
        label = "precomputed" if is_precomputed else "PH-based, learned"
        print(f"\n  --- Descriptor: {desc} ({label}) ---")
        results['results'][desc] = {}
        t_desc = time.time()

        for n in n_values:
            results['results'][desc][str(n)] = {}
            seed_scores = []

            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx * 100
                eval_count += 1
                print(f"    [{eval_count}/{total_evals}] n={n}, seed={seed}...",
                      end=" ", flush=True)

                t_point = time.time()
                try:
                    point = run_single_point(
                        dataset, desc, n, seed, n_folds,
                        images=dummy_images, labels=labels,
                        class_names=class_names,
                        diagrams_gray=diagrams_gray,
                        diagrams_R=diagrams_R, diagrams_G=diagrams_G,
                        diagrams_B=diagrams_B,
                        precomputed_features=precomputed.get(desc),
                    )
                    seed_scores.append(point['mean_bal_acc'])
                    results['class_balance_flags'].extend(point['class_balance_flags'])
                    elapsed = time.time() - t_point
                    print(f"acc={point['mean_bal_acc']:.4f} ({elapsed:.1f}s)")
                except Exception as e:
                    import traceback
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    seed_scores.append(None)

            valid_scores = [s for s in seed_scores if s is not None]
            if valid_scores:
                results['results'][desc][str(n)] = {
                    'mean': float(np.mean(valid_scores)),
                    'std': float(np.std(valid_scores)),
                    'seeds': seed_scores,
                    'class_counts': point['class_counts'] if seed_scores[-1] is not None else {},
                    'feature_dim': point.get('feature_dim', 0),
                }
            else:
                results['results'][desc][str(n)] = {
                    'mean': None, 'std': None, 'seeds': seed_scores,
                    'class_counts': {}, 'feature_dim': 0,
                }

        dt = time.time() - t_desc
        results['timing'][desc] = round(dt, 1)
        print(f"  {desc} done in {dt:.1f}s")

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {output_path}")

    # Free PH data and non-learned PH features
    del diagrams_gray, diagrams_R, diagrams_G, diagrams_B
    for desc in non_learned_ph:
        precomputed.pop(desc, None)
    gc.collect()

    # Process image-based descriptors (need to load full images)
    if img_descriptors:
        print(f"\n  Loading images for image-based descriptors...")
        t0 = time.time()
        images, labels, class_names = load_dataset(dataset, n_samples=n_max, seed=42)
        print(f"  Loaded {len(labels)} images in {time.time()-t0:.1f}s "
              f"(shape={images.shape})")

        # Pre-extract features for all image-based descriptors (extract once)
        print(f"  Pre-extracting features for {len(img_descriptors)} "
              f"image-based descriptors...")
        for desc in img_descriptors:
            t0 = time.time()
            desc_cfg = rules.get_descriptor_config(desc, object_type, color_mode)
            if color_mode == 'per_channel' and images.ndim == 4:
                features = extract_features_per_channel(
                    desc,
                    images_R=images[:, :, :, 0],
                    images_G=images[:, :, :, 1],
                    images_B=images[:, :, :, 2],
                    params=desc_cfg['params'],
                    expected_dim_per_channel=desc_cfg['dim_per_channel'],
                )
            else:
                features = extract_features(
                    desc, images=images,
                    params=desc_cfg['params'], expected_dim=desc_cfg['total_dim'],
                )
            precomputed[desc] = features
            print(f"    {desc}: {features.shape} in {time.time()-t0:.1f}s")

        # Free images — only need feature matrices from here
        del images
        gc.collect()

        dummy_images = np.zeros((len(labels), 1, 1), dtype=np.float32)

        for desc in img_descriptors:
            print(f"\n  --- Descriptor: {desc} (image-based, precomputed) ---")
            results['results'][desc] = {}
            t_desc = time.time()

            for n in n_values:
                results['results'][desc][str(n)] = {}
                seed_scores = []

                for seed_idx in range(n_seeds):
                    seed = 42 + seed_idx * 100
                    eval_count += 1
                    print(f"    [{eval_count}/{total_evals}] n={n}, seed={seed}...",
                          end=" ", flush=True)

                    t_point = time.time()
                    try:
                        point = run_single_point(
                            dataset, desc, n, seed, n_folds,
                            images=dummy_images, labels=labels,
                            class_names=class_names,
                            precomputed_features=precomputed[desc],
                        )
                        seed_scores.append(point['mean_bal_acc'])
                        results['class_balance_flags'].extend(point['class_balance_flags'])
                        elapsed = time.time() - t_point
                        print(f"acc={point['mean_bal_acc']:.4f} ({elapsed:.1f}s)")
                    except Exception as e:
                        import traceback
                        print(f"ERROR: {e}")
                        traceback.print_exc()
                        seed_scores.append(None)

                valid_scores = [s for s in seed_scores if s is not None]
                if valid_scores:
                    results['results'][desc][str(n)] = {
                        'mean': float(np.mean(valid_scores)),
                        'std': float(np.std(valid_scores)),
                        'seeds': seed_scores,
                        'class_counts': point['class_counts'] if seed_scores[-1] is not None else {},
                        'feature_dim': point.get('feature_dim', 0),
                    }
                else:
                    results['results'][desc][str(n)] = {
                        'mean': None, 'std': None, 'seeds': seed_scores,
                        'class_counts': {}, 'feature_dim': 0,
                    }

            dt = time.time() - t_desc
            results['timing'][desc] = round(dt, 1)
            print(f"  {desc} done in {dt:.1f}s")

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved to {output_path}")

        # Free precomputed image features
        for desc in img_descriptors:
            precomputed.pop(desc, None)
        gc.collect()

    # Deduplicate class balance flags
    results['class_balance_flags'] = sorted(set(results['class_balance_flags']))

    # Add convergence analysis
    results['convergence'] = check_convergence(results)
    results['convergence']['ranking_spearman'] = check_ranking_stability(results, n_values)

    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Final results saved to {output_path}")

    return results


# =============================================================================
# Convergence criteria
# =============================================================================

def check_convergence(
    results: Dict,
    epsilon: float = 0.01,
    max_seed_std: float = 0.02,
) -> Dict[str, Any]:
    """Check convergence criterion.

    Converged when:
      1. |acc(n) - acc(n_max)| < epsilon for ALL top-3 descriptors
      2. cross-seed std < max_seed_std at that n

    Returns: {epsilon, max_seed_std, convergence_n, per_descriptor}
    """
    n_values = results['n_values']
    n_max = str(n_values[-1])  # Last n-value is n_max
    descriptors = results['descriptors_tested']

    per_descriptor = {}
    convergence_ns = []

    for desc in descriptors:
        desc_results = results['results'].get(desc, {})
        ref_entry = desc_results.get(n_max, {})
        ref_acc = ref_entry.get('mean')

        if ref_acc is None:
            per_descriptor[desc] = {'convergence_n': None, 'reason': 'no reference accuracy'}
            continue

        conv_n = None
        for n in n_values:
            entry = desc_results.get(str(n), {})
            acc = entry.get('mean')
            std = entry.get('std')

            if acc is None:
                continue

            if abs(acc - ref_acc) < epsilon and (std is not None and std < max_seed_std):
                conv_n = n
                break

        per_descriptor[desc] = {
            'convergence_n': conv_n,
            'ref_accuracy': ref_acc,
        }
        if conv_n is not None:
            convergence_ns.append(conv_n)

    # Overall convergence = max across all descriptors
    overall_n = max(convergence_ns) if convergence_ns else None

    return {
        'epsilon': epsilon,
        'max_seed_std': max_seed_std,
        'convergence_n': overall_n,
        'per_descriptor': per_descriptor,
    }


def check_ranking_stability(
    results: Dict,
    n_values: List[int],
) -> Dict[str, Any]:
    """Check ranking stability via Spearman rho.

    At each n: rank descriptors by mean accuracy.
    Compute Spearman rho between ranking(n) and ranking(n_max).
    """
    descriptors = results['descriptors_tested']
    n_max = str(n_values[-1])

    # Get reference ranking at n_max
    ref_accs = []
    for desc in descriptors:
        entry = results['results'].get(desc, {}).get(n_max, {})
        ref_accs.append(entry.get('mean', 0) or 0)

    if len(set(ref_accs)) <= 1:
        # All same accuracy — can't compute meaningful ranking
        return {'rho_per_n': {}, 'ranking_convergence_n': None}

    rho_per_n = {}
    ranking_conv_n = None

    for n in n_values:
        n_str = str(n)
        accs = []
        for desc in descriptors:
            entry = results['results'].get(desc, {}).get(n_str, {})
            accs.append(entry.get('mean', 0) or 0)

        if len(set(accs)) <= 1:
            rho_per_n[n_str] = None
            continue

        rho, _ = spearmanr(ref_accs, accs)
        rho_per_n[n_str] = round(float(rho), 4) if not np.isnan(rho) else None

        if ranking_conv_n is None and rho_per_n[n_str] is not None and rho_per_n[n_str] > 0.9:
            ranking_conv_n = n

    return {
        'rho_per_n': rho_per_n,
        'ranking_convergence_n': ranking_conv_n,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convergence analysis for TopoBenchmark')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., BloodMNIST)')
    parser.add_argument('--descriptors', type=str, default='top3',
                        help='Descriptors: "top3" or comma-separated names')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of top descriptors (if --descriptors=top3)')
    parser.add_argument('--n-values', type=str, default=None,
                        help='Comma-separated n values (default: auto from dataset)')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of seeds per n-value')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n-max', type=int, default=None,
                        help='Override n_max (default: from config). '
                             'Requires PH cache at this n.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: auto)')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset
    if dataset not in DATASETS:
        print(f"ERROR: Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    # Determine descriptors
    if args.descriptors == 'top3' or args.descriptors.startswith('top'):
        k = args.k
        if args.descriptors.startswith('top') and args.descriptors != 'top3':
            try:
                k = int(args.descriptors[3:])
            except ValueError:
                pass
        descriptors = get_top_k_descriptors(dataset, k=k)
        print(f"Top-{k} descriptors for {dataset}: {descriptors}")
    else:
        descriptors = [d.strip() for d in args.descriptors.split(',')]
        for d in descriptors:
            if d not in ALL_DESCRIPTORS:
                print(f"WARNING: Unknown descriptor '{d}'")

    # Determine n-values
    n_max_override = args.n_max
    if args.n_values:
        n_values = [int(x) for x in args.n_values.split(',')]
    else:
        n_values = get_n_values(dataset, n_max_override=n_max_override)

    # Output path — use different filename for extended runs
    if args.output:
        output_path = Path(args.output)
    elif n_max_override:
        output_path = CONVERGENCE_DIR / f"{dataset}_convergence_n{n_max_override}.json"
    else:
        output_path = None

    # Run
    results = run_convergence(
        dataset=dataset,
        descriptors=descriptors,
        n_values=n_values,
        n_seeds=args.n_seeds,
        n_folds=args.n_folds,
        output_path=output_path,
        n_max_override=n_max_override,
    )

    # Print summary
    conv = results.get('convergence', {})
    print(f"\n{'='*60}")
    print(f"  CONVERGENCE SUMMARY: {dataset}")
    print(f"{'='*60}")
    print(f"  Overall convergence n: {conv.get('convergence_n')}")
    for desc, info in conv.get('per_descriptor', {}).items():
        print(f"    {desc}: n={info.get('convergence_n')} "
              f"(ref_acc={info.get('ref_accuracy', 'N/A')})")

    ranking = conv.get('ranking_spearman', {})
    print(f"  Ranking convergence n: {ranking.get('ranking_convergence_n')}")
    for n_str, rho in ranking.get('rho_per_n', {}).items():
        print(f"    n={n_str}: rho={rho}")

    if results.get('class_balance_flags'):
        print(f"\n  Class balance warnings:")
        for flag in results['class_balance_flags']:
            print(f"    - {flag}")


if __name__ == '__main__':
    main()
