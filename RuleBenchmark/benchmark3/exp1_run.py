#!/usr/bin/env python
"""
Benchmark3 Experiment 1: Main Benchmark Runner.

Runs all 15 descriptors × 4 classifiers on a single dataset.
Pipeline: Load data → PH (GUDHI+joblib) → Extract descriptors → 5-fold CV → Save results.

Usage:
    python benchmarks/benchmark3/exp1_run.py --dataset BloodMNIST
    python benchmarks/benchmark3/exp1_run.py --dataset BloodMNIST --n-samples 5000 --n-jobs -1
    python benchmarks/benchmark3/exp1_run.py --dataset RetinaMNIST,BreakHis  # combined job
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import time
import json
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Dict, List, Any, Optional

from RuleBenchmark.benchmark3.config import (
    DESCRIPTORS, EXPECTED_DIMS, ACTIVE_CLASSIFIERS,
    PH_CONFIG, EVALUATION, EXECUTION, MANY_CLASS_DATASETS,
    get_classifier, get_output_dir,
)
from RuleBenchmark.benchmark3.data_loader import load_dataset


# =============================================================================
# PH COMPUTATION (GUDHI + joblib)
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
            if np.isfinite(d) and d > b
        ]
    return pd


def compute_ph(images: np.ndarray, n_jobs: int = -1) -> tuple:
    """Compute PH using GUDHI + joblib parallelization."""
    from joblib import Parallel, delayed

    print(f"  Computing PH with GUDHI + joblib (n_jobs={n_jobs}, {images.shape[1]}x{images.shape[2]})...")
    start = time.time()
    diagrams = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_gudhi_ph_single)(img) for img in images
    )
    elapsed = time.time() - start
    print(f"  PH done: {elapsed:.1f}s ({elapsed/len(images)*1000:.1f}ms/image)")
    return diagrams, elapsed


# =============================================================================
# DESCRIPTOR EXTRACTION (reuse from trial_run.py)
# =============================================================================

from RuleBenchmark.benchmark3.trial_run import (
    extract_descriptor,
)


# =============================================================================
# CLASSIFICATION WITH 5-FOLD CV
# =============================================================================

def evaluate_all_classifiers(
    features: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    seed: int = 42,
    n_folds: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all classifiers on one descriptor's features."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        balanced_accuracy_score, accuracy_score,
        f1_score, precision_score, recall_score, cohen_kappa_score,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone

    results = {}
    n_classes = len(np.unique(labels))

    for clf_name in ACTIVE_CLASSIFIERS:
        clf = get_classifier(clf_name, seed=seed, dataset_name=dataset_name, n_classes=n_classes)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_metrics = []
        total_train_time = 0
        total_predict_time = 0

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
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
                clf_copy = clone(clf)

                t0 = time.time()
                clf_copy.fit(X_train, y_train)
                train_time = time.time() - t0

                t0 = time.time()
                y_pred = clf_copy.predict(X_val)
                predict_time = time.time() - t0

                total_train_time += train_time
                total_predict_time += predict_time

                # Compute metrics
                n_classes = len(np.unique(labels))
                avg = 'macro' if n_classes > 2 else 'binary'

                metrics = {
                    'accuracy': float(accuracy_score(y_val, y_pred)),
                    'balanced_accuracy': float(balanced_accuracy_score(y_val, y_pred)),
                    'f1_macro': float(f1_score(y_val, y_pred, average='macro', zero_division=0)),
                    'f1_weighted': float(f1_score(y_val, y_pred, average='weighted', zero_division=0)),
                    'precision_macro': float(precision_score(y_val, y_pred, average='macro', zero_division=0)),
                    'recall_macro': float(recall_score(y_val, y_pred, average='macro', zero_division=0)),
                    'cohen_kappa': float(cohen_kappa_score(y_val, y_pred)),
                    'train_time': train_time,
                    'predict_time': predict_time,
                }
                fold_metrics.append(metrics)

            except Exception as e:
                print(f"    WARNING: {clf_name} fold {fold_idx} failed: {e}")
                fold_metrics.append({
                    'accuracy': 0.0, 'balanced_accuracy': 0.0,
                    'f1_macro': 0.0, 'f1_weighted': 0.0,
                    'precision_macro': 0.0, 'recall_macro': 0.0,
                    'cohen_kappa': 0.0,
                    'train_time': 0.0, 'predict_time': 0.0,
                    'error': str(e),
                })

        # Aggregate across folds
        metric_names = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
                        'precision_macro', 'recall_macro', 'cohen_kappa']
        agg = {}
        for m in metric_names:
            values = [fm[m] for fm in fold_metrics]
            agg[f'{m}_mean'] = float(np.mean(values))
            agg[f'{m}_std'] = float(np.std(values))

        agg['total_train_time'] = total_train_time
        agg['total_predict_time'] = total_predict_time
        agg['n_folds'] = n_folds
        agg['errors'] = [fm.get('error') for fm in fold_metrics if 'error' in fm]

        results[clf_name] = agg

    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_benchmark(
    dataset_name: str,
    n_samples: int = 5000,
    n_jobs: int = -1,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    descriptor_filter: Optional[str] = None,
):
    """Run full benchmark for one dataset.

    Pipeline:
      1. Load dataset (stratified, 224×224 grayscale)
      2. Compute PH (GUDHI + joblib)
      3. Extract all 15 descriptors
      4. Evaluate all 4 classifiers with 5-fold CV
      5. Save results
    """
    import torch

    if output_dir is None:
        output_dir = get_output_dir('exp1')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  BENCHMARK3 EXP1: {dataset_name}")
    print("=" * 70)
    print(f"  n_samples: {n_samples}")
    print(f"  n_jobs: {n_jobs}")
    print(f"  seed: {seed}")
    print(f"  n_folds: {EVALUATION['n_folds']}")
    print(f"  descriptors: {len(DESCRIPTORS)}")
    print(f"  classifiers: {ACTIVE_CLASSIFIERS}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"  output: {output_dir}")
    print("=" * 70)

    total_start = time.time()
    all_results = {}

    # --- Step 1: Load data ---
    print(f"\n[1/4] LOADING DATA: {dataset_name}")
    print("-" * 50)
    t0 = time.time()
    images, labels, class_names = load_dataset(dataset_name, n_samples=n_samples, seed=seed)
    load_time = time.time() - t0
    n_classes = len(class_names)
    actual_n = len(images)

    all_results['metadata'] = {
        'dataset': dataset_name,
        'n_samples': actual_n,
        'n_classes': n_classes,
        'class_names': class_names,
        'image_shape': list(images.shape),
        'seed': seed,
        'n_folds': EVALUATION['n_folds'],
        'load_time': load_time,
    }

    # --- Filter descriptors if specified ---
    if descriptor_filter:
        selected = [d.strip() for d in descriptor_filter.split(',')]
        run_descriptors = {k: v for k, v in DESCRIPTORS.items() if k in selected}
        print(f"\n  Running subset: {list(run_descriptors.keys())} ({len(run_descriptors)}/{len(DESCRIPTORS)})")
    else:
        run_descriptors = DESCRIPTORS

    # (merge with previous results happens at save time to support parallel runs)

    # --- Step 2: Compute PH (skip if only image-based descriptors) ---
    from RuleBenchmark.benchmark3.config import PH_BASED_DESCRIPTORS, IMAGE_BASED_DESCRIPTORS
    need_ph = any(d in PH_BASED_DESCRIPTORS for d in run_descriptors)

    if need_ph:
        print(f"\n[2/4] COMPUTING PH")
        print("-" * 50)
        diagrams, ph_time = compute_ph(images, n_jobs=n_jobs)

        n_h0 = np.mean([len(d.get('H0', [])) for d in diagrams])
        n_h1 = np.mean([len(d.get('H1', [])) for d in diagrams])
        print(f"  Avg bars: H0={n_h0:.1f}, H1={n_h1:.1f}")

        all_results['metadata']['ph_time'] = ph_time
        all_results['metadata']['avg_h0_bars'] = float(n_h0)
        all_results['metadata']['avg_h1_bars'] = float(n_h1)
    else:
        print(f"\n[2/4] SKIPPING PH (all selected descriptors are image-based)")
        diagrams = None
        ph_time = 0.0

    # --- Step 3: Extract descriptors ---
    print(f"\n[3/4] EXTRACTING DESCRIPTORS")
    print("-" * 50)
    print(f"{'Descriptor':<32} {'Dim':>5} {'Actual':>6} {'Time':>7} {'Status'}")
    print("-" * 70)

    features_dict = {}
    descriptor_timings = {}
    all_correct = True

    for desc_name, desc_config in run_descriptors.items():
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
        status = "OK" if (dim_ok and nan_ok) else "WARN"

        if not dim_ok:
            status = f"DIM({actual_dim}!={expected_dim})"
            all_correct = False
        elif not nan_ok:
            status = f"NaN:{nan_pct:.1f}%"
            all_correct = False

        print(f"{desc_name:<32} {expected_dim:>5} {actual_dim:>6} {elapsed:>6.1f}s {status}")

        # Clean features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_dict[desc_name] = features
        descriptor_timings[desc_name] = elapsed

        all_results[f'descriptor_{desc_name}'] = {
            'dim': actual_dim,
            'expected_dim': expected_dim,
            'extraction_time': elapsed,
            'nan_pct': float(nan_pct),
            'zero_pct': float(zero_pct),
            'errors': errors[:5] if errors else [],
        }

    total_extract_time = sum(descriptor_timings.values())
    print(f"\n  Total extraction: {total_extract_time:.1f}s")
    all_results['metadata']['total_extraction_time'] = total_extract_time

    # --- Save features immediately after extraction (prevent loss if killed) ---
    features_file = output_dir / f"{dataset_name}_features.npz"
    save_dict = {f"feat_{k}": v for k, v in features_dict.items()}
    if descriptor_filter and features_file.exists():
        existing = dict(np.load(features_file, allow_pickle=True))
        existing.update(save_dict)
        save_dict = existing
    save_dict['labels'] = labels
    save_dict['class_names'] = class_names
    np.savez_compressed(features_file, **save_dict)
    print(f"  Features saved to: {features_file}")

    # --- Step 4: Classification (with incremental saving) ---
    print(f"\n[4/4] CLASSIFICATION ({EVALUATION['n_folds']}-fold CV)")
    print("-" * 50)

    clf_header = "".join(f"{c:>14}" for c in ACTIVE_CLASSIFIERS)
    print(f"{'Descriptor':<28} {clf_header}")
    print("-" * (28 + 14 * len(ACTIVE_CLASSIFIERS)))

    result_file = output_dir / f"{dataset_name}_results.json"

    for desc_name, features in features_dict.items():
        clf_results = evaluate_all_classifiers(
            features, labels, dataset_name,
            seed=seed, n_folds=EVALUATION['n_folds'],
        )

        row = f"{desc_name:<28}"
        for clf_name in ACTIVE_CLASSIFIERS:
            acc = clf_results[clf_name]['balanced_accuracy_mean']
            std = clf_results[clf_name]['balanced_accuracy_std']
            row += f"  {acc:.3f}±{std:.2f}"
        print(row)

        all_results[f'classification_{desc_name}'] = clf_results

        # Incremental save: write after each descriptor's classification
        save_results = all_results.copy()
        if descriptor_filter and result_file.exists():
            with open(result_file, 'r') as f:
                previous = json.load(f)
            previous.update(save_results)
            save_results = previous
        with open(result_file, 'w') as f:
            json.dump(save_results, f, indent=2)

    # --- Final save ---
    total_time = time.time() - total_start
    all_results['metadata']['total_time'] = total_time
    all_results['metadata']['all_dimensions_correct'] = all_correct

    if descriptor_filter and result_file.exists():
        with open(result_file, 'r') as f:
            previous = json.load(f)
        previous.update(all_results)
        all_results = previous
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {result_file}")
    print(f"  Features saved to: {features_file}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  COMPLETE: {dataset_name}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Dimensions correct: {all_correct}")
    print(f"{'='*70}")

    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark3 Exp1: Main Benchmark")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (or comma-separated for combined jobs)')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Max samples per dataset')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs for PH (-1 = all cores)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/benchmark3/exp1)')
    parser.add_argument('--descriptors', type=str, default=None,
                        help='Comma-separated list of descriptors to run (default: all)')

    args = parser.parse_args()

    # Handle combined datasets (e.g., "RetinaMNIST,BreakHis")
    datasets = [d.strip() for d in args.dataset.split(',')]

    output_dir = Path(args.output_dir) if args.output_dir else get_output_dir('exp1')

    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"  DATASET: {dataset_name}")
        print(f"{'#'*70}\n")

        run_benchmark(
            dataset_name=dataset_name,
            n_samples=args.n_samples,
            n_jobs=args.n_jobs,
            seed=args.seed,
            output_dir=output_dir,
            descriptor_filter=args.descriptors,
        )

        gc.collect()

    print("\nAll datasets complete.")
