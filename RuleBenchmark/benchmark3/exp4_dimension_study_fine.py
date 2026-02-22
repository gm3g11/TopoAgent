#!/usr/bin/env python
"""
Experiment 4.1: Fine-Grained Dimension Search with Binary Search Optimization

Uses memory-safe dimension ranges and optional golden section search
for unimodal descriptors.

Usage:
    python benchmarks/benchmark3/exp4_dimension_study_fine.py --descriptor ATOL --dataset BloodMNIST
    python benchmarks/benchmark3/exp4_dimension_study_fine.py --descriptor persistence_image --use-cache
    python benchmarks/benchmark3/exp4_dimension_study_fine.py --descriptor ATOL --binary-search
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import pickle
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
    MEMORY_SAFE_DIMENSION_SEARCH, ALL_PARAMETERS,
    BINARY_SEARCH_DESCRIPTORS, FULL_SEARCH_DESCRIPTORS,
    OBJECT_TYPE_DATASETS, TRIAL_DATASETS,
    ALL_DESCRIPTORS, PH_BASED_DESCRIPTORS, IMAGE_BASED_DESCRIPTORS,
    LEARNED_DESCRIPTORS, RESULTS_PATH,
)
from RuleBenchmark.benchmark3.exp4_dimension_study import (
    load_dataset, compute_ph_gudhi, extract_descriptor,
    evaluate_with_cv, evaluate_learned_descriptor_cv,
)
from RuleBenchmark.benchmark3.exp4_precompute_ph import (
    PH_CACHE_PATH, get_cache_path, load_cached_ph,
)
from RuleBenchmark.benchmark3.config import get_classifier


def golden_section_search(
    descriptor: str,
    dataset: str,
    images: np.ndarray,
    diagrams: List[Dict],
    labels: np.ndarray,
    values: List,
    dimensions: List[int],
    fixed_params: Dict,
    n_folds: int = 5,
    seed: int = 42,
    tol: int = 2,
    classifier_name: str = 'TabPFN',
) -> Tuple[Any, int, float, Dict]:
    """
    Find optimal dimension using golden section search.
    Stops at OOM and records the dimension limit.

    Returns:
        (best_value, best_dim, best_accuracy, full_results)
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    results = {}
    oom_at_dim = None

    def evaluate_at_index(idx: int) -> float:
        """Evaluate accuracy at given index. Returns -1 for OOM."""
        nonlocal oom_at_dim

        if idx in results:
            return results[idx]['accuracy']

        value = values[idx]
        dim = dimensions[idx]

        # Build params
        params = fixed_params.copy()
        control_param = MEMORY_SAFE_DIMENSION_SEARCH[descriptor]['control_param']
        params[control_param] = value

        print(f"    Evaluating {control_param}={value} ({dim}D)...", end=' ')
        start = time.time()

        try:
            if descriptor in LEARNED_DESCRIPTORS:
                cv_result = evaluate_learned_descriptor_cv(
                    descriptor, diagrams, labels, params, dim,
                    n_folds=n_folds, seed=seed, dataset_name=dataset,
                    classifier_name=classifier_name,
                )
            else:
                features = extract_descriptor(
                    descriptor, images, diagrams, params, dim,
                )
                cv_result = evaluate_with_cv(
                    features, labels, n_folds=n_folds, seed=seed,
                    dataset_name=dataset, classifier_name=classifier_name,
                )

            # Check for OOM marker
            if cv_result.get('oom', False):
                print(f"OOM at {dim}D")
                oom_at_dim = dim
                results[idx] = {
                    'value': value,
                    'dim': dim,
                    'accuracy': -1.0,
                    'oom': True,
                }
                return -1.0

            acc = cv_result['mean']
            elapsed = time.time() - start
            print(f"{acc:.3f} ({elapsed:.1f}s)")

            results[idx] = {
                'value': value,
                'dim': dim,
                'accuracy': acc,
                'std': cv_result['std'],
                'time': elapsed,
            }
            return acc

        except Exception as e:
            error_str = str(e)
            if 'CUDA' in error_str or 'out of memory' in error_str.lower():
                print(f"OOM at {dim}D")
                oom_at_dim = dim
                results[idx] = {
                    'value': value,
                    'dim': dim,
                    'accuracy': -1.0,
                    'oom': True,
                    'error': error_str,
                }
                return -1.0
            else:
                print(f"ERROR: {e}")
                results[idx] = {
                    'value': value,
                    'dim': dim,
                    'accuracy': 0.0,
                    'std': 0.0,
                    'error': error_str,
                }
                return 0.0

    # Golden section search
    a, b = 0, len(values) - 1
    c = int(b - (b - a) / phi)
    d = int(a + (b - a) / phi)

    while abs(b - a) > tol:
        acc_c = evaluate_at_index(c)
        acc_d = evaluate_at_index(d)

        # If we hit OOM, stop search and focus on lower dimensions
        if acc_c < 0 or acc_d < 0:
            if acc_c < 0:
                b = c  # OOM at c, search lower
            if acc_d < 0:
                b = d  # OOM at d, search lower
            print(f"    OOM hit - constraining search to lower dimensions")
            break

        if acc_c > acc_d:
            b = d
        else:
            a = c

        c = int(b - (b - a) / phi)
        d = int(a + (b - a) / phi)

    # Fine search around peak (only valid range)
    peak_indices = list(range(max(0, a - 1), min(len(values), b + 2)))
    for idx in peak_indices:
        if oom_at_dim and dimensions[idx] >= oom_at_dim:
            continue  # Skip dimensions at or above OOM limit
        evaluate_at_index(idx)

    # Find best (only among successful evaluations)
    valid_results = {k: v for k, v in results.items() if isinstance(k, int) and v.get('accuracy', -1) >= 0}
    if valid_results:
        best_idx = max(valid_results.keys(), key=lambda i: valid_results[i]['accuracy'])
        best = results[best_idx]
    else:
        best = {'value': values[0], 'dim': dimensions[0], 'accuracy': 0.0}

    # Add OOM info to results
    if oom_at_dim:
        results['oom_at_dim'] = oom_at_dim

    return best['value'], best['dim'], best.get('accuracy', 0.0), results


def full_search(
    descriptor: str,
    dataset: str,
    images: np.ndarray,
    diagrams: List[Dict],
    labels: np.ndarray,
    values: List,
    dimensions: List[int],
    fixed_params: Dict,
    n_folds: int = 5,
    seed: int = 42,
    classifier_name: str = 'TabPFN',
) -> Tuple[Any, int, float, Dict]:
    """
    Full linear search through all dimension values.
    Stops at OOM and records the dimension limit.

    Returns:
        (best_value, best_dim, best_accuracy, full_results)
    """
    results = {}
    best_acc = -1
    best_idx = 0
    oom_at_dim = None

    control_param = MEMORY_SAFE_DIMENSION_SEARCH[descriptor]['control_param']

    for idx, (value, dim) in enumerate(zip(values, dimensions)):
        # Build params
        params = fixed_params.copy()
        params[control_param] = value

        print(f"    {control_param}={value} ({dim}D)...", end=' ')
        start = time.time()

        try:
            if descriptor in LEARNED_DESCRIPTORS:
                cv_result = evaluate_learned_descriptor_cv(
                    descriptor, diagrams, labels, params, dim,
                    n_folds=n_folds, seed=seed, dataset_name=dataset,
                    classifier_name=classifier_name,
                )
            else:
                features = extract_descriptor(
                    descriptor, images, diagrams, params, dim,
                )
                cv_result = evaluate_with_cv(
                    features, labels, n_folds=n_folds, seed=seed,
                    dataset_name=dataset, classifier_name=classifier_name,
                )

            # Check for OOM marker
            if cv_result.get('oom', False):
                print(f"OOM at {dim}D - stopping search")
                oom_at_dim = dim
                results[idx] = {
                    'value': value,
                    'dim': dim,
                    'accuracy': -1.0,
                    'oom': True,
                }
                break

            acc = cv_result['mean']
            elapsed = time.time() - start
            print(f"{acc:.3f} ({elapsed:.1f}s)")

            results[idx] = {
                'value': value,
                'dim': dim,
                'accuracy': acc,
                'std': cv_result['std'],
                'time': elapsed,
            }

            # Track best
            if acc > best_acc:
                best_acc = acc
                best_idx = idx

        except Exception as e:
            error_str = str(e)
            if 'CUDA' in error_str or 'out of memory' in error_str.lower():
                print(f"OOM at {dim}D - stopping search")
                oom_at_dim = dim
                results[idx] = {
                    'value': value,
                    'dim': dim,
                    'accuracy': -1.0,
                    'oom': True,
                    'error': error_str,
                }
                break
            else:
                print(f"ERROR: {e}")
                results[idx] = {
                    'value': value,
                    'dim': dim,
                    'accuracy': 0.0,
                    'std': 0.0,
                    'error': error_str,
                }

    # Find best (only among successful evaluations)
    valid_results = {k: v for k, v in results.items() if v.get('accuracy', -1) >= 0}
    if valid_results:
        best_idx = max(valid_results.keys(), key=lambda i: valid_results[i]['accuracy'])
        best = results[best_idx]
    else:
        best = {'value': values[0], 'dim': dimensions[0], 'accuracy': 0.0}

    # Add OOM info to results
    if oom_at_dim:
        results['oom_at_dim'] = oom_at_dim

    return best['value'], best['dim'], best.get('accuracy', 0.0), results


def run_fine_dimension_search(
    descriptor: str,
    dataset: str,
    images: np.ndarray,
    diagrams: List[Dict],
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    use_binary_search: bool = None,
    classifier_name: str = 'TabPFN',
) -> Dict:
    """Run fine-grained dimension search for one descriptor on one dataset."""

    config = MEMORY_SAFE_DIMENSION_SEARCH.get(descriptor)
    if config is None:
        raise ValueError(f"Unknown descriptor: {descriptor}")

    values = config['values']
    dimensions = config['dimensions']
    fixed_params = config.get('fixed', {})

    print(f"\n  Descriptor: {descriptor}")
    print(f"  Memory tier: {config['memory_tier']}")
    print(f"  Search space: {len(values)} points ({min(dimensions)}D - {max(dimensions)}D)")
    print(f"  Fixed params: {fixed_params}")

    # Determine search strategy
    if use_binary_search is None:
        use_binary_search = descriptor in BINARY_SEARCH_DESCRIPTORS

    if use_binary_search and len(values) > 5:
        print(f"  Strategy: Golden section search")
        best_value, best_dim, best_acc, results = golden_section_search(
            descriptor, dataset, images, diagrams, labels,
            values, dimensions, fixed_params,
            n_folds=n_folds, seed=seed, classifier_name=classifier_name,
        )
    else:
        print(f"  Strategy: Full search")
        best_value, best_dim, best_acc, results = full_search(
            descriptor, dataset, images, diagrams, labels,
            values, dimensions, fixed_params,
            n_folds=n_folds, seed=seed, classifier_name=classifier_name,
        )

    control_param = config['control_param']
    print(f"\n  BEST: {control_param}={best_value} ({best_dim}D) = {best_acc:.3f}")

    return {
        'descriptor': descriptor,
        'dataset': dataset,
        'control_param': control_param,
        'best_value': best_value,
        'best_dim': best_dim,
        'best_accuracy': best_acc,
        'fixed_params': fixed_params,
        'search_method': 'golden_section' if use_binary_search else 'full',
        'n_evaluations': len(results),
        'all_results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-grained dimension search")
    parser.add_argument('--descriptor', type=str, required=True,
                       help='Descriptor to evaluate')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Single dataset (default: all trial datasets)')
    parser.add_argument('--object-type', type=str, default=None,
                       help='Object type to evaluate')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cache', action='store_true',
                       help='Use precomputed PH cache')
    parser.add_argument('--binary-search', action='store_true',
                       help='Force binary search')
    parser.add_argument('--no-binary-search', action='store_true',
                       help='Force full search')
    parser.add_argument('--classifier', type=str, default='TabPFN',
                       choices=['TabPFN', 'XGBoost', 'RandomForest'],
                       help='Classifier to use (default: TabPFN)')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    # Validate descriptor
    if args.descriptor not in MEMORY_SAFE_DIMENSION_SEARCH:
        print(f"ERROR: Unknown descriptor '{args.descriptor}'")
        print(f"Available: {list(MEMORY_SAFE_DIMENSION_SEARCH.keys())}")
        sys.exit(1)

    print("=" * 70)
    print(f"  FINE-GRAINED DIMENSION SEARCH: {args.descriptor}")
    print("=" * 70)
    print(f"  n_samples: {args.n_samples}")
    print(f"  n_folds: {args.n_folds}")
    print(f"  classifier: {args.classifier}")
    print(f"  use_cache: {args.use_cache}")
    print("=" * 70)

    # Determine datasets
    if args.dataset:
        datasets = [args.dataset]
    elif args.object_type:
        datasets = OBJECT_TYPE_DATASETS.get(args.object_type, [])
    else:
        # All trial datasets
        datasets = []
        for ds_list in TRIAL_DATASETS.values():
            datasets.extend(ds_list)

    # Determine search strategy
    use_binary = None
    if args.binary_search:
        use_binary = True
    elif args.no_binary_search:
        use_binary = False

    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*60}")

        # Load data
        if args.use_cache and args.descriptor in PH_BASED_DESCRIPTORS:
            # Try to load from cache (new combined format)
            try:
                cache_data = load_cached_ph(dataset, args.n_samples)
                # Support both old and new cache formats
                if 'diagrams_sublevel' in cache_data:
                    # New combined format - use sublevel for dimension search
                    # (combined feature extraction doubles dimension at extraction time)
                    diagrams = cache_data['diagrams_sublevel']
                    diagrams_superlevel = cache_data.get('diagrams_superlevel')
                    print(f"  Loaded PH from cache ({len(diagrams)} samples, combined format)")
                else:
                    # Old format - single filtration
                    diagrams = cache_data['diagrams']
                    diagrams_superlevel = None
                    print(f"  Loaded PH from cache ({len(diagrams)} samples, old format)")
                labels = cache_data['labels']
                images = None  # Not needed for PH-based descriptors
            except FileNotFoundError:
                print(f"  Cache not found, computing PH...")
                images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
                diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)
                diagrams_superlevel = None
        else:
            images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
            if args.descriptor in PH_BASED_DESCRIPTORS:
                diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)
                diagrams_superlevel = None
            else:
                diagrams = None
                diagrams_superlevel = None

        # Run dimension search
        result = run_fine_dimension_search(
            descriptor=args.descriptor,
            dataset=dataset,
            images=images,
            diagrams=diagrams,
            labels=labels,
            n_folds=args.n_folds,
            seed=args.seed,
            use_binary_search=use_binary,
            classifier_name=args.classifier,
        )

        all_results[dataset] = result

        # Memory cleanup
        del images, diagrams
        gc.collect()

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: {args.descriptor}")
    print(f"{'='*70}")

    for dataset, result in all_results.items():
        print(f"  {dataset}: {result['control_param']}={result['best_value']} "
              f"({result['best_dim']}D) = {result['best_accuracy']:.3f}")

    # Save results
    # When running a single dataset, include dataset name to avoid overwriting
    # multi-dataset results from earlier runs
    if args.output:
        output_path = Path(args.output)
    elif args.dataset:
        output_path = RESULTS_PATH / f"exp4_fine_{args.descriptor}_{args.dataset}_n{args.n_samples}.json"
    else:
        output_path = RESULTS_PATH / f"exp4_fine_{args.descriptor}_n{args.n_samples}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file before overwriting
    if output_path.exists():
        import shutil
        backup_path = output_path.with_suffix(f'.backup_{int(time.time())}.json')
        shutil.copy2(output_path, backup_path)
        print(f"\n  Backed up existing file to: {backup_path}")

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
