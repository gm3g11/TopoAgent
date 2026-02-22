#!/usr/bin/env python
"""
Experiment 4.2: Parameter Search

Find optimal parameters P* for each descriptor at its optimal dimension D*.
D* is loaded from Experiment 1 results.

Usage:
    python benchmarks/benchmark3/exp4_parameter_search.py --trial
    python benchmarks/benchmark3/exp4_parameter_search.py --descriptor persistence_image
    python benchmarks/benchmark3/exp4_parameter_search.py --full
    python benchmarks/benchmark3/exp4_parameter_search.py --dim-results results/exp4/exp4_fine_*.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import gc
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import ParameterGrid

from RuleBenchmark.benchmark3.exp4_config import (
    ALL_PARAMETERS, MEMORY_SAFE_DIMENSION_SEARCH,
    OBJECT_TYPE_DATASETS, TRIAL_DATASETS, RESULTS_PATH,
    LEARNED_DESCRIPTORS, PH_BASED_DESCRIPTORS, IMAGE_BASED_DESCRIPTORS,
)
from RuleBenchmark.benchmark3.exp4_dimension_study import (
    load_dataset, compute_ph_gudhi, extract_descriptor,
    evaluate_with_cv, evaluate_learned_descriptor_cv,
)
from RuleBenchmark.benchmark3.exp4_precompute_ph import load_cached_ph


# =============================================================================
# PARAMETER GRIDS (Non-dimension parameters only)
# =============================================================================

PARAM_SEARCH_GRIDS = {
    # Descriptors with additional parameters to tune (beyond dimension)
    'persistence_image': {
        'params': {
            'sigma': [0.05, 0.1, 0.15, 0.2, 0.25],
            'weight_function': ['linear', 'squared', 'const'],
        },
        'dimension_param': 'resolution',
    },
    'persistence_landscapes': {
        'params': {
            'n_bins': [50, 75, 100, 150],
            'combine_dims': [True, False],
        },
        'dimension_param': 'n_layers',
    },
    'persistence_silhouette': {
        'params': {
            'power': [0.5, 1.0, 1.5, 2.0],
        },
        'dimension_param': 'n_bins',
    },
    'betti_curves': {
        'params': {
            'normalize': [True, False],
        },
        'dimension_param': 'n_bins',
    },
    'persistence_entropy': {
        'params': {
            'mode': ['scalar', 'vector'],
            'normalized': [True, False],
        },
        'dimension_param': 'n_bins',
    },
    'template_functions': {
        'params': {
            'template_type': ['tent', 'gaussian'],
        },
        'dimension_param': 'n_templates',
    },
    'minkowski_functionals': {
        'params': {
            'adaptive': [True, False],
        },
        'dimension_param': 'n_thresholds',
    },
    'euler_characteristic_transform': {
        'params': {
            'n_heights': [10, 15, 20, 30, 40],
        },
        'dimension_param': 'n_directions',
    },
    'lbp_texture': {
        'params': {
            'method': ['uniform', 'default', 'ror'],
        },
        'dimension_param': 'scales',
    },
    'edge_histogram': {
        'params': {
            'n_orientation_bins': [4, 8, 12, 16],
        },
        'dimension_param': 'n_spatial_cells',
    },
    # Descriptors with only dimension parameter (no additional params to tune)
    # These are skipped in parameter search
    'persistence_codebook': None,  # Only codebook_size (dimension)
    'tropical_coordinates': None,  # Only max_terms (dimension)
    'ATOL': None,                  # Only n_centers (dimension)
    'euler_characteristic_curve': None,  # Only resolution (dimension)
    'persistence_statistics': None,      # Only subset (dimension)
}


# =============================================================================
# LOAD DIMENSION RESULTS
# =============================================================================

def load_dimension_results(results_path: str = None) -> Dict:
    """
    Load optimal dimensions D* from Experiment 1 results.

    Returns: {dataset: {descriptor: {'best_value': X, 'best_dim': Y, 'classifier': ...}}}
    """
    if results_path is None:
        # First try the consolidated optimal dimensions file
        consolidated_path = RESULTS_PATH / "exp4_optimal_dimensions.json"
        if consolidated_path.exists():
            results_path = str(consolidated_path)
        else:
            # Find latest dimension results
            pattern = str(RESULTS_PATH / "exp4_fine_*.json")
            files = glob.glob(pattern)
            if not files:
                print("  No dimension results found, using default medium dimensions")
                return None
            results_path = max(files, key=lambda f: Path(f).stat().st_mtime)

    print(f"  Loading dimension results from: {results_path}")

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Parse results into {dataset: {descriptor: {best_value, best_dim, classifier}}}
    dim_results = {}

    # Check if this is the new consolidated format
    if 'metadata' in data:
        # New consolidated format: {dataset: {descriptor: {best_value, best_dim, ...}}}
        for dataset, ds_data in data.items():
            if dataset == 'metadata':
                continue
            if isinstance(ds_data, dict):
                dim_results[dataset] = {}
                for descriptor, desc_data in ds_data.items():
                    if isinstance(desc_data, dict) and 'best_value' in desc_data:
                        dim_results[dataset][descriptor] = {
                            'best_value': desc_data['best_value'],
                            'best_dim': desc_data['best_dim'],
                            'classifier': desc_data.get('classifier', 'TabPFN'),
                        }
    else:
        # Old format: {dataset: {descriptor: ..., best_value: ...}}
        for dataset, ds_data in data.items():
            if isinstance(ds_data, dict) and 'best_value' in ds_data:
                # Single descriptor result
                descriptor = ds_data.get('descriptor', 'unknown')
                if dataset not in dim_results:
                    dim_results[dataset] = {}
                dim_results[dataset][descriptor] = {
                    'best_value': ds_data['best_value'],
                    'best_dim': ds_data['best_dim'],
                    'classifier': ds_data.get('classifier', 'TabPFN'),
                }

    return dim_results if dim_results else None


def get_optimal_dimension(descriptor: str, dataset: str, dim_results: Dict = None) -> Tuple[Any, int, str]:
    """Get optimal dimension and classifier for a descriptor on a dataset.

    Returns: (dimension_value, dimension_size, classifier_name)
    """

    if dim_results and dataset in dim_results and descriptor in dim_results[dataset]:
        info = dim_results[dataset][descriptor]
        classifier = info.get('classifier', 'TabPFN')
        return info['best_value'], info['best_dim'], classifier

    # Fallback to medium dimension
    config = MEMORY_SAFE_DIMENSION_SEARCH.get(descriptor, {})
    if config:
        values = config['values']
        dimensions = config['dimensions']
        mid_idx = len(values) // 2
        # Default: use XGBoost for persistence_landscapes (high dim), TabPFN otherwise
        classifier = 'XGBoost' if descriptor == 'persistence_landscapes' else 'TabPFN'
        return values[mid_idx], dimensions[mid_idx], classifier

    return None, 100, 'TabPFN'


# =============================================================================
# PARAMETER SEARCH
# =============================================================================

def run_parameter_search(
    descriptor: str,
    dataset: str,
    images: np.ndarray,
    diagrams: List[Dict],
    labels: np.ndarray,
    dim_value: Any,
    dim_size: int,
    n_folds: int = 5,
    seed: int = 42,
    classifier_name: str = 'TabPFN',
) -> Dict:
    """
    Search for optimal parameters at fixed dimension D*.

    Args:
        descriptor: Descriptor name
        dataset: Dataset name
        images: Image array
        diagrams: PH diagrams
        labels: Labels
        dim_value: Optimal dimension parameter value (from Exp 1)
        dim_size: Actual dimension size
        n_folds: CV folds
        seed: Random seed
        classifier_name: 'TabPFN' or 'XGBoost'

    Returns:
        Dict with best parameters and all results
    """

    grid_config = PARAM_SEARCH_GRIDS.get(descriptor)

    if grid_config is None:
        print(f"    {descriptor}: No additional parameters to tune (dimension-only)")
        return {
            'descriptor': descriptor,
            'dataset': dataset,
            'status': 'dimension_only',
            'best_params': {MEMORY_SAFE_DIMENSION_SEARCH[descriptor]['control_param']: dim_value},
            'best_dim': dim_size,
            'classifier': classifier_name,
        }

    param_grid = grid_config['params']
    dim_param = grid_config['dimension_param']

    # Generate all parameter combinations
    all_configs = list(ParameterGrid(param_grid))

    print(f"\n    {descriptor}: Testing {len(all_configs)} parameter configs at {dim_param}={dim_value} ({dim_size}D) [{classifier_name}]")

    results = []
    best_acc = -1
    best_config = None

    for config_idx, params in enumerate(all_configs):
        # Add dimension parameter
        full_params = params.copy()
        full_params[dim_param] = dim_value

        print(f"      [{config_idx+1}/{len(all_configs)}] {params}...", end=' ')
        start = time.time()

        try:
            if descriptor in LEARNED_DESCRIPTORS:
                cv_result = evaluate_learned_descriptor_cv(
                    descriptor, diagrams, labels, full_params, dim_size,
                    n_folds=n_folds, seed=seed, dataset_name=dataset,
                    classifier_name=classifier_name,
                )
            else:
                features = extract_descriptor(
                    descriptor, images, diagrams, full_params, dim_size,
                )
                cv_result = evaluate_with_cv(
                    features, labels, n_folds=n_folds, seed=seed,
                    dataset_name=dataset, classifier_name=classifier_name,
                )

            # Check for OOM
            if cv_result.get('oom', False):
                print(f"OOM")
                results.append({
                    'params': params,
                    'accuracy': -1.0,
                    'oom': True,
                })
                continue

            acc = cv_result['mean']
            elapsed = time.time() - start
            print(f"{acc:.3f} ({elapsed:.1f}s)")

            results.append({
                'params': params,
                'accuracy': acc,
                'std': cv_result['std'],
                'time': elapsed,
            })

            if acc > best_acc:
                best_acc = acc
                best_config = params.copy()

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'params': params,
                'accuracy': 0.0,
                'error': str(e),
            })

    # Combine best params with dimension
    if best_config:
        best_config[dim_param] = dim_value

    return {
        'descriptor': descriptor,
        'dataset': dataset,
        'dimension_param': dim_param,
        'dimension_value': dim_value,
        'dimension_size': dim_size,
        'classifier': classifier_name,
        'best_params': best_config,
        'best_accuracy': best_acc,
        'all_results': results,
        'n_configs_tested': len(all_configs),
    }


def run_full_parameter_search(
    datasets: List[str],
    descriptors: List[str] = None,
    dim_results: Dict = None,
    n_samples: int = 2000,
    n_folds: int = 5,
    seed: int = 42,
    n_jobs: int = 4,
    use_cache: bool = True,
    classifier_override: str = None,
) -> Dict:
    """Run parameter search for all descriptors on all datasets.

    Args:
        classifier_override: If specified, use this classifier for all descriptors.
                           Otherwise, use the classifier from dimension results.
    """

    if descriptors is None:
        # Only descriptors with parameters to tune
        descriptors = [d for d, v in PARAM_SEARCH_GRIDS.items() if v is not None]

    all_results = {
        'n_samples': n_samples,
        'n_folds': n_folds,
        'seed': seed,
        'results': {},
    }

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*60}")

        # Check if any descriptors need images (image-based descriptors)
        needs_images = any(d in IMAGE_BASED_DESCRIPTORS for d in descriptors)

        # Load data
        if use_cache:
            try:
                cache_data = load_cached_ph(dataset, n_samples)
                # Support both old and new cache formats
                if 'diagrams_sublevel' in cache_data:
                    # New combined format - use sublevel for parameter search
                    diagrams = cache_data['diagrams_sublevel']
                    print(f"  Loaded PH from cache ({len(diagrams)} samples, combined format)")
                else:
                    # Old format - single filtration
                    diagrams = cache_data['diagrams']
                    print(f"  Loaded PH from cache ({len(diagrams)} samples, old format)")
                labels = cache_data['labels']

                # Load images separately if needed for image-based descriptors
                if needs_images:
                    images, _ = load_dataset(dataset, n_samples=n_samples, seed=seed)
                    print(f"  Also loaded images for image-based descriptors")
                else:
                    images = None
            except FileNotFoundError:
                print(f"  Cache not found, loading fresh...")
                images, labels = load_dataset(dataset, n_samples=n_samples, seed=seed)
                diagrams = compute_ph_gudhi(images, n_jobs=n_jobs)
        else:
            images, labels = load_dataset(dataset, n_samples=n_samples, seed=seed)
            diagrams = compute_ph_gudhi(images, n_jobs=n_jobs)

        all_results['results'][dataset] = {}

        for descriptor in descriptors:
            # Get optimal dimension and classifier from Exp 1
            dim_value, dim_size, auto_classifier = get_optimal_dimension(descriptor, dataset, dim_results)

            # Use override if specified, otherwise use auto-detected classifier
            classifier_name = classifier_override if classifier_override else auto_classifier

            # Run parameter search
            result = run_parameter_search(
                descriptor=descriptor,
                dataset=dataset,
                images=images,
                diagrams=diagrams,
                labels=labels,
                dim_value=dim_value,
                dim_size=dim_size,
                n_folds=n_folds,
                seed=seed,
                classifier_name=classifier_name,
            )

            all_results['results'][dataset][descriptor] = result

            # Print best
            if result.get('best_params'):
                print(f"    BEST: {result['best_params']} = {result.get('best_accuracy', 0):.3f}")

        # Memory cleanup
        del diagrams
        if images is not None:
            del images
        gc.collect()

    return all_results


def analyze_parameter_results(results: Dict) -> Dict:
    """Analyze parameter search results to derive rules."""

    analysis = {
        'per_descriptor': {},
        'rules': {},
    }

    # Aggregate by descriptor
    for dataset, ds_results in results.get('results', {}).items():
        for descriptor, desc_result in ds_results.items():
            if descriptor not in analysis['per_descriptor']:
                analysis['per_descriptor'][descriptor] = []

            if desc_result.get('best_params'):
                analysis['per_descriptor'][descriptor].append({
                    'dataset': dataset,
                    'best_params': desc_result['best_params'],
                    'best_accuracy': desc_result.get('best_accuracy', 0),
                })

    # Find most common best params per descriptor
    for descriptor, results_list in analysis['per_descriptor'].items():
        if not results_list:
            continue

        # Count parameter combinations
        param_counts = {}
        param_accs = {}

        for r in results_list:
            params = r['best_params']
            # Convert to hashable
            param_key = tuple(sorted((k, str(v)) for k, v in params.items()))

            if param_key not in param_counts:
                param_counts[param_key] = 0
                param_accs[param_key] = []

            param_counts[param_key] += 1
            param_accs[param_key].append(r['best_accuracy'])

        # Find most common / best performing
        best_key = max(param_counts.keys(), key=lambda k: (param_counts[k], np.mean(param_accs[k])))
        best_params = dict(best_key)

        analysis['rules'][descriptor] = {
            'recommended_params': best_params,
            'frequency': param_counts[best_key],
            'mean_accuracy': np.mean(param_accs[best_key]),
            'n_datasets': len(results_list),
        }

    return analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 4.2: Parameter Search")
    parser.add_argument('--trial', action='store_true', help='Quick trial')
    parser.add_argument('--full', action='store_true', help='Full experiment')
    parser.add_argument('--descriptor', type=str, default=None,
                       help='Specific descriptor')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset')
    parser.add_argument('--dim-results', type=str, default=None,
                       help='Path to dimension results from Exp 1')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cache', action='store_true',
                       help='Use precomputed PH cache')
    parser.add_argument('--classifier', type=str, default=None,
                       choices=['TabPFN', 'XGBoost'],
                       help='Override classifier (default: auto based on dimension)')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    print("=" * 70)
    print("  EXPERIMENT 4.2: PARAMETER SEARCH")
    print("=" * 70)
    print(f"  n_samples: {args.n_samples}")
    print(f"  n_folds: {args.n_folds}")
    print(f"  classifier: {args.classifier if args.classifier else 'auto (from Exp1 results)'}")
    print(f"  use_cache: {args.use_cache}")
    print("=" * 70)

    # Load dimension results from Exp 1
    dim_results = load_dimension_results(args.dim_results)

    # Determine descriptors
    if args.descriptor:
        descriptors = [args.descriptor]
    else:
        # Only those with parameters to tune
        descriptors = [d for d, v in PARAM_SEARCH_GRIDS.items() if v is not None]

    print(f"  Descriptors with params to tune: {len(descriptors)}")
    for d in descriptors:
        grid = PARAM_SEARCH_GRIDS.get(d, {})
        if grid:
            n_configs = 1
            for vals in grid['params'].values():
                n_configs *= len(vals)
            print(f"    {d}: {n_configs} configs")

    # Determine datasets
    if args.dataset:
        datasets = [args.dataset]
    elif args.trial:
        datasets = []
        for ds_list in TRIAL_DATASETS.values():
            datasets.extend(ds_list)
    else:
        datasets = []
        for ds_list in OBJECT_TYPE_DATASETS.values():
            datasets.extend(ds_list)

    # Run parameter search
    results = run_full_parameter_search(
        datasets=datasets,
        descriptors=descriptors,
        dim_results=dim_results,
        n_samples=args.n_samples,
        n_folds=args.n_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        use_cache=args.use_cache,
        classifier_override=args.classifier,
    )

    # Analyze
    analysis = analyze_parameter_results(results)
    results['analysis'] = analysis

    # Print rules
    print(f"\n\n{'='*70}")
    print("  PARAMETER RULES (Recommended)")
    print(f"{'='*70}")
    for descriptor, rule in analysis['rules'].items():
        print(f"\n  {descriptor}:")
        print(f"    Params: {rule['recommended_params']}")
        print(f"    Accuracy: {rule['mean_accuracy']:.3f} (from {rule['n_datasets']} datasets)")

    # Save results - use descriptor+dataset-specific file to avoid overwriting
    if args.output:
        output_path = Path(args.output)
    elif args.descriptor and args.dataset:
        # Single descriptor + single dataset - include both to avoid overwrite
        output_path = RESULTS_PATH / f"exp4_param_{args.descriptor}_{args.dataset}_n{args.n_samples}.json"
    elif args.descriptor:
        # Single descriptor mode - save to descriptor-specific file
        output_path = RESULTS_PATH / f"exp4_param_{args.descriptor}_n{args.n_samples}.json"
    else:
        output_path = RESULTS_PATH / f"exp4_parameter_search_n{args.n_samples}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file before overwriting
    if output_path.exists():
        import shutil
        backup_path = output_path.with_suffix(f'.backup_{int(time.time())}.json')
        shutil.copy2(output_path, backup_path)
        print(f"\n  Backed up existing file to: {backup_path}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
