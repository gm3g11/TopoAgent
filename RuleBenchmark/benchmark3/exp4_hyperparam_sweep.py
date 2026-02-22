#!/usr/bin/env python
"""
Experiment 4.2: Hyperparameter Sensitivity Study

Grid search for hyperparameters of top descriptors per object type.

Usage:
    python benchmarks/benchmark3/exp4_hyperparam_sweep.py --trial
    python benchmarks/benchmark3/exp4_hyperparam_sweep.py --descriptor persistence_image --object-type discrete_cells
    python benchmarks/benchmark3/exp4_hyperparam_sweep.py --full
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import itertools
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from RuleBenchmark.benchmark3.exp4_config import (
    PARAM_GRIDS, OBJECT_TYPE_DATASETS, TRIAL_DATASETS,
    ALL_DESCRIPTORS, LEARNED_DESCRIPTORS, RESULTS_PATH,
    PH_BASED_DESCRIPTORS, IMAGE_BASED_DESCRIPTORS,
)
from RuleBenchmark.benchmark3.exp4_dimension_study import (
    load_dataset, compute_ph_gudhi, extract_descriptor,
    evaluate_with_cv, evaluate_learned_descriptor_cv,
    _map_params_to_tool,
)
from RuleBenchmark.benchmark3.config import get_classifier


# =============================================================================
# HYPERPARAMETER SWEEP
# =============================================================================

def run_hyperparam_sweep(
    descriptor: str,
    object_type: str,
    datasets: List[str],
    n_samples: int = 50,
    n_folds: int = 3,
    seed: int = 42,
    n_jobs: int = 4,
    max_configs: int = None,
) -> Dict:
    """Run hyperparameter sweep for one descriptor on datasets of one object type."""

    param_grid = PARAM_GRIDS.get(descriptor, {})
    if not param_grid:
        print(f"  No parameter grid defined for {descriptor}")
        return {}

    # Generate all parameter combinations
    all_configs = list(ParameterGrid(param_grid))
    if max_configs and len(all_configs) > max_configs:
        print(f"  Limiting configs from {len(all_configs)} to {max_configs}")
        np.random.seed(seed)
        all_configs = list(np.random.choice(all_configs, max_configs, replace=False))

    print(f"\n{'='*60}")
    print(f"  Descriptor: {descriptor}")
    print(f"  Object Type: {object_type}")
    print(f"  Datasets: {datasets}")
    print(f"  Param configs: {len(all_configs)}")
    print(f"{'='*60}")

    results = {
        'descriptor': descriptor,
        'object_type': object_type,
        'datasets': datasets,
        'n_configs': len(all_configs),
        'param_grid': param_grid,
        'results': [],
    }

    # Load all datasets once
    dataset_data = {}
    for dataset in datasets:
        print(f"\n  Loading {dataset}...")
        images, labels = load_dataset(dataset, n_samples=n_samples, seed=seed)
        diagrams = None
        if descriptor in PH_BASED_DESCRIPTORS:
            diagrams = compute_ph_gudhi(images, n_jobs=n_jobs)
        dataset_data[dataset] = {
            'images': images,
            'labels': labels,
            'diagrams': diagrams,
        }

    # Run grid search
    print(f"\n  Running {len(all_configs)} configurations...")
    for config_idx, params in enumerate(all_configs):
        config_results = {
            'params': params,
            'dataset_results': {},
        }

        print(f"\n  Config {config_idx+1}/{len(all_configs)}: {params}")

        for dataset in datasets:
            data = dataset_data[dataset]
            images = data['images']
            labels = data['labels']
            diagrams = data['diagrams']

            # Estimate expected dimension from params
            expected_dim = _estimate_dimension(descriptor, params)

            start = time.time()

            try:
                if descriptor in LEARNED_DESCRIPTORS:
                    cv_result = evaluate_learned_descriptor_cv(
                        descriptor, diagrams, labels, params, expected_dim,
                        n_folds=n_folds, seed=seed, dataset_name=dataset,
                    )
                else:
                    features = extract_descriptor(
                        descriptor, images, diagrams, params, expected_dim,
                    )
                    cv_result = evaluate_with_cv(
                        features, labels, n_folds=n_folds, seed=seed,
                        dataset_name=dataset,
                    )

                elapsed = time.time() - start
                acc_mean = cv_result['mean']
                acc_std = cv_result['std']

                config_results['dataset_results'][dataset] = {
                    'accuracy': acc_mean,
                    'std': acc_std,
                    'time': elapsed,
                }

                print(f"    {dataset}: {acc_mean:.3f}±{acc_std:.2f}")

            except Exception as e:
                print(f"    {dataset}: ERROR - {e}")
                config_results['dataset_results'][dataset] = {
                    'accuracy': 0.0,
                    'std': 0.0,
                    'error': str(e),
                }

        # Compute mean across datasets
        accs = [r['accuracy'] for r in config_results['dataset_results'].values()]
        config_results['mean_accuracy'] = np.mean(accs)
        config_results['std_accuracy'] = np.std(accs)

        results['results'].append(config_results)

    return results


def _estimate_dimension(descriptor: str, params: Dict) -> int:
    """Estimate output dimension from parameters."""

    if descriptor == 'persistence_image':
        res = params.get('resolution', 20)
        return res * res * 2

    elif descriptor == 'persistence_landscapes':
        n_layers = params.get('n_layers', 4)
        n_bins = params.get('n_bins', 100)
        return n_layers * n_bins  # combine_dims=True

    elif descriptor in ['persistence_silhouette', 'betti_curves', 'persistence_entropy']:
        n_bins = params.get('n_bins', 100)
        return n_bins * 2

    elif descriptor == 'persistence_codebook':
        codebook_size = params.get('codebook_size', 32)
        return codebook_size * 2

    elif descriptor == 'ATOL':
        n_centers = params.get('n_centers', 16)
        return n_centers * 2

    elif descriptor == 'tropical_coordinates':
        max_terms = params.get('max_terms', 5)
        return max_terms * 4 * 2

    elif descriptor == 'template_functions':
        n_templates = params.get('n_templates', 25)
        return n_templates * 2

    elif descriptor == 'minkowski_functionals':
        n_thresholds = params.get('n_thresholds', 10)
        return n_thresholds * 3

    elif descriptor == 'euler_characteristic_curve':
        resolution = params.get('resolution', 100)
        return resolution

    elif descriptor == 'euler_characteristic_transform':
        n_directions = params.get('n_directions', 32)
        n_heights = params.get('n_heights', 20)
        return n_directions * n_heights

    elif descriptor == 'persistence_statistics':
        subset = params.get('subset', 'basic')
        dims = {'basic': 28, 'extended': 42, 'full': 62}
        return dims.get(subset, 28)

    elif descriptor == 'lbp_texture':
        P = params.get('P', 8)
        R = params.get('R', 1.0)
        return P + 2  # uniform LBP

    elif descriptor == 'edge_histogram':
        n_orientation_bins = params.get('n_orientation_bins', 8)
        n_spatial_cells = params.get('n_spatial_cells', 10)
        return n_orientation_bins * n_spatial_cells

    else:
        return 100  # Default


def analyze_sweep_results(results: Dict) -> Dict:
    """Analyze sweep results to find best parameters."""

    if not results.get('results'):
        return {}

    # Sort by mean accuracy
    sorted_configs = sorted(
        results['results'],
        key=lambda x: x['mean_accuracy'],
        reverse=True
    )

    best_config = sorted_configs[0]
    worst_config = sorted_configs[-1]

    # Analyze parameter sensitivity
    param_sensitivity = {}
    param_grid = results.get('param_grid', {})

    for param_name, param_values in param_grid.items():
        param_accs = {v: [] for v in param_values}

        for config_result in results['results']:
            param_val = config_result['params'].get(param_name)
            if param_val in param_accs:
                param_accs[param_val].append(config_result['mean_accuracy'])

        param_sensitivity[param_name] = {
            v: {
                'mean': np.mean(accs) if accs else 0.0,
                'std': np.std(accs) if accs else 0.0,
                'n': len(accs),
            }
            for v, accs in param_accs.items()
        }

    analysis = {
        'best_params': best_config['params'],
        'best_accuracy': best_config['mean_accuracy'],
        'worst_params': worst_config['params'],
        'worst_accuracy': worst_config['mean_accuracy'],
        'accuracy_range': best_config['mean_accuracy'] - worst_config['mean_accuracy'],
        'param_sensitivity': param_sensitivity,
        'top_5_configs': [
            {'params': c['params'], 'accuracy': c['mean_accuracy']}
            for c in sorted_configs[:5]
        ],
    }

    return analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 4.2: Hyperparameter Sensitivity Study")
    parser.add_argument('--trial', action='store_true', help='Quick trial with 50 samples')
    parser.add_argument('--full', action='store_true', help='Full run with 5000 samples')
    parser.add_argument('--descriptor', type=str, default=None,
                       help='Run for specific descriptor')
    parser.add_argument('--object-type', type=str, default=None,
                       help='Run for specific object type')
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--n-folds', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--max-configs', type=int, default=None,
                       help='Maximum configs per descriptor')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

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
    print("  EXPERIMENT 4.2: HYPERPARAMETER SENSITIVITY STUDY")
    print("=" * 70)
    print(f"  n_samples: {n_samples}")
    print(f"  n_folds: {n_folds}")
    print(f"  seed: {args.seed}")
    print("=" * 70)

    all_results = {}

    # Determine what to run
    if args.descriptor and args.object_type:
        # Single descriptor + object type
        descriptors_to_run = [args.descriptor]
        object_types_to_run = {args.object_type: TRIAL_DATASETS.get(args.object_type, [])}
    elif args.descriptor:
        # Single descriptor, all object types
        descriptors_to_run = [args.descriptor]
        object_types_to_run = TRIAL_DATASETS
    elif args.object_type:
        # All descriptors, single object type
        descriptors_to_run = list(PARAM_GRIDS.keys())
        object_types_to_run = {args.object_type: TRIAL_DATASETS.get(args.object_type, [])}
    else:
        # Trial: top 3 descriptors for trial object type
        descriptors_to_run = ['persistence_image', 'ATOL', 'betti_curves']
        object_types_to_run = {'discrete_cells': ['BloodMNIST']}

    for obj_type, datasets in object_types_to_run.items():
        all_results[obj_type] = {}

        for descriptor in descriptors_to_run:
            results = run_hyperparam_sweep(
                descriptor=descriptor,
                object_type=obj_type,
                datasets=datasets,
                n_samples=n_samples,
                n_folds=n_folds,
                seed=args.seed,
                n_jobs=args.n_jobs,
                max_configs=args.max_configs,
            )

            if results:
                analysis = analyze_sweep_results(results)
                results['analysis'] = analysis
                all_results[obj_type][descriptor] = results

                # Print best params
                if analysis:
                    print(f"\n  BEST for {descriptor} on {obj_type}:")
                    print(f"    Params: {analysis['best_params']}")
                    print(f"    Accuracy: {analysis['best_accuracy']:.3f}")
                    print(f"    Range: {analysis['accuracy_range']:.3f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_PATH / f"exp4_hyperparam_sweep_n{n_samples}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
