#!/usr/bin/env python
"""
Experiment 4.3: Rule Validation

Validate derived rules on held-out datasets.
Compare rule-based selection vs fixed defaults.

Usage:
    python benchmarks/benchmark3/exp4_rule_validation.py --trial
    python benchmarks/benchmark3/exp4_rule_validation.py --rules-file results/exp4/rules.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Dict, List, Any

from RuleBenchmark.benchmark3.exp4_config import (
    DIMENSION_CONFIGS, EXPECTED_DIMS, OBJECT_TYPE_DATASETS,
    ALL_DESCRIPTORS, LEARNED_DESCRIPTORS, RESULTS_PATH,
    DATASET_OBJECT_TYPE, get_params, get_dimension,
)
from RuleBenchmark.benchmark3.exp4_dimension_study import (
    load_dataset, compute_ph_gudhi, extract_descriptor,
    evaluate_with_cv, evaluate_learned_descriptor_cv,
)


# =============================================================================
# DEFAULT RULES (Medium dimensions, default params)
# =============================================================================

DEFAULT_RULES = {
    'dimension_level': 'medium',
    'params': {
        'persistence_image': {'resolution': 20, 'sigma': 0.1, 'weight_function': 'linear'},
        'persistence_landscapes': {'n_layers': 4, 'n_bins': 100, 'combine_dims': True},
        'persistence_silhouette': {'n_bins': 100, 'power': 1.0},
        'betti_curves': {'n_bins': 100, 'normalize': False},
        'persistence_entropy': {'mode': 'vector', 'n_bins': 100},
        'persistence_codebook': {'codebook_size': 32},
        'tropical_coordinates': {'max_terms': 5},
        'ATOL': {'n_centers': 16},
        'template_functions': {'n_templates': 25, 'template_type': 'tent'},
        'minkowski_functionals': {'n_thresholds': 10, 'adaptive': True},
        'euler_characteristic_curve': {'resolution': 100},
        'euler_characteristic_transform': {'n_directions': 32, 'n_heights': 20},
        'persistence_statistics': {'subset': 'basic'},
        'lbp_texture': {'scales': [(8, 1.0), (16, 2.0), (24, 3.0)]},
        'edge_histogram': {'n_orientation_bins': 8, 'n_spatial_cells': 10},
    },
}

# Example derived rules (will be loaded from Phase 1/2 results)
EXAMPLE_DERIVED_RULES = {
    'discrete_cells': {
        'dimension_level': {
            'persistence_image': 'medium',
            'ATOL': 'medium',
            'betti_curves': 'medium',
            'persistence_landscapes': 'small',
        },
        'best_params': {
            'persistence_image': {'resolution': 20, 'sigma': 0.1, 'weight_function': 'linear'},
            'ATOL': {'n_centers': 16},
            'betti_curves': {'n_bins': 100, 'normalize': False},
        },
        'recommended_descriptors': ['ATOL', 'persistence_image', 'betti_curves'],
    },
    'glands_lumens': {
        'dimension_level': {
            'persistence_image': 'large',
            'persistence_landscapes': 'large',
        },
        'best_params': {
            'persistence_image': {'resolution': 25, 'sigma': 0.15, 'weight_function': 'linear'},
            'persistence_landscapes': {'n_layers': 7, 'n_bins': 100, 'combine_dims': True},
        },
        'recommended_descriptors': ['persistence_image', 'persistence_landscapes', 'template_functions'],
    },
}


# =============================================================================
# RULE VALIDATION
# =============================================================================

def validate_rules(
    derived_rules: Dict,
    validation_datasets: List[str],
    n_samples: int = 50,
    n_folds: int = 3,
    seed: int = 42,
    n_jobs: int = 4,
) -> Dict:
    """Validate derived rules on held-out datasets."""

    results = {
        'validation_datasets': validation_datasets,
        'n_samples': n_samples,
        'n_folds': n_folds,
        'comparisons': {},
    }

    for dataset in validation_datasets:
        print(f"\n{'='*60}")
        print(f"  Validating on: {dataset}")
        print(f"{'='*60}")

        obj_type = DATASET_OBJECT_TYPE.get(dataset, 'unknown')
        print(f"  Object type: {obj_type}")

        # Load data
        images, labels = load_dataset(dataset, n_samples=n_samples, seed=seed)
        diagrams = compute_ph_gudhi(images, n_jobs=n_jobs)

        results['comparisons'][dataset] = {
            'object_type': obj_type,
            'descriptor_comparisons': {},
        }

        # Get rules for this object type
        obj_rules = derived_rules.get(obj_type, {})
        recommended = obj_rules.get('recommended_descriptors', ALL_DESCRIPTORS[:5])

        for descriptor in recommended:
            print(f"\n  Descriptor: {descriptor}")

            # Get derived params
            dim_level = obj_rules.get('dimension_level', {}).get(descriptor, 'medium')
            derived_params = obj_rules.get('best_params', {}).get(
                descriptor,
                get_params(descriptor, dim_level)
            )
            derived_dim = get_dimension(descriptor, dim_level)

            # Get default params
            default_params = DEFAULT_RULES['params'].get(
                descriptor,
                get_params(descriptor, 'medium')
            )
            default_dim = get_dimension(descriptor, 'medium')

            print(f"    Derived: {derived_params} ({derived_dim}D)")
            print(f"    Default: {default_params} ({default_dim}D)")

            # Evaluate with derived rules
            print(f"    Evaluating derived...", end=' ')
            start = time.time()
            try:
                if descriptor in LEARNED_DESCRIPTORS:
                    derived_result = evaluate_learned_descriptor_cv(
                        descriptor, diagrams, labels, derived_params, derived_dim,
                        n_folds=n_folds, seed=seed, dataset_name=dataset,
                    )
                else:
                    derived_features = extract_descriptor(
                        descriptor, images, diagrams, derived_params, derived_dim,
                    )
                    derived_result = evaluate_with_cv(
                        derived_features, labels, n_folds=n_folds, seed=seed,
                        dataset_name=dataset,
                    )
                derived_acc = derived_result['mean']
                derived_time = time.time() - start
                print(f"{derived_acc:.3f} ({derived_time:.1f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                derived_acc = 0.0
                derived_time = 0.0

            # Evaluate with default rules
            print(f"    Evaluating default...", end=' ')
            start = time.time()
            try:
                if descriptor in LEARNED_DESCRIPTORS:
                    default_result = evaluate_learned_descriptor_cv(
                        descriptor, diagrams, labels, default_params, default_dim,
                        n_folds=n_folds, seed=seed, dataset_name=dataset,
                    )
                else:
                    default_features = extract_descriptor(
                        descriptor, images, diagrams, default_params, default_dim,
                    )
                    default_result = evaluate_with_cv(
                        default_features, labels, n_folds=n_folds, seed=seed,
                        dataset_name=dataset,
                    )
                default_acc = default_result['mean']
                default_time = time.time() - start
                print(f"{default_acc:.3f} ({default_time:.1f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                default_acc = 0.0
                default_time = 0.0

            # Compare
            improvement = derived_acc - default_acc
            print(f"    Improvement: {improvement:+.3f}")

            results['comparisons'][dataset]['descriptor_comparisons'][descriptor] = {
                'derived_params': derived_params,
                'derived_dim': derived_dim,
                'derived_accuracy': derived_acc,
                'derived_time': derived_time,
                'default_params': default_params,
                'default_dim': default_dim,
                'default_accuracy': default_acc,
                'default_time': default_time,
                'improvement': improvement,
            }

    return results


def summarize_validation(results: Dict) -> Dict:
    """Summarize validation results."""

    summary = {
        'n_datasets': len(results['validation_datasets']),
        'overall_improvement': [],
        'per_descriptor': {},
        'wins': 0,
        'losses': 0,
        'ties': 0,
    }

    for dataset, comparison in results['comparisons'].items():
        for desc, metrics in comparison['descriptor_comparisons'].items():
            improvement = metrics['improvement']
            summary['overall_improvement'].append(improvement)

            if desc not in summary['per_descriptor']:
                summary['per_descriptor'][desc] = []
            summary['per_descriptor'][desc].append(improvement)

            if improvement > 0.01:
                summary['wins'] += 1
            elif improvement < -0.01:
                summary['losses'] += 1
            else:
                summary['ties'] += 1

    summary['mean_improvement'] = np.mean(summary['overall_improvement']) if summary['overall_improvement'] else 0.0
    summary['std_improvement'] = np.std(summary['overall_improvement']) if summary['overall_improvement'] else 0.0

    for desc in summary['per_descriptor']:
        improvements = summary['per_descriptor'][desc]
        summary['per_descriptor'][desc] = {
            'mean': np.mean(improvements),
            'std': np.std(improvements),
            'n': len(improvements),
        }

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp 4.3: Rule Validation")
    parser.add_argument('--trial', action='store_true', help='Quick trial')
    parser.add_argument('--rules-file', type=str, default=None,
                       help='JSON file with derived rules')
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--n-folds', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    print("=" * 70)
    print("  EXPERIMENT 4.3: RULE VALIDATION")
    print("=" * 70)

    # Load derived rules
    if args.rules_file:
        with open(args.rules_file, 'r') as f:
            derived_rules = json.load(f)
        print(f"  Loaded rules from: {args.rules_file}")
    else:
        derived_rules = EXAMPLE_DERIVED_RULES
        print(f"  Using example rules")

    # Validation datasets (different from training)
    if args.trial:
        validation_datasets = ['DermaMNIST']
    else:
        # Use datasets not in TRIAL_DATASETS
        validation_datasets = ['TissueMNIST', 'OCTMNIST', 'PneumoniaMNIST']

    print(f"  Validation datasets: {validation_datasets}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  n_folds: {args.n_folds}")
    print("=" * 70)

    # Run validation
    results = validate_rules(
        derived_rules=derived_rules,
        validation_datasets=validation_datasets,
        n_samples=args.n_samples,
        n_folds=args.n_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    # Summarize
    summary = summarize_validation(results)
    results['summary'] = summary

    print(f"\n\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean improvement: {summary['mean_improvement']:+.3f} ± {summary['std_improvement']:.3f}")
    print(f"  Wins/Ties/Losses: {summary['wins']}/{summary['ties']}/{summary['losses']}")
    print(f"\n  Per-descriptor:")
    for desc, stats in summary['per_descriptor'].items():
        print(f"    {desc}: {stats['mean']:+.3f} ± {stats['std']:.3f} (n={stats['n']})")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_PATH / f"exp4_rule_validation_n{args.n_samples}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
