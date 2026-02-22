#!/usr/bin/env python
"""
Experiment 4 - Boundary Extension for descriptors at search range limits.

Extends search ranges for descriptors that peaked at the boundary of
MEMORY_SAFE_DIMENSION_SEARCH.

Supported datasets and descriptors:

OrganAMNIST (MEDIUM priority):
  1. edge_histogram: cells=64 (MAX) -> test 72, 80, 96
  2. ATOL: n_centers=24 (near MAX) -> test 32, 40, 48
  3. tropical_coordinates: max_terms=20 (MAX) -> test 25, 30
  4. betti_curves: n_bins=50 (MIN) -> test 30, 20, 10

PathMNIST (LOW priority):
  5. tropical_coordinates: max_terms=20 (MAX) -> test 25, 30

DermaMNIST (LOW priority):
  6. tropical_coordinates: max_terms=20 (MAX) -> test 25, 30

RetinaMNIST (LOW priority):
  7. persistence_landscapes: n_layers=15 (MIN) -> test 10, 12, 13

BloodMNIST (LOW priority):
  8. minkowski_functionals: n_thresh=40 (MAX) -> test 45, 50, 60

RetinaMNIST (LOW priority):
  9. minkowski_functionals: n_thresh=40 (MAX) -> test 45, 50, 60

Usage:
    python benchmarks/benchmark3/exp4_boundary_extension.py --dataset OrganAMNIST
    python benchmarks/benchmark3/exp4_boundary_extension.py --dataset OrganAMNIST --descriptor edge_histogram
    python benchmarks/benchmark3/exp4_boundary_extension.py --dataset PathMNIST --descriptor tropical_coordinates
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

from RuleBenchmark.benchmark3.exp4_config import (
    RESULTS_PATH, LEARNED_DESCRIPTORS, IMAGE_BASED_DESCRIPTORS,
)
from RuleBenchmark.benchmark3.exp4_dimension_study import (
    load_dataset, compute_ph_gudhi, extract_descriptor, evaluate_with_cv,
    evaluate_learned_descriptor_cv,
)
from RuleBenchmark.benchmark3.exp4_precompute_ph import load_cached_ph


# ============================================================================
# All boundary extension configurations, keyed by dataset
# ============================================================================
ALL_BOUNDARY_EXTENSIONS = {
    "OrganAMNIST": {
        "edge_histogram": {
            "control_param": "n_spatial_cells",
            "current_best": {"value": 64, "dim": 512, "accuracy": 0.837},
            "extension_values": [72, 80, 96],
            "extension_dims": [576, 640, 768],
            "fixed_params": {"n_orientation_bins": 8},
            "direction": "higher",
            "priority": "MEDIUM",
        },
        "ATOL": {
            "control_param": "n_centers",
            "current_best": {"value": 24, "dim": 48, "accuracy": 0.757},
            "extension_values": [32, 40, 48],
            "extension_dims": [64, 80, 96],
            "fixed_params": {},
            "direction": "higher",
            "priority": "MEDIUM",
        },
        "tropical_coordinates": {
            "control_param": "max_terms",
            "current_best": {"value": 20, "dim": 160, "accuracy": 0.622},
            "extension_values": [25, 30],
            "extension_dims": [200, 240],
            "fixed_params": {},
            "direction": "higher",
            "priority": "MEDIUM",
        },
        "betti_curves": {
            "control_param": "n_bins",
            "current_best": {"value": 50, "dim": 100, "accuracy": 0.682},
            "extension_values": [30, 20, 10],
            "extension_dims": [60, 40, 20],
            "fixed_params": {"normalize": False},
            "direction": "lower",
            "priority": "MEDIUM",
        },
    },
    "PathMNIST": {
        "tropical_coordinates": {
            "control_param": "max_terms",
            "current_best": {"value": 20, "dim": 160, "accuracy": 0.701},
            "extension_values": [25, 30],
            "extension_dims": [200, 240],
            "fixed_params": {},
            "direction": "higher",
            "priority": "LOW",
        },
    },
    "DermaMNIST": {
        "tropical_coordinates": {
            "control_param": "max_terms",
            "current_best": {"value": 20, "dim": 160, "accuracy": 0.406},
            "extension_values": [25, 30],
            "extension_dims": [200, 240],
            "fixed_params": {},
            "direction": "higher",
            "priority": "LOW",
        },
    },
    "RetinaMNIST": {
        "persistence_landscapes": {
            "control_param": "n_layers",
            "current_best": {"value": 15, "dim": 1500, "accuracy": 0.362},
            "extension_values": [13, 12, 10],
            "extension_dims": [1300, 1200, 1000],
            "fixed_params": {"n_bins": 100, "combine_dims": True},
            "direction": "lower",
            "priority": "LOW",
            "classifier": "XGBoost",
        },
        "minkowski_functionals": {
            "control_param": "n_thresholds",
            "current_best": {"value": 40, "dim": 120, "accuracy": 0.211},
            "extension_values": [45, 50, 60],
            "extension_dims": [135, 150, 180],
            "fixed_params": {"adaptive": False},
            "direction": "higher",
            "priority": "LOW",
        },
    },
    "BloodMNIST": {
        "minkowski_functionals": {
            "control_param": "n_thresholds",
            "current_best": {"value": 40, "dim": 120, "accuracy": 0.611},
            "extension_values": [45, 50, 60],
            "extension_dims": [135, 150, 180],
            "fixed_params": {"adaptive": False},
            "direction": "higher",
            "priority": "LOW",
        },
    },
}

# Build valid choices for argparse
ALL_DATASETS = list(ALL_BOUNDARY_EXTENSIONS.keys())
ALL_DESCRIPTORS_FLAT = set()
for ds_descs in ALL_BOUNDARY_EXTENSIONS.values():
    ALL_DESCRIPTORS_FLAT.update(ds_descs.keys())
ALL_DESCRIPTORS_FLAT = sorted(ALL_DESCRIPTORS_FLAT)


def evaluate_descriptor(
    descriptor, dataset_name, images, diagrams, labels, params, dim,
    n_folds=5, seed=42, classifier_name="TabPFN",
):
    """Evaluate a single descriptor configuration."""
    if descriptor in LEARNED_DESCRIPTORS:
        cv_result = evaluate_learned_descriptor_cv(
            descriptor, diagrams, labels, params, dim,
            n_folds=n_folds, seed=seed,
            dataset_name=dataset_name,
            classifier_name=classifier_name,
        )
    else:
        features = extract_descriptor(
            descriptor, images, diagrams, params, dim,
        )
        cv_result = evaluate_with_cv(
            features, labels,
            n_folds=n_folds, seed=seed,
            dataset_name=dataset_name,
            classifier_name=classifier_name,
        )
    return cv_result


def main():
    parser = argparse.ArgumentParser(description="Exp4: Boundary Extension")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=ALL_DATASETS,
                       help='Dataset to run boundary extension on')
    parser.add_argument('--descriptor', type=str, default=None,
                       choices=ALL_DESCRIPTORS_FLAT,
                       help='Specific descriptor (default: all for that dataset)')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cache', action='store_true', default=True)
    parser.add_argument('--classifier', type=str, default=None,
                       choices=['TabPFN', 'XGBoost'],
                       help='Override classifier (default: per-descriptor config)')
    args = parser.parse_args()

    dataset = args.dataset
    dataset_extensions = ALL_BOUNDARY_EXTENSIONS.get(dataset, {})

    if args.descriptor:
        if args.descriptor not in dataset_extensions:
            print(f"ERROR: {args.descriptor} has no boundary extension for {dataset}")
            print(f"Available for {dataset}: {list(dataset_extensions.keys())}")
            sys.exit(1)
        descriptors = {args.descriptor: dataset_extensions[args.descriptor]}
    else:
        descriptors = dataset_extensions

    if not descriptors:
        print(f"ERROR: No boundary extensions defined for {dataset}")
        sys.exit(1)

    print("=" * 70)
    print(f"  BOUNDARY EXTENSION - {dataset}")
    print("=" * 70)
    for desc, config in descriptors.items():
        print(f"  {desc}: {config['control_param']}={config['current_best']['value']} "
              f"(current best, acc={config['current_best']['accuracy']:.3f})")
        print(f"    Extension: {config['extension_values']} ({config['direction']})"
              f" [{config.get('priority', 'MEDIUM')}]")
    print(f"  Dataset: {dataset}")
    print("=" * 70)
    sys.stdout.flush()

    # Load data once
    print(f"\n  Loading data...")
    needs_images = any(d in IMAGE_BASED_DESCRIPTORS for d in descriptors)
    needs_ph = any(d not in IMAGE_BASED_DESCRIPTORS for d in descriptors)

    images = None
    diagrams = None
    labels = None

    if args.use_cache and needs_ph:
        try:
            cache_data = load_cached_ph(dataset, args.n_samples)
            if 'diagrams_sublevel' in cache_data:
                diagrams = cache_data['diagrams_sublevel']
                print(f"  Loaded PH from cache ({len(diagrams)} samples)")
            else:
                diagrams = cache_data['diagrams']
                print(f"  Loaded PH from cache ({len(diagrams)} samples, old format)")
            labels = cache_data['labels']
        except FileNotFoundError:
            print(f"  PH cache not found, computing from scratch...")
            images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
            diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)
    elif needs_ph:
        images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
        diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)

    # For image-based descriptors, load images if not already loaded
    if needs_images and images is None:
        print(f"  Loading images for image-based descriptors...")
        images, labels_img = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
        if labels is None:
            labels = labels_img

    sys.stdout.flush()

    all_results = {
        "experiment": "boundary_extension",
        "dataset": dataset,
        "n_samples": args.n_samples,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "results": {},
    }

    for descriptor, config in descriptors.items():
        # Determine classifier
        if args.classifier:
            classifier = args.classifier
        else:
            classifier = config.get("classifier", "TabPFN")

        print(f"\n{'='*60}")
        print(f"  Descriptor: {descriptor}")
        print(f"  Current best: {config['control_param']}={config['current_best']['value']} "
              f"({config['current_best']['dim']}D) = {config['current_best']['accuracy']:.3f}")
        print(f"  Testing: {config['extension_values']}")
        print(f"  Classifier: {classifier}")
        print(f"{'='*60}")
        sys.stdout.flush()

        control_param = config['control_param']
        ext_results = []

        for value, dim in zip(config['extension_values'], config['extension_dims']):
            params = config['fixed_params'].copy()
            params[control_param] = value

            print(f"    {control_param}={value} ({dim}D)...", end=" ")
            sys.stdout.flush()
            start = time.time()

            try:
                cv_result = evaluate_descriptor(
                    descriptor, dataset, images, diagrams, labels, params, dim,
                    n_folds=args.n_folds, seed=args.seed,
                    classifier_name=classifier,
                )

                if cv_result.get('oom', False):
                    print(f"OOM at {dim}D")
                    ext_results.append({
                        "value": value, "dim": dim,
                        "accuracy": -1.0, "oom": True,
                    })
                    break

                acc = cv_result['mean']
                std = cv_result['std']
                elapsed = time.time() - start
                print(f"{acc:.4f} +/- {std:.4f} ({elapsed:.1f}s)")
                sys.stdout.flush()

                ext_results.append({
                    "value": value, "dim": dim,
                    "accuracy": acc, "std": std,
                    "time": elapsed,
                })

            except Exception as e:
                error_str = str(e)
                if 'CUDA' in error_str or 'out of memory' in error_str.lower():
                    print(f"OOM at {dim}D")
                    ext_results.append({
                        "value": value, "dim": dim,
                        "accuracy": -1.0, "oom": True, "error": error_str,
                    })
                    break
                else:
                    print(f"ERROR: {e}")
                    ext_results.append({
                        "value": value, "dim": dim,
                        "accuracy": 0.0, "error": error_str,
                    })

        # Analyze results
        valid = [r for r in ext_results if r.get('accuracy', -1) >= 0]
        best_ext = None
        if valid:
            best_ext = max(valid, key=lambda x: x['accuracy'])
            current_acc = config['current_best']['accuracy']

            if best_ext['accuracy'] > current_acc:
                print(f"\n  >> IMPROVED: {control_param}={best_ext['value']} ({best_ext['dim']}D) "
                      f"= {best_ext['accuracy']:.4f}")
                print(f"    Previous: {config['current_best']['value']} "
                      f"({config['current_best']['dim']}D) = {current_acc:.4f}")
                print(f"    Gain: +{best_ext['accuracy'] - current_acc:.4f}")
            else:
                print(f"\n  PEAKED: Current best confirmed at "
                      f"{control_param}={config['current_best']['value']}")
                print(f"    Best extension: {control_param}={best_ext['value']} "
                      f"= {best_ext['accuracy']:.4f} (<= {current_acc:.4f})")
        else:
            print(f"\n  FAILED: All extension values failed (OOM or error)")
        sys.stdout.flush()

        all_results["results"][descriptor] = {
            "config": {k: v for k, v in config.items() if k != 'priority'},
            "classifier": classifier,
            "extension_results": ext_results,
            "best_extension": best_ext,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"  BOUNDARY EXTENSION SUMMARY - {dataset}")
    print(f"{'='*70}")
    for descriptor, result in all_results["results"].items():
        config = result["config"]
        best_ext = result.get("best_extension")
        current = config["current_best"]["accuracy"]
        if best_ext and best_ext["accuracy"] > current:
            print(f"  {descriptor}: {current:.3f} -> {best_ext['accuracy']:.3f} "
                  f"(+{best_ext['accuracy']-current:.3f}, "
                  f"{config['control_param']}={best_ext['value']})")
        elif best_ext:
            print(f"  {descriptor}: {current:.3f} (peaked, extension max={best_ext['accuracy']:.3f})")
        else:
            print(f"  {descriptor}: {current:.3f} (extension failed)")
    sys.stdout.flush()

    # Save results - include descriptor name when running single to avoid overwrites
    if args.descriptor:
        output_path = RESULTS_PATH / (
            f"exp4_boundary_extension_{dataset}_{args.descriptor}_n{args.n_samples}.json"
        )
    else:
        output_path = RESULTS_PATH / (
            f"exp4_boundary_extension_{dataset}_n{args.n_samples}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        import shutil
        backup_path = output_path.with_suffix(f'.backup_{int(time.time())}.json')
        shutil.copy2(output_path, backup_path)
        print(f"\n  Backed up existing file to: {backup_path}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
