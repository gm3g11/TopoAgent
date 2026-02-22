#!/usr/bin/env python
"""
Experiment 4 - Sigma Extension for persistence_image

3/5 datasets showed monotonic accuracy increase up to sigma=0.25 (max tested).
This script extends the search to sigma=[0.3, 0.35, 0.4] for those 3 datasets.

Only tests: BloodMNIST, PathMNIST, OrganAMNIST
With all 3 weight functions: linear, squared, const
= 9 configs × 3 datasets = 27 evaluations

Usage:
    python benchmarks/benchmark3/exp4_sigma_extension.py
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

from RuleBenchmark.benchmark3.exp4_config import RESULTS_PATH
from RuleBenchmark.benchmark3.exp4_dimension_study import (
    load_dataset, compute_ph_gudhi, extract_descriptor, evaluate_with_cv,
)
from RuleBenchmark.benchmark3.exp4_precompute_ph import load_cached_ph


# Datasets where sigma was at boundary (monotonically increasing to 0.25)
BOUNDARY_DATASETS = {
    "BloodMNIST": {"resolution": 10, "dim_size": 200, "classifier": "TabPFN"},
    "PathMNIST": {"resolution": 26, "dim_size": 1352, "classifier": "TabPFN"},
    "OrganAMNIST": {"resolution": 12, "dim_size": 288, "classifier": "TabPFN"},
}

# Extended sigma values
EXTENDED_SIGMAS = [0.3, 0.35, 0.4]
WEIGHT_FUNCTIONS = ["linear", "squared", "const"]


def main():
    parser = argparse.ArgumentParser(description="Exp4: PI Sigma Extension")
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cache', action='store_true', default=True)
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset (default: all 3 boundary datasets)')
    args = parser.parse_args()

    print("=" * 70)
    print("  PERSISTENCE IMAGE - SIGMA EXTENSION")
    print("=" * 70)
    print(f"  Extended sigmas: {EXTENDED_SIGMAS}")
    print(f"  Weight functions: {WEIGHT_FUNCTIONS}")
    print(f"  Datasets: {list(BOUNDARY_DATASETS.keys())}")
    print(f"  Total configs: {len(EXTENDED_SIGMAS) * len(WEIGHT_FUNCTIONS)} per dataset")
    print("=" * 70)
    sys.stdout.flush()

    # Determine which datasets to test
    if args.dataset:
        datasets = {args.dataset: BOUNDARY_DATASETS[args.dataset]}
    else:
        datasets = BOUNDARY_DATASETS

    all_results = {
        "experiment": "sigma_extension",
        "descriptor": "persistence_image",
        "extended_sigmas": EXTENDED_SIGMAS,
        "weight_functions": WEIGHT_FUNCTIONS,
        "n_samples": args.n_samples,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "results": {},
    }

    for dataset, config in datasets.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset}")
        print(f"  Resolution: {config['resolution']}, Dim: {config['dim_size']}D")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Load data
        if args.use_cache:
            try:
                cache_data = load_cached_ph(dataset, args.n_samples)
                if 'diagrams_sublevel' in cache_data:
                    diagrams = cache_data['diagrams_sublevel']
                    print(f"  Loaded PH from cache ({len(diagrams)} samples)")
                else:
                    diagrams = cache_data['diagrams']
                    print(f"  Loaded PH from cache ({len(diagrams)} samples, old format)")
                labels = cache_data['labels']
                images = None  # Not needed for PI
            except FileNotFoundError:
                print(f"  Cache not found, loading fresh...")
                images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
                diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)
        else:
            images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
            diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)
        sys.stdout.flush()

        dataset_results = []

        for sigma in EXTENDED_SIGMAS:
            for weight_fn in WEIGHT_FUNCTIONS:
                params = {
                    "sigma": sigma,
                    "weight_function": weight_fn,
                    "resolution": config["resolution"],
                }

                print(f"  sigma={sigma}, weight={weight_fn}...", end=" ")
                sys.stdout.flush()
                start = time.time()

                try:
                    features = extract_descriptor(
                        "persistence_image", images, diagrams, params, config["dim_size"],
                    )
                    cv_result = evaluate_with_cv(
                        features, labels,
                        n_folds=args.n_folds,
                        seed=args.seed,
                        dataset_name=dataset,
                        classifier_name=config["classifier"],
                    )

                    acc = cv_result['mean']
                    std = cv_result['std']
                    elapsed = time.time() - start
                    print(f"{acc:.4f} ± {std:.4f} ({elapsed:.1f}s)")
                    sys.stdout.flush()

                    dataset_results.append({
                        "sigma": sigma,
                        "weight_function": weight_fn,
                        "accuracy": acc,
                        "std": std,
                        "time": elapsed,
                    })

                except Exception as e:
                    print(f"ERROR: {e}")
                    sys.stdout.flush()
                    dataset_results.append({
                        "sigma": sigma,
                        "weight_function": weight_fn,
                        "accuracy": 0.0,
                        "error": str(e),
                    })

        # Find best in extension
        if dataset_results:
            best = max(dataset_results, key=lambda x: x.get("accuracy", 0))
            print(f"\n  BEST EXTENSION: sigma={best['sigma']}, weight={best['weight_function']} = {best['accuracy']:.4f}")
            sys.stdout.flush()

        all_results["results"][dataset] = {
            "config": config,
            "extension_results": dataset_results,
            "best_extension": best if dataset_results else None,
        }

        # Memory cleanup
        del diagrams
        if images is not None:
            del images
        gc.collect()

    # Load original results for comparison
    print(f"\n{'='*70}")
    print("  COMPARISON: Original vs Extended")
    print(f"{'='*70}")
    sys.stdout.flush()

    orig_path = RESULTS_PATH / "exp4_parameter_search_n2000.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig_data = json.load(f)

        for dataset in datasets:
            orig_best = orig_data["results"].get(dataset, {}).get("persistence_image", {}).get("best_accuracy", 0)
            ext_best = all_results["results"][dataset].get("best_extension", {})
            ext_acc = ext_best.get("accuracy", 0) if ext_best else 0

            trend = "↑ IMPROVED" if ext_acc > orig_best else ("= PEAKED" if ext_acc <= orig_best else "↓ WORSE")
            print(f"  {dataset}: original={orig_best:.4f}, extended={ext_acc:.4f} {trend}")
            sys.stdout.flush()

    # Save results
    output_path = RESULTS_PATH / "exp4_sigma_extension_n2000.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file before overwriting
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
