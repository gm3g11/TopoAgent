#!/usr/bin/env python
"""
Experiment 4 - Sigma Extension V2 for persistence_image

V1 extended sigma from [0.05-0.25] to [0.3, 0.35, 0.4].
sigma=0.4 still at boundary with monotonically increasing accuracy on 3 datasets.
This script extends further to sigma=[0.45, 0.5, 0.6] to find the actual peak.

Only tests: BloodMNIST, PathMNIST, OrganAMNIST
With all 3 weight functions: linear, squared, const
= 9 configs × 3 datasets = 27 evaluations

Usage:
    python benchmarks/benchmark3/exp4_sigma_extension_v2.py
    python benchmarks/benchmark3/exp4_sigma_extension_v2.py --dataset BloodMNIST
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


# Datasets where sigma was STILL at boundary after v1 extension (monotonically increasing to 0.4)
BOUNDARY_DATASETS = {
    "BloodMNIST": {"resolution": 10, "dim_size": 200, "classifier": "TabPFN"},
    "PathMNIST": {"resolution": 26, "dim_size": 1352, "classifier": "TabPFN"},
    "OrganAMNIST": {"resolution": 12, "dim_size": 288, "classifier": "TabPFN"},
}

# V2 extended sigma values (beyond v1's [0.3, 0.35, 0.4])
EXTENDED_SIGMAS_V2 = [0.45, 0.5, 0.6]
WEIGHT_FUNCTIONS = ["linear", "squared", "const"]


def load_previous_results():
    """Load results from original param search and v1 extension for comparison."""
    previous = {}

    # Load v1 extension results
    v1_path = RESULTS_PATH / "exp4_sigma_extension_n2000.json"
    if v1_path.exists():
        with open(v1_path) as f:
            v1_data = json.load(f)
        for dataset, result in v1_data.get("results", {}).items():
            best_ext = result.get("best_extension", {})
            if best_ext:
                previous[dataset] = {
                    "v1_best_sigma": best_ext.get("sigma"),
                    "v1_best_weight": best_ext.get("weight_function"),
                    "v1_best_accuracy": best_ext.get("accuracy"),
                }

    # Load original param search results
    orig_path = RESULTS_PATH / "exp4_parameter_search_n2000.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig_data = json.load(f)
        for dataset in BOUNDARY_DATASETS:
            orig_best = orig_data.get("results", {}).get(dataset, {}).get(
                "persistence_image", {}
            ).get("best_accuracy", 0)
            if dataset not in previous:
                previous[dataset] = {}
            previous[dataset]["original_best_accuracy"] = orig_best

    return previous


def main():
    parser = argparse.ArgumentParser(description="Exp4: PI Sigma Extension V2")
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-cache', action='store_true', default=True)
    parser.add_argument('--dataset', type=str, default=None,
                       choices=list(BOUNDARY_DATASETS.keys()),
                       help='Specific dataset (default: all 3 boundary datasets)')
    parser.add_argument('--sigmas', type=float, nargs='+', default=None,
                       help='Specific sigma values to test (default: 0.45, 0.5, 0.6)')
    args = parser.parse_args()

    # Override sigmas if specified
    global EXTENDED_SIGMAS_V2
    if args.sigmas:
        EXTENDED_SIGMAS_V2 = args.sigmas

    print("=" * 70)
    print("  PERSISTENCE IMAGE - SIGMA EXTENSION V2")
    print("=" * 70)
    print(f"  V2 sigmas: {EXTENDED_SIGMAS_V2}")
    print(f"  Weight functions: {WEIGHT_FUNCTIONS}")
    print(f"  Datasets: {list(BOUNDARY_DATASETS.keys())}")
    print(f"  Total configs: {len(EXTENDED_SIGMAS_V2) * len(WEIGHT_FUNCTIONS)} per dataset")
    print("=" * 70)
    sys.stdout.flush()

    # Load previous results for comparison
    previous = load_previous_results()
    if previous:
        print("\n  Previous best results:")
        for dataset, prev in previous.items():
            v1_acc = prev.get("v1_best_accuracy", "N/A")
            orig_acc = prev.get("original_best_accuracy", "N/A")
            print(f"    {dataset}: original={orig_acc}, v1_extension={v1_acc}")
        print()
    sys.stdout.flush()

    # Determine which datasets to test
    if args.dataset:
        datasets = {args.dataset: BOUNDARY_DATASETS[args.dataset]}
    else:
        datasets = BOUNDARY_DATASETS

    all_results = {
        "experiment": "sigma_extension_v2",
        "descriptor": "persistence_image",
        "extended_sigmas_v2": EXTENDED_SIGMAS_V2,
        "weight_functions": WEIGHT_FUNCTIONS,
        "n_samples": args.n_samples,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "previous_results": previous,
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
                images = None
            except FileNotFoundError:
                print(f"  Cache not found, loading fresh...")
                images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
                diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)
        else:
            images, labels = load_dataset(dataset, n_samples=args.n_samples, seed=args.seed)
            diagrams = compute_ph_gudhi(images, n_jobs=args.n_jobs)
        sys.stdout.flush()

        dataset_results = []

        for sigma in EXTENDED_SIGMAS_V2:
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

        # Find best in v2 extension
        best = None
        if dataset_results:
            best = max(dataset_results, key=lambda x: x.get("accuracy", 0))
            print(f"\n  BEST V2 EXTENSION: sigma={best['sigma']}, weight={best['weight_function']} = {best['accuracy']:.4f}")

            # Compare with v1
            prev = previous.get(dataset, {})
            v1_acc = prev.get("v1_best_accuracy", 0)
            if v1_acc:
                if best['accuracy'] > v1_acc:
                    print(f"  ↑ IMPROVED over v1: {v1_acc:.4f} → {best['accuracy']:.4f} (+{best['accuracy']-v1_acc:.4f})")
                elif best['accuracy'] < v1_acc:
                    print(f"  ↓ PEAKED at v1: v1={v1_acc:.4f} > v2={best['accuracy']:.4f} (sigma peak is between 0.35-0.45)")
                else:
                    print(f"  = PLATEAU: v1={v1_acc:.4f} == v2={best['accuracy']:.4f}")
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

    # Summary
    print(f"\n{'='*70}")
    print("  FULL SIGMA TRAJECTORY: Original → V1 → V2")
    print(f"{'='*70}")
    sys.stdout.flush()

    for dataset in datasets:
        prev = previous.get(dataset, {})
        orig = prev.get("original_best_accuracy", "?")
        v1 = prev.get("v1_best_accuracy", "?")
        v2_best = all_results["results"][dataset].get("best_extension", {})
        v2 = v2_best.get("accuracy", "?") if v2_best else "?"

        print(f"  {dataset}: original(σ≤0.25)={orig} → v1(σ≤0.4)={v1} → v2(σ≤0.6)={v2}")
        sys.stdout.flush()

    # Save results - include dataset name when running single dataset to avoid overwrites
    if args.dataset:
        output_path = RESULTS_PATH / f"exp4_sigma_extension_v2_{args.dataset}_n{args.n_samples}.json"
    else:
        output_path = RESULTS_PATH / f"exp4_sigma_extension_v2_n{args.n_samples}.json"
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
