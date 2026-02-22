#!/usr/bin/env python
"""
Benchmark4: Check completion status.

Shows which (dataset, descriptor) pairs have results and which are missing.
Also shows basic stats from completed results.

Usage:
    python benchmarks/benchmark4/check_status.py
    python benchmarks/benchmark4/check_status.py --summary
    python benchmarks/benchmark4/check_status.py --ph-cache
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse

from RuleBenchmark.benchmark4.config import (
    DATASETS, ALL_DESCRIPTORS, ACTIVE_CLASSIFIERS,
    RAW_RESULTS_PATH, PH_CACHE_PATH,
)


def check_ph_cache():
    """Check PH cache completion."""
    print("=" * 70)
    print("  PH Cache Status")
    print("=" * 70)
    for dataset, cfg in DATASETS.items():
        cache_file = PH_CACHE_PATH / f"{dataset}_n{cfg['n_samples']}.pkl"
        if cache_file.exists():
            size_mb = cache_file.stat().st_size / 1e6
            print(f"  {dataset:25s}  OK  ({size_mb:.1f} MB)")
        else:
            print(f"  {dataset:25s}  MISSING")
    print()


def check_results(summary=False):
    """Check evaluation result completion."""
    print("=" * 70)
    print("  Evaluation Results Status")
    print("=" * 70)

    done = 0
    missing = 0
    errors = 0
    results_by_dataset = {}

    for dataset in DATASETS:
        results_by_dataset[dataset] = {}
        for desc in ALL_DESCRIPTORS:
            result_file = RAW_RESULTS_PATH / f"{dataset}_{desc}.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    if 'error' in data:
                        results_by_dataset[dataset][desc] = 'ERROR'
                        errors += 1
                    else:
                        best_clf = max(
                            data['classifiers'].items(),
                            key=lambda x: x[1]['balanced_accuracy_mean'])
                        results_by_dataset[dataset][desc] = (
                            best_clf[0],
                            best_clf[1]['balanced_accuracy_mean'])
                        done += 1
                except Exception:
                    results_by_dataset[dataset][desc] = 'CORRUPT'
                    errors += 1
            else:
                results_by_dataset[dataset][desc] = None
                missing += 1

    total = len(DATASETS) * len(ALL_DESCRIPTORS)
    print(f"  Done:    {done:4d} / {total}")
    print(f"  Missing: {missing:4d} / {total}")
    print(f"  Errors:  {errors:4d} / {total}")
    print()

    if summary and done > 0:
        print("=" * 70)
        print("  Best Results per Dataset")
        print("=" * 70)
        for dataset, descs in results_by_dataset.items():
            completed = {d: v for d, v in descs.items()
                        if isinstance(v, tuple)}
            if completed:
                best_desc = max(completed.items(),
                               key=lambda x: x[1][1])
                print(f"  {dataset:25s}  {best_desc[0]:30s}  "
                      f"{best_desc[1][0]:12s}  "
                      f"bal_acc={best_desc[1][1]:.4f}")
            else:
                n_missing = sum(1 for v in descs.values() if v is None)
                print(f"  {dataset:25s}  -- no results yet "
                      f"({n_missing} missing) --")
        print()

    # Show missing datasets
    missing_datasets = [d for d in DATASETS
                       if all(v is None for v in results_by_dataset[d].values())]
    if missing_datasets:
        print(f"  Datasets with NO results: {len(missing_datasets)}")
        for d in missing_datasets:
            print(f"    - {d}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true',
                       help='Show best results per dataset')
    parser.add_argument('--ph-cache', action='store_true',
                       help='Check PH cache status')
    args = parser.parse_args()

    if args.ph_cache:
        check_ph_cache()
    check_results(summary=args.summary)


if __name__ == '__main__':
    main()
