#!/usr/bin/env python
"""
Check PH cache status for all datasets.

Usage:
    python benchmarks/benchmark3/check_ph_cache.py
    python benchmarks/benchmark3/check_ph_cache.py --n-samples 2000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import pickle
from typing import Dict, List

from RuleBenchmark.benchmark3.exp4_config import OBJECT_TYPE_DATASETS, RESULTS_PATH


PH_CACHE_PATH = RESULTS_PATH / "ph_cache"


def find_cache_file(dataset: str, n_samples: int = 2000):
    """Find cache file for a dataset (handles variable sample counts)."""
    # Try exact match first
    exact_path = PH_CACHE_PATH / f"{dataset}_n{n_samples}_combined.pkl"
    if exact_path.exists():
        return exact_path, 'exact'

    # Look for any cache with this dataset
    pattern = f"{dataset}_n*_combined.pkl"
    matches = list(PH_CACHE_PATH.glob(pattern))
    if matches:
        # Return the one with highest n_samples
        best = max(matches, key=lambda p: int(p.stem.split('_n')[1].split('_')[0]))
        return best, 'found'

    # Check old format
    old_path = PH_CACHE_PATH / f"{dataset}_n{n_samples}.pkl"
    if old_path.exists():
        return old_path, 'old_format'

    return None, 'missing'


def check_cache_status(n_samples: int = 2000) -> Dict:
    """Check cache status for all datasets."""

    # Get all unique datasets
    datasets = []
    for obj_type, ds_list in OBJECT_TYPE_DATASETS.items():
        datasets.extend(ds_list)
    datasets = list(dict.fromkeys(datasets))

    results = {
        'cached': [],
        'missing': [],
        'old_format': [],
        'details': {},
    }

    print("=" * 70)
    print(f"  PH Cache Status (target n_samples={n_samples})")
    print("=" * 70)
    print(f"  Cache path: {PH_CACHE_PATH}")
    print("=" * 70)

    for dataset in datasets:
        cache_path, status = find_cache_file(dataset, n_samples)

        if status in ['exact', 'found'] and cache_path:
            # Load and check
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            n_loaded = data.get('n_samples', 0)
            n_requested = data.get('n_samples_requested', n_loaded)
            has_sublevel = 'diagrams_sublevel' in data and data['diagrams_sublevel'] is not None
            has_superlevel = 'diagrams_superlevel' in data and data['diagrams_superlevel'] is not None
            ph_library = data.get('ph_library', 'unknown')
            time_sub = data.get('ph_time_sublevel', 0)
            time_sup = data.get('ph_time_superlevel', 0)

            cache_status = "OK" if (has_sublevel and has_superlevel) else "INCOMPLETE"
            if n_loaded < n_samples and n_loaded < n_requested:
                cache_status = "PARTIAL"
            results['cached'].append(dataset)
            results['details'][dataset] = {
                'status': cache_status,
                'n_samples': n_loaded,
                'n_samples_requested': n_requested,
                'has_sublevel': has_sublevel,
                'has_superlevel': has_superlevel,
                'ph_library': ph_library,
                'time_sublevel': time_sub,
                'time_superlevel': time_sup,
            }

            note = ""
            if n_loaded < n_samples:
                note = f" (max avail)"
            print(f"  {dataset:20s} [{cache_status:8s}] n={n_loaded:4d}{note}, "
                  f"lib={ph_library}, time={time_sub+time_sup:.1f}s")

        elif status == 'old_format':
            # Old format exists
            results['old_format'].append(dataset)
            results['details'][dataset] = {'status': 'OLD_FORMAT'}
            print(f"  {dataset:20s} [OLD FORMAT] - needs re-run with --force")

        else:
            # Missing
            results['missing'].append(dataset)
            results['details'][dataset] = {'status': 'MISSING'}
            print(f"  {dataset:20s} [MISSING]")

    # Summary
    print("=" * 70)
    print(f"  Summary:")
    print(f"    Cached (combined): {len(results['cached'])}/{len(datasets)}")
    print(f"    Old format:        {len(results['old_format'])}")
    print(f"    Missing:           {len(results['missing'])}")

    if results['missing']:
        print(f"\n  Missing datasets: {', '.join(results['missing'])}")
    if results['old_format']:
        print(f"\n  Old format (need --force): {', '.join(results['old_format'])}")

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Check PH cache status")
    parser.add_argument('--n-samples', type=int, default=2000)
    args = parser.parse_args()

    check_cache_status(args.n_samples)


if __name__ == '__main__':
    main()
