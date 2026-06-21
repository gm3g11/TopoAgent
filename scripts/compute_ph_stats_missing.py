#!/usr/bin/env python
"""
Compute PH statistics (avg H0/H1 bar counts) for the 13 datasets
missing from benchmark3/exp1.

Uses GUDHI CubicalComplex (same as exp1_run.py) with joblib multi-core
parallelization on grayscale 224x224 images, sublevel filtration,
n_samples=5000 (or all available if fewer).

Usage:
    # Single dataset
    python scripts/compute_ph_stats_missing.py --dataset PCam

    # All missing
    python scripts/compute_ph_stats_missing.py --all

    # Specific subset
    python scripts/compute_ph_stats_missing.py --dataset PCam,IDRiD,DermaMNIST
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import gc

import numpy as np
from joblib import Parallel, delayed

from RuleBenchmark.benchmark3.data_loader import load_dataset

# Datasets already in exp1 results
EXISTING = {
    'BloodMNIST', 'MalariaCell', 'TissueMNIST',
    'PathMNIST', 'Kvasir', 'BreakHis', 'NCT_CRC_HE',
    'RetinaMNIST',
    'ISIC2019',
    'OrganAMNIST', 'OCTMNIST', 'BrainTumorMRI', 'MURA',
}

# All 26 datasets
ALL_DATASETS = [
    'AML_Cytomorphology', 'APTOS2019', 'BloodMNIST', 'BrainTumorMRI',
    'BreakHis', 'BreastMNIST', 'Chaoyang', 'DermaMNIST', 'GasHisSDB',
    'IDRiD', 'ISIC2019', 'Kvasir', 'LC25000', 'MURA', 'MalariaCell',
    'NCT_CRC_HE', 'OCTMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST',
    'PCam', 'PathMNIST', 'PneumoniaMNIST', 'RetinaMNIST', 'SIPaKMeD',
    'TissueMNIST',
]

MISSING = sorted(set(ALL_DATASETS) - EXISTING)


def _gudhi_ph_single(img: np.ndarray) -> dict:
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


def compute_ph_stats(dataset_name, n_samples=5000, n_jobs=-1, seed=42):
    """Load dataset, compute PH, return stats dict."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")

    # Load
    t0 = time.time()
    images, labels, class_names = load_dataset(dataset_name, n_samples=n_samples, seed=seed)
    load_time = time.time() - t0
    print(f"  Loaded: {images.shape}, {len(class_names)} classes, {load_time:.1f}s")

    # Compute PH with GUDHI (multi-core via joblib)
    print(f"  Computing PH with GUDHI (n_jobs={n_jobs})...")
    t0 = time.time()
    diagrams = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_gudhi_ph_single)(img) for img in images
    )
    ph_time = time.time() - t0
    print(f"  PH done: {ph_time:.1f}s ({ph_time/len(images)*1000:.1f}ms/image)")

    # Stats
    h0_counts = [len(d.get('H0', [])) for d in diagrams]
    h1_counts = [len(d.get('H1', [])) for d in diagrams]

    avg_h0 = float(np.mean(h0_counts))
    avg_h1 = float(np.mean(h1_counts))
    ratio = avg_h1 / avg_h0 if avg_h0 > 0 else float('inf')

    print(f"  avg_h0_bars: {avg_h0:.4f}")
    print(f"  avg_h1_bars: {avg_h1:.4f}")
    print(f"  H1/H0 ratio: {ratio:.2f}")

    return {
        'metadata': {
            'dataset': dataset_name,
            'n_samples': int(len(images)),
            'n_classes': len(class_names),
            'class_names': class_names,
            'image_shape': list(images.shape),
            'seed': seed,
            'load_time': load_time,
            'ph_library': 'gudhi',
            'ph_time': ph_time,
            'avg_h0_bars': avg_h0,
            'avg_h1_bars': avg_h1,
            'h1_h0_ratio': round(ratio, 4),
            'h0_std': float(np.std(h0_counts)),
            'h1_std': float(np.std(h1_counts)),
            'h0_min': int(np.min(h0_counts)),
            'h0_max': int(np.max(h0_counts)),
            'h1_min': int(np.min(h1_counts)),
            'h1_max': int(np.max(h1_counts)),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PH stats for missing datasets")
    parser.add_argument('--dataset', type=str, default=None,
                        help='Comma-separated dataset names')
    parser.add_argument('--all', action='store_true',
                        help='Run all 13 missing datasets')
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                        default='results/benchmark3/exp1')
    args = parser.parse_args()

    if args.all:
        datasets = MISSING
    elif args.dataset:
        datasets = [d.strip() for d in args.dataset.split(',')]
    else:
        parser.error("Specify --dataset or --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets to process: {datasets}")
    print(f"Output: {output_dir}")

    for ds in datasets:
        # Skip if already exists
        outfile = output_dir / f"{ds}_results.json"
        if outfile.exists():
            print(f"\n  SKIP {ds}: {outfile} already exists")
            continue

        try:
            result = compute_ph_stats(ds, n_samples=args.n_samples,
                                      n_jobs=args.n_jobs, seed=args.seed)

            with open(outfile, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {outfile}")

        except Exception as e:
            print(f"  ERROR {ds}: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()

    print("\nDone.")


if __name__ == '__main__':
    main()
