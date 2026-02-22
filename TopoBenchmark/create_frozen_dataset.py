"""
Create Frozen TopoBenchmark Dataset.

After convergence analysis completes, this script:
1. Reads convergence results for all datasets
2. Determines frozen n per dataset (convergence_n or n_max for small datasets)
3. Creates frozen_dataset_config.json (canonical specification)
4. Precomputes PH caches at frozen n per dataset
5. Stores deterministic fold indices for reproducibility

Usage:
    python TopoBenchmark/create_frozen_dataset.py
    python TopoBenchmark/create_frozen_dataset.py --min-n 200   # override minimum
    python TopoBenchmark/create_frozen_dataset.py --dry-run     # just show config
"""

import sys
import json
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from RuleBenchmark.benchmark4.data_loader import load_dataset, rgb_to_channels
from RuleBenchmark.benchmark4.config import DATASETS, PH_CACHE_PATH

# Paths
CONVERGENCE_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "convergence"
FROZEN_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "frozen_ph"
CONFIG_PATH = Path(__file__).parent / "frozen_dataset_config.json"


def load_convergence_results() -> Dict[str, Dict]:
    """Load convergence results for all datasets.

    Prefers extension files (e.g. _convergence_n10000.json) over base files
    since they have more data points and higher n_max.
    """
    results = {}
    # First load all base convergence files
    for f in sorted(CONVERGENCE_DIR.glob("*_convergence.json")):
        if '_convergence_n' in f.name:
            continue  # skip extensions in first pass
        with open(f) as fp:
            data = json.load(fp)
        dataset = data['dataset']
        results[dataset] = data

    # Then override with extension files (higher n_max)
    for f in sorted(CONVERGENCE_DIR.glob("*_convergence_n*.json")):
        with open(f) as fp:
            data = json.load(fp)
        dataset = data['dataset']
        results[dataset] = data  # override base with extension

    return results


def determine_frozen_n(
    convergence_results: Dict[str, Dict],
    min_n: int = 200,
    default_n: int = 500,
) -> Dict[str, Dict[str, Any]]:
    """Determine frozen n per dataset from convergence analysis.

    Rules:
    - Use convergence_n if available and >= min_n
    - Use n_max if dataset is too small for convergence_n
    - Use default_n if no convergence data available
    - Never exceed n_max
    """
    frozen = {}

    for dataset, cfg in DATASETS.items():
        config_n_max = cfg['n_samples']
        n_classes = cfg['n_classes']

        conv_data = convergence_results.get(dataset)
        if conv_data is not None:
            # Use the convergence data's own n_max (may be higher than config
            # for extension runs, e.g. n10000 extensions)
            data_n_max = conv_data.get('n_max', config_n_max)
            conv = conv_data.get('convergence', {})
            conv_n = conv.get('convergence_n')

            if conv_n is not None:
                n = max(conv_n, min_n)
            else:
                # Convergence not found (partial results) — use data's n_max
                # (i.e., use all available data since it hasn't converged)
                n = data_n_max
        else:
            # No convergence data — use default
            n = min(default_n, config_n_max)

        # Never exceed the convergence data's n_max (or config n_max if no data)
        effective_n_max = conv_data.get('n_max', config_n_max) if conv_data else config_n_max
        n = min(n, effective_n_max)

        frozen[dataset] = {
            'n': n,
            'n_max': config_n_max,
            'data_n_max': effective_n_max,
            'n_classes': n_classes,
            'convergence_n': conv_data.get('convergence', {}).get('convergence_n') if conv_data else None,
            'has_convergence_data': conv_data is not None,
        }

    return frozen


def create_fold_indices(
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[int]]]:
    """Create deterministic fold indices for reproducibility."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(labels, labels):
        folds.append({
            'train': train_idx.tolist(),
            'test': test_idx.tolist(),
        })
    return folds


def precompute_frozen_ph(
    dataset: str,
    n: int,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Precompute PH at frozen n for a dataset.

    First checks if existing benchmark4 cache at n_max can be subsampled.
    Otherwise computes from scratch.
    """
    output_path = FROZEN_DIR / f"{dataset}_n{n}.pkl"

    if output_path.exists() and not force:
        print(f"  {dataset}: frozen PH already exists at {output_path}")
        return output_path

    cfg = DATASETS[dataset]
    config_n_max = cfg['n_samples']
    color_mode = cfg['color_mode']

    # Try to subsample from existing cache.
    # First try config n_max, then check for extended caches if n > config_n_max.
    existing_cache = PH_CACHE_PATH / f"{dataset}_n{config_n_max}.pkl"
    n_max = config_n_max

    if not existing_cache.exists() or n > config_n_max:
        # Look for extended PH caches (from extension convergence runs)
        for ext_n in [10000, 5712]:
            ext_cache = PH_CACHE_PATH / f"{dataset}_n{ext_n}.pkl"
            if ext_cache.exists() and ext_n >= n:
                existing_cache = ext_cache
                n_max = ext_n
                break

    if existing_cache.exists() and n <= n_max:
        print(f"  {dataset}: subsampling from benchmark4 cache (n_max={n_max} -> n={n})")
        with open(existing_cache, 'rb') as f:
            cache = pickle.load(f)

        # Use labels from the cache itself (avoids loading the full image dataset)
        labels_full = np.array(cache['labels'])
        class_names = cache.get('class_names', [str(i) for i in range(cfg['n_classes'])])

        # Cap to actual cache size (may differ from reported n_max)
        actual_size = len(labels_full)
        if n > actual_size:
            print(f"  {dataset}: WARNING: requested n={n} > actual cache size {actual_size}, capping")
            n = actual_size
            # Update output path for actual n
            output_path = FROZEN_DIR / f"{dataset}_n{n}.pkl"
            if output_path.exists() and not force:
                print(f"  {dataset}: frozen PH already exists at {output_path}")
                return output_path

        if n >= actual_size:
            # Use everything
            indices = np.arange(actual_size)
        else:
            # Stratified subsample
            rng = np.random.RandomState(seed)
            classes = np.unique(labels_full)
            class_props = {c: np.sum(labels_full == c) / len(labels_full) for c in classes}

            indices = []
            for cls, prop in class_props.items():
                cls_idx = np.where(labels_full == cls)[0]
                n_cls = max(1, int(round(prop * n)))
                if n_cls > len(cls_idx):
                    n_cls = len(cls_idx)
                chosen = rng.choice(cls_idx, n_cls, replace=False)
                indices.extend(chosen)
            indices = np.array(sorted(indices))

            # Adjust count
            if len(indices) > n:
                indices = rng.choice(indices, n, replace=False)
                indices = np.sort(indices)
            elif len(indices) < n:
                remaining = np.setdiff1d(np.arange(actual_size), indices)
                extra = rng.choice(remaining, n - len(indices), replace=False)
                indices = np.sort(np.concatenate([indices, extra]))

        # Subsample all arrays
        frozen_cache = {
            'dataset': dataset,
            'n_samples': len(indices),
            'labels': [cache['labels'][i] for i in indices],
            'class_names': cache.get('class_names', class_names),
            'color_mode': color_mode,
            'seed': seed,
            'source': 'subsampled_from_benchmark4',
            'original_n_max': n_max,
        }

        diags_gray = cache.get('diagrams_gray_sublevel')
        if diags_gray is not None:
            frozen_cache['diagrams_gray_sublevel'] = [diags_gray[i] for i in indices]

        if color_mode == 'per_channel':
            for ch in ['R', 'G', 'B']:
                key = f'diagrams_{ch}_sublevel'
                ch_diags = cache.get(key)
                if ch_diags is not None:
                    frozen_cache[key] = [ch_diags[i] for i in indices]

        # Save fold indices
        labels_arr = np.array(frozen_cache['labels'])
        frozen_cache['fold_indices'] = create_fold_indices(labels_arr)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(frozen_cache, f)

        print(f"  {dataset}: saved frozen PH ({len(indices)} samples) to {output_path}")
        return output_path

    else:
        # Compute from scratch
        print(f"  {dataset}: computing PH from scratch (n={n})")
        from RuleBenchmark.benchmark4.precompute_ph import compute_ph

        images, labels, class_names = load_dataset(dataset, n_samples=n, seed=seed)
        imgs_gray = images.astype(np.float32) / 255.0
        if imgs_gray.ndim == 4:
            imgs_gray = np.mean(imgs_gray, axis=-1)

        diagrams_gray, lib = compute_ph(imgs_gray)

        frozen_cache = {
            'dataset': dataset,
            'n_samples': len(labels),
            'labels': labels.tolist(),
            'class_names': class_names,
            'color_mode': color_mode,
            'seed': seed,
            'source': 'computed_fresh',
            'ph_library': lib,
            'diagrams_gray_sublevel': diagrams_gray,
        }

        if color_mode == 'per_channel' and images.ndim == 4:
            for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
                ch_imgs = images[:, :, :, ch_idx].astype(np.float32) / 255.0
                ch_diags, _ = compute_ph(ch_imgs)
                frozen_cache[f'diagrams_{ch_name}_sublevel'] = ch_diags

        labels_arr = np.array(labels)
        frozen_cache['fold_indices'] = create_fold_indices(labels_arr)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(frozen_cache, f)

        print(f"  {dataset}: saved frozen PH ({len(labels)} samples) to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Create frozen TopoBenchmark dataset')
    parser.add_argument('--min-n', type=int, default=200,
                        help='Minimum frozen n (default: 200)')
    parser.add_argument('--default-n', type=int, default=500,
                        help='Default n when no convergence data (default: 500)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show config, do not precompute PH')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated list of datasets (default: all)')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if frozen PH exists')
    args = parser.parse_args()

    print("=" * 60)
    print("  Create Frozen TopoBenchmark Dataset")
    print("=" * 60)

    # Load convergence results
    conv_results = load_convergence_results()
    print(f"\n  Convergence results found for {len(conv_results)}/{len(DATASETS)} datasets")
    if conv_results:
        print(f"  Datasets: {sorted(conv_results.keys())}")

    missing = set(DATASETS.keys()) - set(conv_results.keys())
    if missing:
        print(f"  Missing convergence data: {sorted(missing)}")

    # Determine frozen n
    frozen = determine_frozen_n(conv_results, min_n=args.min_n, default_n=args.default_n)

    # Filter datasets if specified
    if args.datasets:
        target_datasets = [d.strip() for d in args.datasets.split(',')]
        frozen = {k: v for k, v in frozen.items() if k in target_datasets}

    # Display config
    print(f"\n  Frozen Dataset Configuration:")
    print(f"  {'Dataset':<25} {'n':>6} {'cfg_n':>6} {'data_n':>6} {'conv_n':>7} {'cls':>4} {'data?':>6}")
    print(f"  {'-'*68}")
    total_samples = 0
    for dataset in sorted(frozen.keys()):
        info = frozen[dataset]
        conv_str = str(info['convergence_n']) if info['convergence_n'] else 'N/A'
        data_str = 'yes' if info['has_convergence_data'] else 'no'
        data_n_str = str(info['data_n_max']) if info['data_n_max'] != info['n_max'] else ''
        print(f"  {dataset:<25} {info['n']:>6} {info['n_max']:>6} {data_n_str:>6} {conv_str:>7} "
              f"{info['n_classes']:>4} {data_str:>6}")
        total_samples += info['n']
    print(f"  {'-'*68}")
    print(f"  Total: {total_samples} samples across {len(frozen)} datasets")

    # Create config JSON
    config = {
        'version': '1.0',
        'created': datetime.now().strftime('%Y-%m-%d'),
        'seed': 42,
        'n_folds': 5,
        'min_n': args.min_n,
        'default_n': args.default_n,
        'n_datasets': len(frozen),
        'total_samples': total_samples,
        'datasets': {k: {'n': v['n'], 'n_max': v['n_max'],
                         'data_n_max': v['data_n_max'],
                         'convergence_n': v['convergence_n'],
                         'n_classes': v['n_classes']}
                     for k, v in sorted(frozen.items())},
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved to {CONFIG_PATH}")

    if args.dry_run:
        print("\n  [DRY RUN] Skipping PH precomputation.")
        return

    # Precompute frozen PH
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Precomputing frozen PH caches...")
    t0 = time.time()

    for dataset in sorted(frozen.keys()):
        n = frozen[dataset]['n']
        try:
            precompute_frozen_ph(dataset, n, seed=42, force=args.force)
        except Exception as e:
            print(f"  ERROR on {dataset}: {e}")

    print(f"\n  Total PH precomputation: {time.time()-t0:.1f}s")
    print(f"  Frozen PH caches: {FROZEN_DIR}")
    print("  Done!")


if __name__ == '__main__':
    main()
