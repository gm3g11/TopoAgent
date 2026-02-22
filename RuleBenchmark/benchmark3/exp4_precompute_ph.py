#!/usr/bin/env python
"""
Experiment 4: Precompute Persistence Homology for All Datasets

This script computes PH once per dataset and caches to disk.
Always computes BOTH sublevel and superlevel filtrations for combined feature extraction.

PH Library Priority: CuPH (GPU) > Cripser (CPU parallel) > GUDHI (baseline)

Usage:
    python benchmarks/benchmark3/exp4_precompute_ph.py --n-samples 2000
    python benchmarks/benchmark3/exp4_precompute_ph.py --dataset BloodMNIST --n-samples 2000
    python benchmarks/benchmark3/exp4_precompute_ph.py --dataset BloodMNIST --n-jobs 16 --force
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add CuPH to path if available (optional GPU-accelerated PH)
CUPH_PATH = Path(os.environ.get("CUPH_PATH", "/opt/CuPH"))
if CUPH_PATH.exists():
    sys.path.insert(0, str(CUPH_PATH))

import argparse
import pickle
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Dict, List, Tuple, Optional

from RuleBenchmark.benchmark3.exp4_config import (
    OBJECT_TYPE_DATASETS, RESULTS_PATH,
)
from RuleBenchmark.benchmark3.config import DATASETS
from RuleBenchmark.benchmark3.data_loader import load_dataset as _load_dataset_raw


def load_dataset(dataset_name: str, n_samples: int = 2000, seed: int = 42):
    """
    Load dataset with adaptive resize for PH computation.

    Resize policy:
      - If native image size > 1024: resize to 1024
      - Otherwise: keep native size

    MedMNIST: uses pre-resized 224×224 NPZ files.
    External: uses native size (capped at 1024).

    Returns:
        images: (N, H, W) grayscale float32 [0, 1]
        labels: (N,) integer labels
    """
    # Determine target image size based on dataset config
    dataset_info = DATASETS.get(dataset_name, {})
    native_size = dataset_info.get('image_size', 224)

    # Handle 'variable' size - use 1024 as max
    # Only resize if > 2048 (e.g., IDRiD 4288), keep others like APTOS2019 (1050)
    if isinstance(native_size, str):
        target_size = 1024
    elif native_size > 2048:
        target_size = 1024
        print(f"    Image size {native_size} > 2048, resizing to 1024")
    else:
        target_size = native_size

    # Load dataset with target size
    images, labels, class_names = _load_dataset_raw(
        dataset_name,
        n_samples=n_samples,
        seed=seed,
        image_size=target_size
    )

    # Verify image shape
    if images.ndim == 3:
        h, w = images.shape[1], images.shape[2]
        print(f"    Image size {h}×{w}, using as-is")

    print(f"    Loaded {len(images)} images, shape={images.shape}, classes={len(class_names)}")

    return images, labels


# =============================================================================
# CHECK AVAILABLE PH LIBRARIES
# =============================================================================

def check_cuph_available() -> bool:
    """Check if CuPH (GPU) is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import cuph
        return True
    except ImportError:
        return False


def check_cripser_available() -> bool:
    """Check if Cripser is available."""
    try:
        import cripser
        return True
    except ImportError:
        return False


def check_gudhi_available() -> bool:
    """Check if GUDHI is available."""
    try:
        import gudhi
        return True
    except ImportError:
        return False


# Global flags for available libraries
HAS_CUPH = check_cuph_available()
HAS_CRIPSER = check_cripser_available()
HAS_GUDHI = check_gudhi_available()


# =============================================================================
# PH COMPUTATION: CuPH (GPU) - Fastest
# =============================================================================

def compute_ph_cuph(
    images: np.ndarray,
    filtration: str = 'sublevel',
    batch_size: int = 256,
) -> List[Dict]:
    """
    Compute PH using CuPH (GPU) - FASTEST method.

    Performance: ~14,444 img/s on H100

    Args:
        images: (N, H, W) grayscale images [0, 255] or [0, 1]
        filtration: 'sublevel' or 'superlevel'
        batch_size: Batch size for GPU processing

    Returns:
        List of {'H0': [...], 'H1': [...]} dictionaries
    """
    import torch
    import cuph

    n_images = len(images)
    print(f"    Using CuPH (GPU) for {n_images} images...")

    # Ensure images are in [0, 255] range for CuPH
    if images.max() <= 1.0:
        images = (images * 255.0).astype(np.float32)

    # Convert to torch tensor on GPU
    imgs_tensor = torch.from_numpy(images.astype(np.float32)).cuda()

    diagrams = []

    for start_idx in range(0, n_images, batch_size):
        end_idx = min(start_idx + batch_size, n_images)
        batch = imgs_tensor[start_idx:end_idx]

        if filtration == 'superlevel':
            batch = 255.0 - batch

        # Compute H0 and H1
        h0_pairs, h0_lengths = cuph.compute_h0(batch)
        h1_pairs, h1_lengths = cuph.compute_h1(batch)

        # Convert to list of dicts
        for i in range(len(batch)):
            h0_len = h0_lengths[i].item()
            h1_len = h1_lengths[i].item()

            h0_list = h0_pairs[i, :h0_len].cpu().numpy().tolist()
            h1_list = h1_pairs[i, :h1_len].cpu().numpy().tolist()

            # Convert to dict format
            if filtration == 'superlevel':
                h0_list = [{'birth': float(255.0 - d), 'death': float(255.0 - b)}
                          for b, d in h0_list if b < d]
                h1_list = [{'birth': float(255.0 - d), 'death': float(255.0 - b)}
                          for b, d in h1_list if b < d]
            else:
                h0_list = [{'birth': float(b), 'death': float(d)}
                          for b, d in h0_list if b < d]
                h1_list = [{'birth': float(b), 'death': float(d)}
                          for b, d in h1_list if b < d]

            diagrams.append({'H0': h0_list, 'H1': h1_list})

        if (end_idx % 1000 == 0) or (end_idx == n_images):
            print(f"      CuPH: {end_idx}/{n_images}")

    # Clear GPU memory
    del imgs_tensor
    torch.cuda.empty_cache()

    return diagrams


# =============================================================================
# PH COMPUTATION: Cripser (CPU Parallel) - Fast
# =============================================================================

def compute_ph_cripser_single(
    img: np.ndarray,
    filtration: str = 'sublevel',
) -> Dict:
    """Compute PH for single image using Cripser."""
    import cripser

    # Ensure image is in proper range
    if img.max() <= 1.0:
        img = img * 255.0

    if filtration == 'superlevel':
        img = 255.0 - img

    result = cripser.computePH(img.astype(np.float64), maxdim=1)

    h0_pairs = []
    h1_pairs = []

    # Cripser returns: [dim, birth, death, ...] per row
    for row in result:
        dim = int(row[0])
        birth = float(row[1])
        death = float(row[2])

        # Skip infinite/essential features
        if not np.isfinite(birth) or not np.isfinite(death):
            continue
        if abs(death) > 1e6 or abs(birth) > 1e6:
            continue
        if death <= birth:
            continue

        if filtration == 'superlevel':
            birth, death = 255.0 - death, 255.0 - birth

        if dim == 0:
            h0_pairs.append({'birth': birth, 'death': death})
        elif dim == 1:
            h1_pairs.append({'birth': birth, 'death': death})

    return {'H0': h0_pairs, 'H1': h1_pairs}


def compute_ph_cripser_parallel(
    images: np.ndarray,
    filtration: str = 'sublevel',
    n_jobs: int = 8,
) -> List[Dict]:
    """
    Compute PH using Cripser with joblib parallelization.

    Performance: ~306 img/s with 16 cores

    Args:
        images: (N, H, W) grayscale images
        filtration: 'sublevel' or 'superlevel'
        n_jobs: Number of parallel jobs

    Returns:
        List of {'H0': [...], 'H1': [...]} dictionaries
    """
    from joblib import Parallel, delayed

    n_images = len(images)
    print(f"    Using Cripser (CPU, {n_jobs} cores) for {n_images} images...")

    diagrams = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_ph_cripser_single)(img, filtration) for img in images
    )

    print(f"      Cripser: completed {n_images} images")
    return diagrams


# =============================================================================
# PH COMPUTATION: GUDHI (Baseline)
# =============================================================================

def compute_ph_gudhi_single(
    img: np.ndarray,
    filtration: str = 'sublevel',
) -> Dict:
    """Compute PH for single image using GUDHI."""
    import gudhi

    # Normalize to [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    if filtration == 'superlevel':
        img = 1.0 - img

    cc = gudhi.CubicalComplex(top_dimensional_cells=img)
    cc.persistence()

    pd = {}
    for dim in [0, 1]:
        intervals = cc.persistence_intervals_in_dimension(dim)
        pairs = []
        for b, d in intervals:
            if d == float('inf'):
                d = img.max()
            if d > b:
                pairs.append({'birth': float(b), 'death': float(d)})
        pd[f'H{dim}'] = pairs

    return pd


def compute_ph_gudhi_parallel(
    images: np.ndarray,
    filtration: str = 'sublevel',
    n_jobs: int = 8,
) -> List[Dict]:
    """
    Compute PH using GUDHI with joblib parallelization.

    Performance: ~96 img/s with 16 cores (baseline)

    Args:
        images: (N, H, W) grayscale images
        filtration: 'sublevel' or 'superlevel'
        n_jobs: Number of parallel jobs

    Returns:
        List of {'H0': [...], 'H1': [...]} dictionaries
    """
    from joblib import Parallel, delayed

    n_images = len(images)
    print(f"    Using GUDHI (CPU, {n_jobs} cores) for {n_images} images...")

    diagrams = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_ph_gudhi_single)(img, filtration) for img in images
    )

    print(f"      GUDHI: completed {n_images} images")
    return diagrams


# =============================================================================
# UNIFIED PH COMPUTATION (Auto-select best library)
# =============================================================================

def compute_ph_fast(
    images: np.ndarray,
    filtration: str = 'sublevel',
    n_jobs: int = 8,
) -> Tuple[List[Dict], str]:
    """
    Compute PH using best available method.

    Priority: CuPH (GPU) > Cripser (CPU parallel) > GUDHI (baseline)

    Args:
        images: (N, H, W) grayscale images
        filtration: 'sublevel' or 'superlevel'
        n_jobs: Number of parallel jobs for CPU methods

    Returns:
        (diagrams, library_used)
    """
    # Try CuPH first (GPU)
    if HAS_CUPH:
        try:
            diagrams = compute_ph_cuph(images, filtration)
            return diagrams, 'cuph'
        except Exception as e:
            print(f"    WARNING: CuPH failed ({e}), falling back to Cripser")

    # Try Cripser (CPU parallel)
    if HAS_CRIPSER:
        try:
            diagrams = compute_ph_cripser_parallel(images, filtration, n_jobs)
            return diagrams, 'cripser'
        except Exception as e:
            print(f"    WARNING: Cripser failed ({e}), falling back to GUDHI")

    # Fallback to GUDHI
    if HAS_GUDHI:
        diagrams = compute_ph_gudhi_parallel(images, filtration, n_jobs)
        return diagrams, 'gudhi'

    raise RuntimeError("No PH library available! Install gudhi, cripser, or cuph.")


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

# Cache directory
PH_CACHE_PATH = RESULTS_PATH / "ph_cache"
PH_CACHE_PATH.mkdir(parents=True, exist_ok=True)


def get_cache_path(dataset: str, n_samples: int) -> Path:
    """Get cache file path for a dataset."""
    return PH_CACHE_PATH / f"{dataset}_n{n_samples}_combined.pkl"


def find_existing_cache(dataset: str) -> Optional[Path]:
    """Find any existing cache file for a dataset."""
    pattern = f"{dataset}_n*_combined.pkl"
    matches = list(PH_CACHE_PATH.glob(pattern))
    if matches:
        # Return the one with highest n_samples
        return max(matches, key=lambda p: int(p.stem.split('_n')[1].split('_')[0]))
    return None


def precompute_ph_for_dataset(
    dataset: str,
    n_samples: int = 2000,
    seed: int = 42,
    n_jobs: int = 8,
    force: bool = False,
) -> Dict:
    """
    Precompute PH for a dataset and save to cache.

    Always computes BOTH sublevel and superlevel filtrations for combined feature extraction.
    For small datasets (e.g., RetinaMNIST with 1600 samples), uses max available.

    Cache structure:
    {
        'dataset': str,
        'n_samples': int,          # Actual loaded count
        'n_samples_requested': int, # Originally requested count
        'labels': np.ndarray,
        'diagrams_sublevel': List[Dict],   # For sublevel features
        'diagrams_superlevel': List[Dict], # For superlevel features
        # Combined = concat(extract(sublevel), extract(superlevel)) at extraction time
        'image_shape': tuple,
        'seed': int,
        'ph_library': str,
        'ph_time_sublevel': float,
        'ph_time_superlevel': float,
    }
    """
    # Check if already cached (with any sample count >= requested or max available)
    if not force:
        existing = find_existing_cache(dataset)
        if existing:
            with open(existing, 'rb') as f:
                cached_data = pickle.load(f)
            cached_n = cached_data.get('n_samples', 0)
            # Accept if we have enough samples or it's the max available for this dataset
            if cached_n >= n_samples or cached_n == cached_data.get('n_samples_requested', n_samples):
                print(f"  {dataset}: Already cached at {existing} (n={cached_n})")
                return {'status': 'cached', 'path': str(existing), 'n_samples': cached_n}

    print(f"\n{'='*60}")
    print(f"  Computing PH for: {dataset}")
    print(f"  Requested samples: {n_samples} (will use max available if smaller)")
    print(f"  Filtrations: sublevel + superlevel (for combined)")
    print(f"{'='*60}")

    start_total = time.time()

    # Load dataset (will return min(requested, available) samples)
    try:
        images, labels = load_dataset(dataset, n_samples=n_samples, seed=seed)
        actual_n = len(images)
        if actual_n < n_samples:
            print(f"  NOTE: Dataset has only {actual_n} samples (requested {n_samples})")
        print(f"  Loaded {actual_n} images, shape={images.shape}")
    except Exception as e:
        print(f"  ERROR loading {dataset}: {e}")
        return {'status': 'error', 'error': str(e)}

    # Compute PH for SUBLEVEL filtration
    print(f"\n  [1/2] Computing SUBLEVEL filtration...")
    start_sub = time.time()
    diagrams_sublevel, library_used = compute_ph_fast(images, 'sublevel', n_jobs)
    time_sublevel = time.time() - start_sub
    print(f"    Sublevel time: {time_sublevel:.1f}s ({len(images)/time_sublevel:.1f} img/s)")

    # Compute PH for SUPERLEVEL filtration
    print(f"\n  [2/2] Computing SUPERLEVEL filtration...")
    start_sup = time.time()
    diagrams_superlevel, _ = compute_ph_fast(images, 'superlevel', n_jobs)
    time_superlevel = time.time() - start_sup
    print(f"    Superlevel time: {time_superlevel:.1f}s ({len(images)/time_superlevel:.1f} img/s)")

    # Save to cache (use actual loaded count in filename)
    actual_n = len(images)
    cache_path = get_cache_path(dataset, actual_n)

    cache_data = {
        'dataset': dataset,
        'n_samples': actual_n,
        'n_samples_requested': n_samples,
        'labels': labels,
        'diagrams_sublevel': diagrams_sublevel,
        'diagrams_superlevel': diagrams_superlevel,
        'image_shape': images.shape[1:],
        'seed': seed,
        'ph_library': library_used,
        'ph_time_sublevel': time_sublevel,
        'ph_time_superlevel': time_superlevel,
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    total_time = time.time() - start_total

    # Memory cleanup
    del images, diagrams_sublevel, diagrams_superlevel
    gc.collect()

    print(f"\n  Saved to: {cache_path}")
    print(f"  Library used: {library_used}")
    print(f"  Total PH time: {time_sublevel + time_superlevel:.1f}s")
    print(f"  Total time: {total_time:.1f}s")

    return {
        'status': 'computed',
        'path': str(cache_path),
        'n_samples': actual_n,
        'n_samples_requested': n_samples,
        'ph_library': library_used,
        'ph_time_sublevel': time_sublevel,
        'ph_time_superlevel': time_superlevel,
        'total_time': total_time,
    }


def load_cached_ph(dataset: str, n_samples: int = 2000) -> Dict:
    """
    Load PH from cache.

    Returns cache data with both filtrations:
    - diagrams_sublevel: for sublevel features
    - diagrams_superlevel: for superlevel features
    - For combined: concatenate features from both at extraction time
    """
    cache_path = get_cache_path(dataset, n_samples)

    if not cache_path.exists():
        # Try old cache format (single filtration)
        old_cache_path = PH_CACHE_PATH / f"{dataset}_n{n_samples}.pkl"
        if old_cache_path.exists():
            print(f"  WARNING: Found old cache format at {old_cache_path}")
            print(f"  Please re-run precomputation with --force for combined filtration support")
            with open(old_cache_path, 'rb') as f:
                data = pickle.load(f)
            # Convert old format to new format (sublevel only)
            if 'diagrams' in data and 'diagrams_sublevel' not in data:
                data['diagrams_sublevel'] = data['diagrams']
                data['diagrams_superlevel'] = None
            return data
        raise FileNotFoundError(f"No cache found: {cache_path}")

    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    return data


def main():
    parser = argparse.ArgumentParser(description="Precompute PH for all datasets (combined filtration)")
    parser.add_argument('--dataset', type=str, default=None,
                       help='Single dataset to compute')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Number of samples per dataset')
    parser.add_argument('--n-jobs', type=int, default=8,
                       help='Number of parallel jobs for CPU PH computation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                       help='Force recomputation even if cached')

    args = parser.parse_args()

    print("=" * 70)
    print("  EXPERIMENT 4: PRECOMPUTE PERSISTENCE HOMOLOGY (COMBINED)")
    print("=" * 70)
    print(f"  n_samples: {args.n_samples}")
    print(f"  n_jobs: {args.n_jobs}")
    print(f"  seed: {args.seed}")
    print(f"  cache_path: {PH_CACHE_PATH}")
    print(f"\n  Available PH libraries:")
    print(f"    CuPH (GPU):  {'YES' if HAS_CUPH else 'NO'}")
    print(f"    Cripser:     {'YES' if HAS_CRIPSER else 'NO'}")
    print(f"    GUDHI:       {'YES' if HAS_GUDHI else 'NO'}")
    print("=" * 70)

    # Determine datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        # All datasets from all object types
        datasets = []
        for obj_type, ds_list in OBJECT_TYPE_DATASETS.items():
            datasets.extend(ds_list)
        # Remove duplicates while preserving order
        datasets = list(dict.fromkeys(datasets))

    print(f"\nDatasets to process: {len(datasets)}")
    for ds in datasets:
        print(f"  - {ds}")

    # Process each dataset
    results = {}
    total_start = time.time()

    for dataset in datasets:
        result = precompute_ph_for_dataset(
            dataset=dataset,
            n_samples=args.n_samples,
            seed=args.seed,
            n_jobs=args.n_jobs,
            force=args.force,
        )
        results[dataset] = result

    total_time = time.time() - total_start

    # Summary
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    computed = sum(1 for r in results.values() if r['status'] == 'computed')
    cached = sum(1 for r in results.values() if r['status'] == 'cached')
    errors = sum(1 for r in results.values() if r['status'] == 'error')

    print(f"  Computed: {computed}")
    print(f"  Cached: {cached}")
    print(f"  Errors: {errors}")
    print(f"  Total time: {total_time:.1f}s")

    # Show timing breakdown for computed datasets
    if computed > 0:
        print(f"\n  Timing breakdown:")
        for ds, r in results.items():
            if r['status'] == 'computed':
                lib = r.get('ph_library', 'unknown')
                t_sub = r.get('ph_time_sublevel', 0)
                t_sup = r.get('ph_time_superlevel', 0)
                print(f"    {ds}: {t_sub + t_sup:.1f}s (sub={t_sub:.1f}s, sup={t_sup:.1f}s) [{lib}]")

    if errors > 0:
        print(f"\n  Errors:")
        for ds, r in results.items():
            if r['status'] == 'error':
                print(f"    {ds}: {r['error']}")


if __name__ == '__main__':
    main()
