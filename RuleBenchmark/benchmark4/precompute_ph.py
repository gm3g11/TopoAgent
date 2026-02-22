#!/usr/bin/env python
"""
Benchmark4: Precompute Persistent Homology for 26 Datasets.

Supports per-channel RGB (R, G, B computed separately) and grayscale.
PH Library Priority: CuPH (GPU) > Cripser (CPU parallel).

Storage format: numpy arrays for compactness.
Each diagram: {'H0': ndarray(n,2), 'H1': ndarray(m,2)} with columns [birth, death].

Cache format per dataset:
  {dataset}_n{n_samples}.pkl = {
      'dataset', 'n_samples', 'labels', 'class_names',
      'image_shape', 'color_mode', 'seed',
      'diagrams_gray_sublevel': List[Dict],   # always present
      'diagrams_R_sublevel': List[Dict],      # RGB only
      'diagrams_G_sublevel': List[Dict],      # RGB only
      'diagrams_B_sublevel': List[Dict],      # RGB only
      'ph_library', 'ph_time_seconds',
  }

Usage:
    python benchmarks/benchmark4/precompute_ph.py --dataset BloodMNIST --n-samples 100
    python benchmarks/benchmark4/precompute_ph.py --all
    python benchmarks/benchmark4/precompute_ph.py --dataset BloodMNIST --compare --n-samples 50
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CuPH path (optional GPU-accelerated PH)
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

from RuleBenchmark.benchmark4.config import (
    DATASETS, ALL_DATASETS, PH_CACHE_PATH, PH_CONFIG,
)
from RuleBenchmark.benchmark4.data_loader import (
    load_dataset, rgb_to_channels, _to_grayscale_float,
    prepare_batched_loader,
)


# =============================================================================
# PH Library Detection
# =============================================================================

def check_cuph():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import cuph
        return True
    except (ImportError, OSError):
        return False

def check_cripser():
    try:
        import cripser
        return True
    except ImportError:
        return False

HAS_CUPH = check_cuph()
HAS_CRIPSER = check_cripser()


# =============================================================================
# CuPH (GPU)
# =============================================================================

def compute_ph_cuph(images: np.ndarray, batch_size: int = 256) -> List[Dict]:
    """Compute sublevel PH using CuPH (GPU).

    Args:
        images: (N, H, W) float32 [0, 1] or [0, 255]
    Returns:
        List of {'H0': ndarray(n,2), 'H1': ndarray(m,2)}
    """
    import torch
    import cuph

    if images.max() <= 1.0:
        images = (images * 255.0).astype(np.float32)

    imgs_tensor = torch.from_numpy(images.astype(np.float32)).cuda()
    diagrams = []

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = imgs_tensor[start:end]

        h0_pairs, h0_lengths = cuph.compute_h0(batch)
        h1_pairs, h1_lengths = cuph.compute_h1(batch)

        for i in range(len(batch)):
            h0_len = h0_lengths[i].item()
            h1_len = h1_lengths[i].item()

            h0_raw = h0_pairs[i, :h0_len].cpu().numpy()
            h1_raw = h1_pairs[i, :h1_len].cpu().numpy()

            # Filter: keep only valid pairs where birth < death
            h0 = h0_raw[h0_raw[:, 0] < h0_raw[:, 1]] if len(h0_raw) > 0 else np.empty((0, 2), dtype=np.float32)
            h1 = h1_raw[h1_raw[:, 0] < h1_raw[:, 1]] if len(h1_raw) > 0 else np.empty((0, 2), dtype=np.float32)

            diagrams.append({'H0': h0.astype(np.float32), 'H1': h1.astype(np.float32)})

    del imgs_tensor
    torch.cuda.empty_cache()
    return diagrams


# =============================================================================
# Cripser (CPU parallel)
# =============================================================================

def _cripser_single(img: np.ndarray) -> Dict:
    """Single-image PH via Cripser. Returns numpy arrays."""
    import cripser as cr

    if img.max() <= 1.0:
        img = img * 255.0

    result = cr.computePH(img.astype(np.float64), maxdim=1)

    h0_list, h1_list = [], []
    for row in result:
        dim, b, d = int(row[0]), float(row[1]), float(row[2])
        if not np.isfinite(b) or not np.isfinite(d):
            continue
        if abs(d) > 1e6 or abs(b) > 1e6 or d <= b:
            continue
        if dim == 0:
            h0_list.append([b, d])
        elif dim == 1:
            h1_list.append([b, d])

    h0 = np.array(h0_list, dtype=np.float32) if h0_list else np.empty((0, 2), dtype=np.float32)
    h1 = np.array(h1_list, dtype=np.float32) if h1_list else np.empty((0, 2), dtype=np.float32)
    return {'H0': h0, 'H1': h1}


def compute_ph_cripser(images: np.ndarray, n_jobs: int = 16) -> List[Dict]:
    """Compute sublevel PH using Cripser with joblib parallelism."""
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_cripser_single)(img) for img in images
    )


# =============================================================================
# Unified PH Computation
# =============================================================================

def compute_ph(images: np.ndarray, n_jobs: int = 16) -> Tuple[List[Dict], str]:
    """Auto-select best PH library. Returns (diagrams, library_name)."""
    if HAS_CUPH:
        try:
            return compute_ph_cuph(images, PH_CONFIG['batch_size_cuph']), 'cuph'
        except Exception as e:
            print(f"  WARNING: CuPH failed ({e}), trying Cripser")
    if HAS_CRIPSER:
        try:
            return compute_ph_cripser(images, n_jobs), 'cripser'
        except Exception as e:
            print(f"  WARNING: Cripser failed ({e})")
    raise RuntimeError("No PH library available! Need CuPH or Cripser.")


# =============================================================================
# Format Conversion Utilities
# =============================================================================

def diagrams_to_dict_list(diagrams: List[Dict]) -> List[Dict]:
    """Convert numpy-format diagrams to list-of-dicts format.

    Input:  [{'H0': ndarray(n,2), 'H1': ndarray(m,2)}, ...]
    Output: [{'H0': [{'birth': b, 'death': d}, ...], 'H1': [...]}, ...]

    Use this when passing to descriptor tools that expect the old format.
    """
    result = []
    for diag in diagrams:
        d = {}
        for key in ['H0', 'H1']:
            arr = diag.get(key)
            if arr is not None and len(arr) > 0:
                d[key] = [{'birth': float(b), 'death': float(d_)}
                          for b, d_ in arr]
            else:
                d[key] = []
        result.append(d)
    return result


# =============================================================================
# CuPH vs Cripser Comparison
# =============================================================================

def compare_cuph_cripser(images: np.ndarray, dataset_name: str = ""):
    """Compare CuPH and Cripser results on a batch of images."""
    if not HAS_CUPH or not HAS_CRIPSER:
        print("  Need both CuPH and Cripser for comparison")
        return

    n = len(images)
    print(f"\n  === CuPH vs Cripser Comparison ({dataset_name}, n={n}) ===")

    # Cripser
    t0 = time.time()
    diags_cr = compute_ph_cripser(images, n_jobs=16)
    t_cr = time.time() - t0
    print(f"  Cripser: {t_cr:.2f}s ({n/t_cr:.0f} img/s)")

    # CuPH
    t0 = time.time()
    diags_cu = compute_ph_cuph(images)
    t_cu = time.time() - t0
    print(f"  CuPH:    {t_cu:.2f}s ({n/t_cu:.0f} img/s)")
    print(f"  Speedup: {t_cr/t_cu:.1f}x")

    # Compare pair counts and values
    h0_count_match, h1_count_match = 0, 0
    h0_val_diffs, h1_val_diffs = [], []

    for i in range(n):
        cr_h0 = np.sort(diags_cr[i]['H0'], axis=0) if len(diags_cr[i]['H0']) > 0 else np.empty((0, 2))
        cu_h0 = np.sort(diags_cu[i]['H0'], axis=0) if len(diags_cu[i]['H0']) > 0 else np.empty((0, 2))
        cr_h1 = np.sort(diags_cr[i]['H1'], axis=0) if len(diags_cr[i]['H1']) > 0 else np.empty((0, 2))
        cu_h1 = np.sort(diags_cu[i]['H1'], axis=0) if len(diags_cu[i]['H1']) > 0 else np.empty((0, 2))

        if len(cr_h0) == len(cu_h0):
            h0_count_match += 1
            if len(cr_h0) > 0:
                diffs = np.abs(cr_h0 - cu_h0).max(axis=1)
                h0_val_diffs.extend(diffs.tolist())
        if len(cr_h1) == len(cu_h1):
            h1_count_match += 1
            if len(cr_h1) > 0:
                diffs = np.abs(cr_h1 - cu_h1).max(axis=1)
                h1_val_diffs.extend(diffs.tolist())

    print(f"  H0 pair count match: {h0_count_match}/{n} ({100*h0_count_match/n:.1f}%)")
    print(f"  H1 pair count match: {h1_count_match}/{n} ({100*h1_count_match/n:.1f}%)")
    if h0_val_diffs:
        arr = np.array(h0_val_diffs)
        print(f"  H0 value diff: mean={arr.mean():.4f}, max={arr.max():.4f}, "
              f"exact_match={100*(arr == 0).mean():.1f}%")
    if h1_val_diffs:
        arr = np.array(h1_val_diffs)
        print(f"  H1 value diff: mean={arr.mean():.4f}, max={arr.max():.4f}, "
              f"exact_match={100*(arr == 0).mean():.1f}%")


# =============================================================================
# Main Precompute Logic
# =============================================================================

def get_cache_path(dataset: str, n_samples: int) -> Path:
    return PH_CACHE_PATH / f"{dataset}_n{n_samples}.pkl"


def precompute_one(
    dataset: str,
    n_samples: Optional[int] = None,
    seed: int = 42,
    n_jobs: int = 16,
    force: bool = False,
    compare: bool = False,
    batch_size: int = 500,
) -> Dict:
    """Precompute PH for one dataset (grayscale + per-channel RGB).

    Uses batched image loading to avoid OOM on large-image datasets.
    Peak memory: batch_size × H × W × channels × 4 bytes.
    E.g., 500 × 1024 × 1024 × 3 × 4 ≈ 6 GB (vs 60 GB for 5000 at once).
    """
    cfg = DATASETS[dataset]
    color_mode = cfg['color_mode']
    effective_n = n_samples if n_samples is not None else cfg['n_samples']

    # Check cache
    cache_path = get_cache_path(dataset, effective_n)
    if not force and cache_path.exists():
        print(f"  {dataset}: cached at {cache_path}")
        return {'status': 'cached', 'path': str(cache_path)}

    print(f"\n{'='*60}")
    print(f"  {dataset} | color={color_mode} | n_samples={effective_n} | batch={batch_size}")
    print(f"{'='*60}")

    t_start = time.time()

    # Prepare batched loader (lightweight — no images loaded yet)
    labels, class_names, load_batch = prepare_batched_loader(
        dataset, n_samples=effective_n, seed=seed)
    actual_n = len(labels)

    print(f"  Samples: {actual_n}, classes: {len(class_names)}")

    # Accumulate PH diagrams across batches
    all_diags_gray = []
    all_diags_R = [] if color_mode == 'per_channel' else None
    all_diags_G = [] if color_mode == 'per_channel' else None
    all_diags_B = [] if color_mode == 'per_channel' else None

    n_batches = (actual_n + batch_size - 1) // batch_size
    lib = 'unknown'
    t_ph_total = 0

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, actual_n)
        bsz = end - start

        print(f"\n  Batch {batch_idx+1}/{n_batches} [{start}:{end}]")

        # Load this batch of images
        t0 = time.time()
        images = load_batch(start, end)
        t_load = time.time() - t0
        print(f"    Loaded: shape={images.shape}, {t_load:.1f}s")

        # Optional comparison (first batch only)
        if compare and batch_idx == 0:
            gray_cmp = _to_grayscale_float(images) if images.ndim == 4 else images
            compare_cuph_cripser(gray_cmp[:min(50, bsz)], dataset)

        # Grayscale PH
        if images.ndim == 4:
            gray_images = _to_grayscale_float(images)
        else:
            gray_images = images

        t0 = time.time()
        diags_batch, lib = compute_ph(gray_images, n_jobs)
        t_gray = time.time() - t0
        all_diags_gray.extend(diags_batch)
        t_ph_total += t_gray

        h0_counts = [len(d['H0']) for d in diags_batch]
        h1_counts = [len(d['H1']) for d in diags_batch]
        print(f"    Gray PH: {t_gray:.1f}s [{lib}], "
              f"H0 mean={np.mean(h0_counts):.0f}, H1 mean={np.mean(h1_counts):.0f}")

        # Per-channel PH (RGB only)
        if color_mode == 'per_channel' and images.ndim == 4:
            R, G, B = rgb_to_channels(images)

            t0 = time.time()
            dr, _ = compute_ph(R, n_jobs)
            all_diags_R.extend(dr)
            t_r = time.time() - t0

            t0 = time.time()
            dg, _ = compute_ph(G, n_jobs)
            all_diags_G.extend(dg)
            t_g = time.time() - t0

            t0 = time.time()
            db, _ = compute_ph(B, n_jobs)
            all_diags_B.extend(db)
            t_b = time.time() - t0

            t_rgb = t_r + t_g + t_b
            t_ph_total += t_rgb
            print(f"    RGB PH: R={t_r:.1f}s, G={t_g:.1f}s, B={t_b:.1f}s")
            del R, G, B

        del images, gray_images, diags_batch
        gc.collect()

    # Infer image shape from first batch
    sample_batch = load_batch(0, min(1, actual_n))
    image_shape = sample_batch.shape[1:]
    del sample_batch

    # Save cache
    cache_data = {
        'dataset': dataset,
        'n_samples': actual_n,
        'labels': labels,
        'class_names': class_names,
        'image_shape': image_shape,
        'color_mode': color_mode,
        'seed': seed,
        'diagrams_gray_sublevel': all_diags_gray,
        'diagrams_R_sublevel': all_diags_R,
        'diagrams_G_sublevel': all_diags_G,
        'diagrams_B_sublevel': all_diags_B,
        'ph_library': lib,
        'ph_time_seconds': t_ph_total,
    }

    PH_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total = time.time() - t_start

    del all_diags_gray, all_diags_R, all_diags_G, all_diags_B
    gc.collect()

    sz = cache_path.stat().st_size / (1024**2)
    print(f"\n  Saved: {cache_path} ({sz:.1f} MB)")
    print(f"  PH time: {t_ph_total:.1f}s")
    print(f"  Total: {total:.1f}s")

    return {
        'status': 'computed', 'path': str(cache_path),
        'n_samples': actual_n, 'ph_library': lib,
        'ph_time': t_ph_total, 'cache_mb': sz,
    }


def load_cached_ph(dataset: str, n_samples: Optional[int] = None) -> Dict:
    """Load PH cache for a dataset."""
    cfg = DATASETS[dataset]
    effective_n = n_samples if n_samples is not None else cfg['n_samples']
    cache_path = get_cache_path(dataset, effective_n)
    if not cache_path.exists():
        raise FileNotFoundError(f"No cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Precompute PH for Benchmark4")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all', action='store_true', help='Process all 26 datasets')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Override n_samples (default: from config)')
    parser.add_argument('--n-jobs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Images per batch for memory-efficient loading (default: 500)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--compare', action='store_true',
                        help='Compare CuPH vs Cripser on each dataset')
    args = parser.parse_args()

    print("=" * 60)
    print("  BENCHMARK4: PRECOMPUTE PERSISTENT HOMOLOGY")
    print("=" * 60)
    print(f"  PH libraries: CuPH={'YES' if HAS_CUPH else 'NO'}, "
          f"Cripser={'YES' if HAS_CRIPSER else 'NO'}")
    print(f"  Cache: {PH_CACHE_PATH}")

    if args.dataset:
        datasets = [args.dataset]
    elif args.all:
        datasets = ALL_DATASETS
    else:
        parser.print_help()
        return

    results = {}
    t_total = time.time()

    for ds in datasets:
        try:
            results[ds] = precompute_one(
                ds, n_samples=args.n_samples, seed=args.seed,
                n_jobs=args.n_jobs, force=args.force, compare=args.compare,
                batch_size=args.batch_size)
        except Exception as e:
            print(f"  ERROR {ds}: {e}")
            import traceback
            traceback.print_exc()
            results[ds] = {'status': 'error', 'error': str(e)}

    elapsed = time.time() - t_total

    # Summary
    computed = [d for d, r in results.items() if r['status'] == 'computed']
    cached = [d for d, r in results.items() if r['status'] == 'cached']
    errors = [d for d, r in results.items() if r.get('status') == 'error']

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(computed)} computed, {len(cached)} cached, {len(errors)} errors")
    print(f"  Total time: {elapsed:.1f}s")
    if computed:
        total_mb = sum(results[d].get('cache_mb', 0) for d in computed)
        print(f"  Total cache size: {total_mb:.0f} MB")
    if errors:
        for d in errors:
            print(f"  ERROR {d}: {results[d].get('error', 'unknown')}")


if __name__ == '__main__':
    main()
