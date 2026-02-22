"""Load performance data and compute dataset characterization features."""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

from .config import (
    EXP1_CSV, DATASETS, DESCRIPTORS, FEATURE_NAMES,
    TARGET_CLASSIFIER, TARGET_METRIC, N_REPEATS, SAMPLES_PER_REPEAT,
    DEFAULT_N_SAMPLES, CACHE_DIR, SEED,
)
from .features import compute_cheap_features_stable


def load_performance_matrix(csv_path=None):
    """Load exp1 results and filter to TabPFN balanced_accuracy.

    Returns:
        DataFrame with columns: [dataset, descriptor, score, score_std]
        Typically 192 rows (13 datasets × 15 descriptors, minus MURA's 3 missing).
    """
    if csv_path is None:
        csv_path = EXP1_CSV

    df = pd.read_csv(csv_path)

    # Filter to TabPFN
    df_tab = df[df["classifier"] == TARGET_CLASSIFIER].copy()

    # Select relevant columns
    result = pd.DataFrame({
        "dataset": df_tab["dataset"],
        "descriptor": df_tab["descriptor"],
        "score": df_tab[TARGET_METRIC],
        "score_std": df_tab[TARGET_METRIC.replace("_mean", "_std")],
    }).reset_index(drop=True)

    # Validate
    n_rows = len(result)
    print(f"[data_loader] Loaded {n_rows} rows from {csv_path}")
    print(f"  Datasets: {result['dataset'].nunique()}")
    print(f"  Descriptors per dataset: {result.groupby('dataset')['descriptor'].count().to_dict()}")

    return result


def _load_dataset_images(dataset_name, n_samples=DEFAULT_N_SAMPLES, seed=SEED):
    """Load images for a dataset using benchmark3's data_loader.

    Returns:
        (images, labels) where images: (N, H, W) float32 [0,1], labels: (N,) int
    """
    # Add benchmark3 path to sys.path for importing its data_loader
    benchmark3_dir = Path(__file__).parent.parent
    if str(benchmark3_dir) not in sys.path:
        sys.path.insert(0, str(benchmark3_dir))

    from data_loader import load_dataset

    images, labels, class_names = load_dataset(
        dataset_name, n_samples=n_samples, seed=seed
    )
    return images, labels


def load_all_dataset_features(n_samples=DEFAULT_N_SAMPLES, n_repeats=N_REPEATS,
                              samples_per_repeat=SAMPLES_PER_REPEAT,
                              cache_path=None, recompute=False, seed=SEED):
    """Compute cheap features for all 13 datasets.

    Args:
        n_samples: Total images to load per dataset
        n_repeats: Subsamples for stability averaging
        samples_per_repeat: Images per subsample
        cache_path: Path to cache CSV (None = default in CACHE_DIR)
        recompute: Force recomputation even if cache exists
        seed: Random seed

    Returns:
        dict: {dataset_name: OrderedDict of 25 features}
    """
    if cache_path is None:
        cache_path = CACHE_DIR / "dataset_characteristics.csv"
    cache_path = Path(cache_path)

    # Try loading from cache
    if cache_path.exists() and not recompute:
        print(f"[data_loader] Loading cached features from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0)
        result = {}
        for dataset in df.index:
            result[dataset] = OrderedDict(
                (col, df.loc[dataset, col]) for col in FEATURE_NAMES
            )
        print(f"  Loaded features for {len(result)} datasets")
        return result

    # Compute features for each dataset
    print(f"[data_loader] Computing features for {len(DATASETS)} datasets...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result = {}

    for i, dataset_name in enumerate(DATASETS):
        print(f"\n  [{i+1}/{len(DATASETS)}] {dataset_name}...")
        t0 = time.time()

        try:
            images, labels = _load_dataset_images(dataset_name, n_samples=n_samples, seed=seed)
            print(f"    Loaded {len(images)} images, shape={images.shape}")

            features = compute_cheap_features_stable(
                images, labels, n_repeats=n_repeats,
                samples_per_repeat=samples_per_repeat, seed=seed,
            )
            result[dataset_name] = features
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            # Fill with zeros so we can still proceed
            result[dataset_name] = OrderedDict((name, 0.0) for name in FEATURE_NAMES)

    # Save cache
    df = pd.DataFrame(result).T
    df.index.name = "dataset"
    df.to_csv(cache_path)
    print(f"\n[data_loader] Cached features to {cache_path}")

    return result


def build_training_data(dataset_features, performance_df, descriptor_list=None):
    """Build training matrix for meta-learner.

    Each row = [25 dataset features + 15 descriptor one-hot] → score

    Args:
        dataset_features: {dataset: OrderedDict of 25 features}
        performance_df: DataFrame with [dataset, descriptor, score, score_std]
        descriptor_list: list of descriptor names (default: DESCRIPTORS)

    Returns:
        X: np.ndarray (n_rows, 40) = 25 continuous + 15 one-hot
        y: np.ndarray (n_rows,) = balanced_accuracy scores
        meta_df: DataFrame with [dataset, descriptor] for LODO splitting
    """
    if descriptor_list is None:
        descriptor_list = DESCRIPTORS

    n_descriptors = len(descriptor_list)
    desc_to_idx = {d: i for i, d in enumerate(descriptor_list)}

    rows_X = []
    rows_y = []
    rows_meta = []

    for _, row in performance_df.iterrows():
        dataset = row["dataset"]
        descriptor = row["descriptor"]
        score = row["score"]

        # Skip if dataset features not available
        if dataset not in dataset_features:
            print(f"  Warning: No features for dataset '{dataset}', skipping")
            continue

        # Skip if descriptor not in our list
        if descriptor not in desc_to_idx:
            print(f"  Warning: Unknown descriptor '{descriptor}', skipping")
            continue

        # 25 continuous features
        feat_vals = [dataset_features[dataset][name] for name in FEATURE_NAMES]

        # 15 one-hot for descriptor
        one_hot = [0.0] * n_descriptors
        one_hot[desc_to_idx[descriptor]] = 1.0

        rows_X.append(feat_vals + one_hot)
        rows_y.append(score)
        rows_meta.append({"dataset": dataset, "descriptor": descriptor})

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)
    meta_df = pd.DataFrame(rows_meta)

    print(f"[data_loader] Built training data: X={X.shape}, y={y.shape}")
    print(f"  Datasets: {meta_df['dataset'].nunique()}, Descriptors: {meta_df['descriptor'].nunique()}")

    return X, y, meta_df
