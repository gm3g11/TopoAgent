#!/usr/bin/env python3
"""Precompute fusion accuracy lookup — concatenated descriptor pairs.

For each dataset × each descriptor pair:
  1. Load PH cache and extract features from both descriptors (via benchmark4 infra)
  2. Concatenate feature vectors
  3. Run 5-fold stratified CV with best classifier (TabPFN or XGBoost)
  4. Store results in fusion_accuracy_lookup.json

Usage:
    # Pilot: 3 datasets × top-3 pairs
    python scripts/precompute_fusion_assets.py --pilot

    # Specific pair on specific dataset
    python scripts/precompute_fusion_assets.py --dataset TissueMNIST --pair ATOL+edge_histogram

    # Full run: top-5 pairs per object type × 26 datasets
    python scripts/precompute_fusion_assets.py --full

    # SGE job mode (single dataset+pair from env vars)
    python scripts/precompute_fusion_assets.py --job
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))

from RuleBenchmark.benchmark4.config import (
    DATASETS, PH_CACHE_PATH, PH_BASED_DESCRIPTORS,
    IMAGE_BASED_DESCRIPTORS, LEARNED_DESCRIPTORS,
)
from RuleBenchmark.benchmark4.optimal_rules import get_rules
from RuleBenchmark.benchmark4.precompute_ph import load_cached_ph
from RuleBenchmark.benchmark4.descriptor_runner import (
    extract_features, extract_features_per_channel, extract_learned_features,
)
from RuleBenchmark.benchmark4.classifier_wrapper import get_classifier
from RuleBenchmark.benchmark4.data_loader import (
    load_dataset, rgb_to_channels, _to_grayscale_float,
)
from topoagent.skills.rules_data import (
    DATASET_TO_OBJECT_TYPE, DATASET_COLOR_MODE,
    BENCHMARK4_TOP_PERFORMERS_BY_TYPE, MOST_COMPLEMENTARY_PAIRS,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "assets"
FUSION_RESULTS_PATH = RESULTS_DIR / "fusion_accuracy_lookup.json"

# Classifiers to try (in priority order)
FUSION_CLASSIFIERS = ["TabPFN", "XGBoost"]


def get_descriptor_features(
    dataset_name: str,
    descriptor: str,
    object_type: str,
    color_mode: str,
    images: np.ndarray,
    labels: np.ndarray,
    ph_cache: dict = None,
    seed: int = 42,
) -> np.ndarray:
    """Extract features for a single descriptor.

    Returns:
        Feature matrix (n_samples, dim) or None if extraction fails.
    """
    rules = get_rules()
    per_channel = (color_mode == "per_channel")
    config = rules.get_descriptor_config(descriptor, object_type, color_mode)
    params = config["params"]
    dim_per_channel = config["dim_per_channel"]

    is_ph_based = descriptor in PH_BASED_DESCRIPTORS or descriptor in LEARNED_DESCRIPTORS
    is_learned = descriptor in LEARNED_DESCRIPTORS

    if per_channel:
        if images.ndim == 4 and images.shape[3] == 3:
            R, G, B = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
        else:
            R = G = B = images if images.ndim == 3 else images[:, :, :, 0]

        if is_ph_based:
            # Load PH cache per channel
            if ph_cache is not None:
                diags_R = ph_cache.get("R")
                diags_G = ph_cache.get("G")
                diags_B = ph_cache.get("B")
            else:
                from RuleBenchmark.benchmark4.precompute_ph import compute_ph
                diags_R, _ = compute_ph(R)
                diags_G, _ = compute_ph(G)
                diags_B, _ = compute_ph(B)

            if is_learned:
                features = extract_learned_features(
                    descriptor,
                    diags_R=diags_R, diags_G=diags_G, diags_B=diags_B,
                    images_R=R, images_G=G, images_B=B,
                    params=params, expected_dim_per_channel=dim_per_channel,
                    labels=labels, seed=seed,
                )
            else:
                features = extract_features_per_channel(
                    descriptor,
                    diags_R=diags_R, diags_G=diags_G, diags_B=diags_B,
                    params=params, expected_dim_per_channel=dim_per_channel,
                )
        else:
            features = extract_features_per_channel(
                descriptor,
                images_R=R, images_G=G, images_B=B,
                params=params, expected_dim_per_channel=dim_per_channel,
            )
    else:
        # Grayscale
        if images.ndim == 4:
            gray = _to_grayscale_float(images)
        else:
            gray = images

        if is_ph_based:
            if ph_cache is not None:
                diagrams = ph_cache.get("gray") or ph_cache.get("R")
            else:
                from RuleBenchmark.benchmark4.precompute_ph import compute_ph
                diagrams, _ = compute_ph(gray)

            if is_learned:
                features = extract_learned_features(
                    descriptor,
                    diagrams=diagrams, images=gray,
                    params=params, expected_dim=dim_per_channel,
                    labels=labels, seed=seed,
                )
            else:
                features = extract_features(
                    descriptor, diagrams=diagrams,
                    params=params, expected_dim=dim_per_channel,
                )
        else:
            features = extract_features(
                descriptor, images=gray,
                params=params, expected_dim=dim_per_channel,
            )

    return features


def evaluate_fusion(
    features_list: list,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Concatenate features and run k-fold CV.

    Args:
        features_list: List of feature matrices to concatenate
        labels: Label array
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Dict with balanced_accuracy, fold_scores, classifier, total_dim
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    # Concatenate features
    X = np.hstack(features_list)
    total_dim = X.shape[1]
    n_classes = len(np.unique(labels))

    # Choose classifier: try TabPFN first, fallback to XGBoost (no GPU needed)
    import torch
    has_gpu = torch.cuda.is_available()

    if total_dim > 2000 or n_classes > 10 or not has_gpu:
        clf_name = "XGBoost"
    else:
        clf_name = "TabPFN"

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores = []

    for train_idx, test_idx in skf.split(X, labels):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Clip extreme values
        X_train = np.clip(X_train, -1e6, 1e6)
        X_test = np.clip(X_test, -1e6, 1e6)

        try:
            clf = get_classifier(clf_name, n_features=total_dim,
                                 n_classes=n_classes, seed=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fold_scores.append(balanced_accuracy_score(y_test, y_pred))
        except Exception as e:
            # Fallback chain: XGBoost -> RandomForest
            fallback = "XGBoost" if clf_name != "XGBoost" else "CatBoost"
            try:
                clf = get_classifier(fallback, n_features=total_dim,
                                     n_classes=n_classes, seed=seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                fold_scores.append(balanced_accuracy_score(y_test, y_pred))
                clf_name = fallback
            except Exception:
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, random_state=seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                fold_scores.append(balanced_accuracy_score(y_test, y_pred))
                clf_name = "RandomForest (fallback)"

    return {
        "balanced_accuracy": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "fold_scores": [float(s) for s in fold_scores],
        "classifier": clf_name,
        "total_dim": total_dim,
        "n_samples": len(labels),
    }


def load_ph_cache_for_dataset(dataset_name: str, color_mode: str):
    """Load pre-computed PH cache for a dataset."""
    cache = {}
    if color_mode == "per_channel":
        for channel in ["R", "G", "B"]:
            cache_path = PH_CACHE_PATH / f"{dataset_name}_{channel}.npz"
            if cache_path.exists():
                data = np.load(str(cache_path), allow_pickle=True)
                cache[channel] = data["diagrams"].tolist()
                print(f"    Loaded PH cache: {cache_path.name} ({len(cache[channel])} diagrams)")
            else:
                print(f"    WARNING: PH cache not found: {cache_path}")
                return None
    else:
        cache_path = PH_CACHE_PATH / f"{dataset_name}.npz"
        if cache_path.exists():
            data = np.load(str(cache_path), allow_pickle=True)
            cache["gray"] = data["diagrams"].tolist()
            print(f"    Loaded PH cache: {cache_path.name} ({len(cache['gray'])} diagrams)")
        else:
            print(f"    WARNING: PH cache not found: {cache_path}")
            return None
    return cache


def get_pilot_experiments():
    """Get pilot experiments: 3 high-gap datasets × 3 complementary pairs.

    Pairs use only supported descriptors (no ATOL/persistence_codebook).
    Selected for complementarity (low Spearman rho from PAIRWISE_SPEARMAN_RHO).
    """
    pilot_datasets = ["TissueMNIST", "OCTMNIST", "MURA"]
    pilot_pairs = [
        ("persistence_statistics", "edge_histogram"),  # rho=0.482 (most complementary supported)
        ("lbp_texture", "edge_histogram"),              # rho=0.581
        ("persistence_statistics", "lbp_texture"),      # rho=0.846
    ]
    experiments = []
    for ds in pilot_datasets:
        for d1, d2 in pilot_pairs:
            experiments.append((ds, d1, d2))
    return experiments


def get_full_experiments():
    """Get full experiments: top-5 pairs per object type × all 26 datasets."""
    experiments = []
    for ds_name, ds_config in DATASETS.items():
        object_type = ds_config["object_type"]
        top5 = BENCHMARK4_TOP_PERFORMERS_BY_TYPE.get(object_type, [])
        top_descs = [e["descriptor"] for e in top5]

        # All pairs among top-5
        for i, d1 in enumerate(top_descs):
            for d2 in top_descs[i + 1:]:
                experiments.append((ds_name, d1, d2))

        # Also add edge_histogram as complement to top-1 if not already in top-5
        if top_descs and "edge_histogram" not in top_descs:
            experiments.append((ds_name, top_descs[0], "edge_histogram"))

    return experiments


def run_fusion_experiment(dataset_name, desc1, desc2, n_samples=5000, seed=42):
    """Run a single fusion experiment.

    Returns:
        Result dict or None if failed.
    """
    print(f"\n  {dataset_name}: {desc1} + {desc2}")

    ds_config = DATASETS.get(dataset_name)
    if ds_config is None:
        print(f"    ERROR: Unknown dataset {dataset_name}")
        return None

    object_type = ds_config["object_type"]
    color_mode = ds_config.get("color_mode", "grayscale")
    effective_n = min(n_samples, ds_config.get("n_samples", n_samples))

    # Load data
    t0 = time.time()
    images, labels, class_names = load_dataset(dataset_name, n_samples=effective_n, seed=seed)
    print(f"    Loaded {len(images)} images in {time.time()-t0:.1f}s")

    # Load PH cache (if either descriptor needs PH)
    ph_based_set = set(PH_BASED_DESCRIPTORS) | set(LEARNED_DESCRIPTORS)
    need_ph = desc1 in ph_based_set or desc2 in ph_based_set
    ph_cache = None
    if need_ph:
        ph_cache = load_ph_cache_for_dataset(dataset_name, color_mode)
        if ph_cache is None:
            print(f"    WARNING: No PH cache, computing on the fly...")

    # Extract features for both descriptors
    t1 = time.time()
    feat1 = get_descriptor_features(
        dataset_name, desc1, object_type, color_mode,
        images, labels, ph_cache=ph_cache, seed=seed,
    )
    if feat1 is None or len(feat1) == 0:
        print(f"    ERROR: Feature extraction failed for {desc1}")
        return None
    print(f"    {desc1}: {feat1.shape} in {time.time()-t1:.1f}s")

    t2 = time.time()
    feat2 = get_descriptor_features(
        dataset_name, desc2, object_type, color_mode,
        images, labels, ph_cache=ph_cache, seed=seed,
    )
    if feat2 is None or len(feat2) == 0:
        print(f"    ERROR: Feature extraction failed for {desc2}")
        return None
    print(f"    {desc2}: {feat2.shape} in {time.time()-t2:.1f}s")

    # Ensure same number of samples
    n = min(len(feat1), len(feat2), len(labels))
    feat1, feat2, labels_cv = feat1[:n], feat2[:n], labels[:n]

    # Evaluate fusion (concatenation)
    t3 = time.time()
    result = evaluate_fusion([feat1, feat2], labels_cv, n_folds=5, seed=seed)
    result["dataset"] = dataset_name
    result["desc1"] = desc1
    result["desc2"] = desc2
    result["desc1_dim"] = feat1.shape[1]
    result["desc2_dim"] = feat2.shape[1]
    result["object_type"] = object_type
    result["color_mode"] = color_mode
    result["time_total"] = round(time.time() - t0, 1)
    print(f"    FUSION: {result['balanced_accuracy']:.3f} ± {result['std']:.3f} "
          f"(dim={result['total_dim']}, clf={result['classifier']}, "
          f"{result['time_total']:.1f}s)")

    # Clean up
    del feat1, feat2, images, labels, ph_cache
    gc.collect()

    return result


def load_existing_results():
    """Load existing fusion results to skip already-computed experiments."""
    if FUSION_RESULTS_PATH.exists():
        with open(FUSION_RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_results(results_dict):
    """Save fusion results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FUSION_RESULTS_PATH, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved {len(results_dict)} fusion results to {FUSION_RESULTS_PATH}")


def make_key(dataset, desc1, desc2):
    """Create a canonical key for a fusion pair (sorted descriptors)."""
    d1, d2 = sorted([desc1, desc2])
    return f"{dataset}|{d1}+{d2}"


def main():
    parser = argparse.ArgumentParser(description="Precompute fusion accuracy lookup")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot: 3 datasets × 3 pairs")
    parser.add_argument("--full", action="store_true",
                        help="Run full: top-5 pairs per type × 26 datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Specific dataset")
    parser.add_argument("--pair", type=str, default=None,
                        help="Specific pair, e.g. 'ATOL+edge_histogram'")
    parser.add_argument("--job", action="store_true",
                        help="SGE job mode: read FUSION_DATASET, FUSION_DESC1, FUSION_DESC2 from env")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip already-computed experiments")
    args = parser.parse_args()

    # Determine experiments
    if args.job:
        dataset = os.environ.get("FUSION_DATASET")
        desc1 = os.environ.get("FUSION_DESC1")
        desc2 = os.environ.get("FUSION_DESC2")
        if not all([dataset, desc1, desc2]):
            print("ERROR: --job mode requires FUSION_DATASET, FUSION_DESC1, FUSION_DESC2 env vars")
            sys.exit(1)
        experiments = [(dataset, desc1, desc2)]
    elif args.dataset and args.pair:
        d1, d2 = args.pair.split("+")
        experiments = [(args.dataset, d1, d2)]
    elif args.pilot:
        experiments = get_pilot_experiments()
    elif args.full:
        experiments = get_full_experiments()
    else:
        print("Specify --pilot, --full, --dataset+--pair, or --job")
        sys.exit(1)

    print(f"Fusion experiments to run: {len(experiments)}")

    # Load existing results
    all_results = load_existing_results()
    n_skipped = 0

    for dataset, desc1, desc2 in experiments:
        key = make_key(dataset, desc1, desc2)
        if args.skip_existing and key in all_results:
            n_skipped += 1
            continue

        result = run_fusion_experiment(
            dataset, desc1, desc2,
            n_samples=args.n_samples, seed=args.seed,
        )

        if result is not None:
            all_results[key] = result
            # Save incrementally
            save_results(all_results)

    if n_skipped > 0:
        print(f"\nSkipped {n_skipped} already-computed experiments")

    # Summary
    print(f"\n{'='*60}")
    print(f"FUSION RESULTS SUMMARY")
    print(f"{'='*60}")

    # Load oracle for comparison
    oracle_path = RESULTS_DIR / "oracle.json"
    oracle_data = {}
    if oracle_path.exists():
        with open(oracle_path) as f:
            oracle_data = json.load(f)

    for key, result in sorted(all_results.items()):
        ds = result["dataset"]
        d1, d2 = result["desc1"], result["desc2"]
        fusion_acc = result["balanced_accuracy"]
        oracle_acc = oracle_data.get(ds, {}).get("accuracy", 0)
        gain = fusion_acc - oracle_acc
        marker = "BEATS ORACLE" if gain > 0.001 else ""
        print(f"  {ds:24s}  {d1:24s} + {d2:24s}  "
              f"fusion={fusion_acc:.3f}  oracle={oracle_acc:.3f}  "
              f"gain={gain:+.3f}  {marker}")


if __name__ == "__main__":
    main()
