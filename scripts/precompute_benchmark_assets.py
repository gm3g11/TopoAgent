#!/usr/bin/env python3
"""Pre-compute all benchmark assets for TopoBenchmark lookup mode.

Generates derived artifacts from benchmark4 results so that the benchmark
runner can operate in pure lookup mode (no GPU, no PH, ~30 seconds).

Outputs → results/topobenchmark/assets/:
  1. accuracy_lookup.json              — {dataset: {descriptor: best_accuracy}}
  2. oracle.json                       — {dataset: {descriptor, accuracy}}
  3. sba.json                          — single best algorithm stats
  4. top_performers.json               — rankings per object type from benchmark4
  5. top_performers_lodo/              — 26 LODO variants (one per held-out dataset)
  6. cheap_features.json               — {dataset: {25 feature values}}
  7. meta_learner_lodo.json            — pre-computed GBR LODO predictions (25 features only)
  8. meta_learner_ridge_lodo.json      — pre-computed RidgeCV LODO predictions (25 features only)
  9. rule_based_predictions.json       — rule-based predictions per dataset
 10. meta_learner_train_test.json      — RidgeCV trained on 5 exp4 datasets (25+15 features)
 11. meta_learner_ridge_train_test.json — alias for #10
 12. exp4_rankings.json                — exp4-only rankings for all 15 descriptors per type

Usage:
    python scripts/precompute_benchmark_assets.py
    python scripts/precompute_benchmark_assets.py --skip-features  # skip slow feature computation
"""

import argparse
import json
import sys
import numpy as np
from collections import defaultdict, OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))

# exp7_meta_learning uses relative imports, so we load it as a package
EXP7_DIR = PROJECT_ROOT / "RuleBenchmark" / "benchmark3" / "exp7_meta_learning"


def _load_exp7_modules():
    """Load exp7 config and features modules (they use relative imports)."""
    import importlib
    # Add benchmark3/ to sys.path so exp7_meta_learning is a findable package
    exp7_parent = str(EXP7_DIR.parent)  # RuleBenchmark/benchmark3/
    if exp7_parent not in sys.path:
        sys.path.insert(0, exp7_parent)
    # Ensure exp7_meta_learning has __init__.py
    exp7_init = EXP7_DIR / "__init__.py"
    if not exp7_init.exists():
        exp7_init.touch()
    exp7_config = importlib.import_module("exp7_meta_learning.config")
    exp7_features = importlib.import_module("exp7_meta_learning.features")
    return exp7_config, exp7_features

ASSETS_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "assets"
B4_CSV = PROJECT_ROOT / "results" / "benchmark4" / "summary" / "benchmark4_all_results.csv"

# All 15 descriptors
EXTENDED_DESCRIPTORS = [
    "persistence_statistics", "persistence_image", "persistence_landscapes",
    "persistence_silhouette", "betti_curves", "persistence_entropy",
    "persistence_codebook", "tropical_coordinates", "ATOL",
    "template_functions", "minkowski_functionals", "euler_characteristic_curve",
    "euler_characteristic_transform", "lbp_texture", "edge_histogram",
]


def load_benchmark4_csv():
    """Load benchmark4 CSV into a list of dicts."""
    import csv
    rows = []
    with open(B4_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} rows from benchmark4 CSV")
    return rows


def build_accuracy_lookup(rows):
    """Build {dataset: {descriptor: best_accuracy}} from benchmark4 rows."""
    lookup = defaultdict(dict)
    for row in rows:
        ds = row["dataset"]
        desc = row["descriptor"]
        acc = float(row["best_accuracy"])
        lookup[ds][desc] = acc
    return dict(lookup)


def build_oracle(accuracy_lookup):
    """Build {dataset: {descriptor, accuracy}} — best descriptor per dataset."""
    oracle = {}
    for ds, descs in accuracy_lookup.items():
        best_desc = max(descs, key=descs.get)
        oracle[ds] = {
            "descriptor": best_desc,
            "accuracy": descs[best_desc],
        }
    return oracle


def build_sba(accuracy_lookup):
    """Build single best algorithm stats."""
    # Compute mean accuracy per descriptor across all datasets it appears in
    desc_accs = defaultdict(list)
    for ds, descs in accuracy_lookup.items():
        for desc, acc in descs.items():
            desc_accs[desc].append((ds, acc))

    # Find SBA with full coverage (26 datasets)
    n_datasets = len(accuracy_lookup)
    full_coverage = {
        desc: entries for desc, entries in desc_accs.items()
        if len(entries) == n_datasets
    }

    if full_coverage:
        best_desc = max(full_coverage, key=lambda d: np.mean([a for _, a in full_coverage[d]]))
        per_dataset = {ds: acc for ds, acc in full_coverage[best_desc]}
        mean_acc = np.mean(list(per_dataset.values()))
    else:
        # Fallback: use descriptor with most coverage
        best_desc = max(desc_accs, key=lambda d: (len(desc_accs[d]), np.mean([a for _, a in desc_accs[d]])))
        per_dataset = {ds: acc for ds, acc in desc_accs[best_desc]}
        mean_acc = np.mean(list(per_dataset.values()))

    return {
        "descriptor": best_desc,
        "mean_accuracy": float(mean_acc),
        "n_datasets": len(per_dataset),
        "total_datasets": n_datasets,
        "per_dataset": per_dataset,
    }


def build_top_performers(rows, accuracy_lookup):
    """Build rankings per object type, aggregated from benchmark4."""
    from TopoBenchmark.config import DATASET_DESCRIPTIONS

    # Group datasets by object type
    ot_datasets = defaultdict(list)
    for ds, desc_info in DATASET_DESCRIPTIONS.items():
        ot = desc_info["object_type"]
        if ds in accuracy_lookup:
            ot_datasets[ot].append(ds)

    top_performers = {}
    for ot, datasets in ot_datasets.items():
        # For each descriptor, compute mean accuracy across datasets in this object type
        desc_stats = defaultdict(list)
        for ds in datasets:
            for desc, acc in accuracy_lookup.get(ds, {}).items():
                desc_stats[desc].append(acc)

        rankings = []
        for desc, accs in desc_stats.items():
            rankings.append({
                "descriptor": desc,
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "datasets": len(accs),
                "per_dataset": {
                    ds: accuracy_lookup[ds].get(desc)
                    for ds in datasets if desc in accuracy_lookup.get(ds, {})
                },
            })

        # Sort by mean accuracy descending
        rankings.sort(key=lambda x: x["mean_accuracy"], reverse=True)
        top_performers[ot] = rankings

    return top_performers


def build_top_performers_lodo(rows, accuracy_lookup):
    """Build 26 LODO variants of top_performers (one per held-out dataset)."""
    from TopoBenchmark.config import DATASET_DESCRIPTIONS

    all_datasets = [ds for ds in DATASET_DESCRIPTIONS if ds in accuracy_lookup]

    # Group datasets by object type
    ds_to_ot = {ds: DATASET_DESCRIPTIONS[ds]["object_type"] for ds in all_datasets}

    lodo_variants = {}
    for held_out in all_datasets:
        held_out_ot = ds_to_ot[held_out]

        # Group remaining datasets by object type
        ot_datasets = defaultdict(list)
        for ds in all_datasets:
            if ds == held_out:
                continue
            ot_datasets[ds_to_ot[ds]].append(ds)

        top_performers = {}
        for ot, datasets in ot_datasets.items():
            desc_stats = defaultdict(list)
            for ds in datasets:
                for desc, acc in accuracy_lookup.get(ds, {}).items():
                    desc_stats[desc].append(acc)

            rankings = []
            for desc, accs in desc_stats.items():
                rankings.append({
                    "descriptor": desc,
                    "mean_accuracy": float(np.mean(accs)),
                    "datasets": len(accs),
                })
            rankings.sort(key=lambda x: x["mean_accuracy"], reverse=True)
            top_performers[ot] = rankings

        lodo_variants[held_out] = top_performers

    return lodo_variants


def compute_all_cheap_features(accuracy_lookup, n_samples=500):
    """Compute cheap features for all 26 datasets."""
    from data_loader import load_dataset
    _, exp7_features = _load_exp7_modules()
    compute_cheap_features_stable = exp7_features.compute_cheap_features_stable

    all_features = {}
    datasets = sorted(accuracy_lookup.keys())

    for i, ds in enumerate(datasets, 1):
        print(f"  [{i}/{len(datasets)}] Computing features for {ds}...", end=" ", flush=True)
        try:
            images, labels, class_names = load_dataset(ds, n_samples=n_samples, seed=42)

            # Convert to grayscale if needed
            if images.ndim == 4 and images.shape[3] == 3:
                gray = images.mean(axis=3).astype(np.float32)
            elif images.ndim == 4 and images.shape[3] == 1:
                gray = images[:, :, :, 0].astype(np.float32)
            else:
                gray = images.astype(np.float32)

            # Normalize to [0, 1] if needed
            if gray.max() > 1.0:
                gray = gray / 255.0

            features = compute_cheap_features_stable(gray, labels, n_repeats=5,
                                                     samples_per_repeat=100, seed=42)
            all_features[ds] = dict(features)
            print(f"OK ({len(features)} features)")
        except Exception as e:
            print(f"FAILED: {e}")
            all_features[ds] = None

    return all_features


def build_meta_learner_lodo(accuracy_lookup, cheap_features):
    """Train GBR in LODO fashion and save predictions + feature importances.

    Uses [25 cheap features] + [15 descriptor one-hot] = 40D features.
    This fixes the degeneracy bug where the original 25-feature model
    predicted the same score for all descriptors on each dataset.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    exp7_config, _ = _load_exp7_modules()
    GBR_PARAMS = exp7_config.GBR_PARAMS
    FEATURE_NAMES = exp7_config.FEATURE_NAMES

    datasets = [ds for ds in sorted(accuracy_lookup.keys())
                if cheap_features.get(ds) is not None]

    # Descriptor one-hot encoding
    desc_to_idx = {d: i for i, d in enumerate(EXTENDED_DESCRIPTORS)}
    n_desc = len(EXTENDED_DESCRIPTORS)

    # Build feature matrix: each row = (dataset, descriptor) → 25 cheap + 15 one-hot
    all_X = []
    all_y = []
    all_meta = []  # (dataset, descriptor)

    for ds in datasets:
        feat = cheap_features[ds]
        cheap_vec = [feat[fn] for fn in FEATURE_NAMES]
        for desc in EXTENDED_DESCRIPTORS:
            acc = accuracy_lookup.get(ds, {}).get(desc)
            if acc is not None:
                onehot = [0.0] * n_desc
                onehot[desc_to_idx[desc]] = 1.0
                all_X.append(cheap_vec + onehot)
                all_y.append(acc)
                all_meta.append((ds, desc))

    all_X = np.array(all_X)
    all_y = np.array(all_y)
    print(f"  Meta-learner data: {len(all_X)} samples ({len(datasets)} datasets × up to {n_desc} descriptors), "
          f"{all_X.shape[1]} features (25 cheap + {n_desc} descriptor one-hot)")

    lodo_predictions = {}
    all_importances = []  # Collect feature importances from each LODO fold

    for held_out in datasets:
        # Train on all except held_out
        train_mask = np.array([m[0] != held_out for m in all_meta])
        test_mask = ~train_mask

        if test_mask.sum() == 0:
            continue

        X_train, y_train = all_X[train_mask], all_y[train_mask]
        X_test = all_X[test_mask]
        test_descs = [all_meta[i][1] for i in np.where(test_mask)[0]]

        gbr = GradientBoostingRegressor(**GBR_PARAMS)
        gbr.fit(X_train, y_train)

        all_importances.append(gbr.feature_importances_)

        pred_scores = gbr.predict(X_test)
        scores = {desc: float(score) for desc, score in zip(test_descs, pred_scores)}

        # Pick descriptor with highest predicted score
        best_desc = max(scores, key=scores.get)

        lodo_predictions[held_out] = {
            "predicted": best_desc,
            "predicted_score": float(scores[best_desc]),
            "scores": scores,
            "n_train": int(train_mask.sum()),
        }

    # Aggregate feature importances across LODO folds
    all_feature_names = FEATURE_NAMES + [f"desc_{d}" for d in EXTENDED_DESCRIPTORS]
    feature_importance = _aggregate_feature_importances(
        np.array(all_importances), all_feature_names
    )

    return lodo_predictions, feature_importance


def _aggregate_feature_importances(importances_array, feature_names):
    """Aggregate GBR feature importances across LODO folds.

    Args:
        importances_array: (n_folds, n_features) array
        feature_names: list of feature names

    Returns:
        Dict with ranked features, per-feature stats, and category breakdown.
    """
    mean_imp = importances_array.mean(axis=0)
    std_imp = importances_array.std(axis=0)

    # Rank by mean importance
    ranked_idx = np.argsort(mean_imp)[::-1]
    ranked = []
    for rank, idx in enumerate(ranked_idx, 1):
        ranked.append({
            "rank": rank,
            "feature": feature_names[idx],
            "mean_importance": float(mean_imp[idx]),
            "std_importance": float(std_imp[idx]),
            "pct_total": float(mean_imp[idx] / mean_imp.sum() * 100),
        })

    # Category breakdown (matches FEATURE_NAMES groups from config.py)
    FEATURE_CATEGORIES = {
        "structure": ["n_samples", "n_classes", "samples_per_class", "class_imbalance"],
        "intensity": ["intensity_mean", "intensity_std", "intensity_skewness",
                       "intensity_kurtosis", "intensity_p95_p5", "intensity_entropy"],
        "gradient": ["edge_density", "gradient_mean"],
        "topology_proxy": ["otsu_components", "otsu_holes", "binary_fill_ratio"],
        "texture": ["glcm_contrast", "glcm_homogeneity"],
        "frequency": ["fft_low_freq_ratio"],
        "stability": ["otsu_stability"],
        "components": ["component_size_mean", "component_size_cv"],
        "polarity": ["fg_bg_contrast", "polarity"],
        "scale": ["largest_component_ratio"],
        "fine_texture": ["laplacian_variance"],
    }

    feat_to_imp = dict(zip(feature_names, mean_imp))
    category_importance = {}
    for cat, feats in FEATURE_CATEGORIES.items():
        cat_imp = sum(feat_to_imp.get(f, 0) for f in feats)
        category_importance[cat] = {
            "total_importance": float(cat_imp),
            "pct_total": float(cat_imp / mean_imp.sum() * 100),
            "n_features": len(feats),
        }

    # Sort categories by importance
    category_ranked = sorted(category_importance.items(),
                              key=lambda x: x[1]["total_importance"], reverse=True)

    return {
        "n_folds": int(importances_array.shape[0]),
        "n_features": int(importances_array.shape[1]),
        "ranked_features": ranked,
        "category_importance": dict(category_ranked),
    }


def build_meta_learner_ridge_lodo(accuracy_lookup, cheap_features):
    """Train RidgeCV in LODO fashion as a simpler sanity check alongside GBR.

    Uses [25 cheap features] + [15 descriptor one-hot] = 40D features.
    """
    from sklearn.linear_model import RidgeCV
    exp7_config, _ = _load_exp7_modules()
    FEATURE_NAMES = exp7_config.FEATURE_NAMES

    datasets = [ds for ds in sorted(accuracy_lookup.keys())
                if cheap_features.get(ds) is not None]

    # Descriptor one-hot encoding
    desc_to_idx = {d: i for i, d in enumerate(EXTENDED_DESCRIPTORS)}
    n_desc = len(EXTENDED_DESCRIPTORS)

    # Build feature matrix with descriptor one-hot
    all_X = []
    all_y = []
    all_meta = []

    for ds in datasets:
        feat = cheap_features[ds]
        cheap_vec = [feat[fn] for fn in FEATURE_NAMES]
        for desc in EXTENDED_DESCRIPTORS:
            acc = accuracy_lookup.get(ds, {}).get(desc)
            if acc is not None:
                onehot = [0.0] * n_desc
                onehot[desc_to_idx[desc]] = 1.0
                all_X.append(cheap_vec + onehot)
                all_y.append(acc)
                all_meta.append((ds, desc))

    all_X = np.array(all_X)
    all_y = np.array(all_y)
    print(f"  RidgeCV meta-learner data: {len(all_X)} samples, {all_X.shape[1]} features")

    lodo_predictions = {}
    for held_out in datasets:
        train_mask = np.array([m[0] != held_out for m in all_meta])
        test_mask = ~train_mask

        if test_mask.sum() == 0:
            continue

        X_train, y_train = all_X[train_mask], all_y[train_mask]
        X_test = all_X[test_mask]
        test_descs = [all_meta[i][1] for i in np.where(test_mask)[0]]

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_train, y_train)

        pred_scores = ridge.predict(X_test)
        scores = {desc: float(score) for desc, score in zip(test_descs, pred_scores)}

        best_desc = max(scores, key=scores.get)

        lodo_predictions[held_out] = {
            "predicted": best_desc,
            "predicted_score": float(scores[best_desc]),
            "scores": scores,
            "n_train": int(train_mask.sum()),
            "alpha": float(ridge.alpha_),
        }

    return lodo_predictions


def build_exp4_rankings():
    """Build exp4-only rankings for all 15 descriptors per object type.

    Loads from exp4_final_recommendations.json and exp4_optimal_dimensions.json.
    Returns TOP_PERFORMERS-style dict (already in rules_data.py, but saved as
    a standalone asset for reproducibility).
    """
    from topoagent.skills.rules_data import TOP_PERFORMERS, OBJECT_TYPES
    # TOP_PERFORMERS in rules_data.py already has all 15 descriptors from exp4
    return dict(TOP_PERFORMERS)


def build_meta_learner_train_test(accuracy_lookup, cheap_features):
    """Train RidgeCV on 5 exp4 datasets with descriptor one-hot, predict 21 test.

    Key fix from v1: Features = [25 cheap features] + [15 descriptor one-hot] = 40D.
    This gives the model descriptor identity so it can learn different accuracy
    patterns for different descriptors (fixing the degeneracy bug).

    Training data: 5 exp4 datasets × 15 descriptors = 75 samples
    Targets: exp4 accuracies (TabPFN)
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import GradientBoostingRegressor

    exp7_config, _ = _load_exp7_modules()
    FEATURE_NAMES = exp7_config.FEATURE_NAMES

    # Load exp4 accuracies (TabPFN, optimal dimensions)
    exp4_path = PROJECT_ROOT / "results" / "benchmark3" / "exp4" / "exp4_optimal_dimensions.json"
    with open(exp4_path, "r") as f:
        exp4_dims = json.load(f)

    # Also load parameter-tuned accuracies for descriptors where tuning matters
    exp4_recs_path = PROJECT_ROOT / "results" / "benchmark3" / "exp4" / "exp4_final_recommendations.json"
    with open(exp4_recs_path, "r") as f:
        exp4_recs = json.load(f)

    # Build best accuracy per (dataset, descriptor) from exp4
    # Use max of dim-search and param-tuned accuracy
    TRAIN_DATASETS = ["BloodMNIST", "PathMNIST", "RetinaMNIST", "DermaMNIST", "OrganAMNIST"]
    TEST_DATASETS = [ds for ds in sorted(accuracy_lookup.keys()) if ds not in TRAIN_DATASETS]

    # Build exp4 accuracy lookup
    exp4_accs = {}
    param_rules = exp4_recs.get("parameter_rules", {})
    DATASET_TO_OT = {
        "BloodMNIST": "discrete_cells", "PathMNIST": "glands_lumens",
        "RetinaMNIST": "vessel_trees", "DermaMNIST": "surface_lesions",
        "OrganAMNIST": "organ_shape",
    }

    for ds in TRAIN_DATASETS:
        if ds not in exp4_dims or ds == "metadata":
            continue
        ot = DATASET_TO_OT[ds]
        exp4_accs[ds] = {}
        for desc in EXTENDED_DESCRIPTORS:
            # Dim search accuracy
            dim_acc = exp4_dims.get(ds, {}).get(desc, {}).get("accuracy", 0)
            # Param-tuned accuracy (may be higher or lower)
            param_acc = (param_rules.get(desc, {})
                        .get("per_object_type", {})
                        .get(ot, {})
                        .get("accuracy", 0))
            exp4_accs[ds][desc] = max(dim_acc, param_acc) if param_acc > 0 else dim_acc

    # Descriptor one-hot encoding
    desc_to_idx = {d: i for i, d in enumerate(EXTENDED_DESCRIPTORS)}
    n_desc = len(EXTENDED_DESCRIPTORS)

    def make_feature_vec(ds_features, descriptor):
        """Build 40D feature vector: [25 cheap features] + [15 descriptor one-hot]."""
        cheap = [ds_features[fn] for fn in FEATURE_NAMES]
        onehot = [0.0] * n_desc
        onehot[desc_to_idx[descriptor]] = 1.0
        return cheap + onehot

    # Build training data from exp4 datasets
    train_X, train_y, train_meta = [], [], []
    for ds in TRAIN_DATASETS:
        if cheap_features.get(ds) is None or ds not in exp4_accs:
            print(f"  WARNING: Skipping {ds} — missing features or exp4 data")
            continue
        feat = cheap_features[ds]
        for desc in EXTENDED_DESCRIPTORS:
            acc = exp4_accs[ds].get(desc)
            if acc is not None and acc > 0:
                train_X.append(make_feature_vec(feat, desc))
                train_y.append(acc)
                train_meta.append((ds, desc))

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    print(f"  Train/test meta-learner: {len(train_X)} training samples "
          f"({len(TRAIN_DATASETS)} datasets × {n_desc} descriptors), "
          f"{len(train_X[0])} features (25 cheap + {n_desc} descriptor one-hot)")

    # Verify non-degeneracy: LOO on training data
    print("  LOO validation on training datasets:")
    for held_ds in TRAIN_DATASETS:
        if cheap_features.get(held_ds) is None:
            continue
        loo_mask = np.array([m[0] != held_ds for m in train_meta])
        X_tr, y_tr = train_X[loo_mask], train_y[loo_mask]
        X_te = train_X[~loo_mask]
        te_descs = [train_meta[i][1] for i in np.where(~loo_mask)[0]]

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_tr, y_tr)
        pred_scores = ridge.predict(X_te)
        scores = dict(zip(te_descs, pred_scores))

        # Check: are predictions different for different descriptors?
        unique_preds = len(set(f"{s:.4f}" for s in pred_scores))
        best_pred = max(scores, key=scores.get)
        actual_best = max(
            ((d, exp4_accs.get(held_ds, {}).get(d, 0)) for d in EXTENDED_DESCRIPTORS),
            key=lambda x: x[1]
        )[0]
        print(f"    {held_ds}: predicted={best_pred}, actual={actual_best}, "
              f"unique_scores={unique_preds}/{len(pred_scores)}")

    # Train final model on ALL training data
    ridge_final = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_final.fit(train_X, train_y)
    print(f"  Final RidgeCV alpha={ridge_final.alpha_:.2f}")

    # Predict for all test datasets
    predictions = {}
    for ds in TEST_DATASETS:
        if cheap_features.get(ds) is None:
            predictions[ds] = {
                "predicted": "persistence_statistics",
                "predicted_score": 0.0,
                "scores": {},
                "n_train": len(train_X),
                "note": "missing features",
            }
            continue

        feat = cheap_features[ds]
        scores = {}
        for desc in EXTENDED_DESCRIPTORS:
            fv = make_feature_vec(feat, desc)
            score = float(ridge_final.predict([fv])[0])
            scores[desc] = score

        best_desc = max(scores, key=scores.get)
        predictions[ds] = {
            "predicted": best_desc,
            "predicted_score": float(scores[best_desc]),
            "scores": scores,
            "n_train": len(train_X),
            "alpha": float(ridge_final.alpha_),
        }

    return predictions


def build_rule_based_predictions(cheap_features):
    """Apply 4 hand-crafted rules from exp7 to each dataset's features."""
    predictions = {}
    for ds, feat in cheap_features.items():
        if feat is None:
            predictions[ds] = {"predicted": "persistence_statistics", "rule": "fallback (no features)"}
            continue

        # 4 rules from exp7 decision tree
        if feat["fg_bg_contrast"] >= 0.326755 and feat["intensity_skewness"] < -0.627868:
            pred = "betti_curves"
            rule = "Rule 1: high contrast + negative skew → discrete objects"
        elif feat["largest_component_ratio"] >= 0.977793:
            pred = "persistence_statistics"
            rule = "Rule 2: dominant component → diffuse/scattered"
        elif feat["intensity_mean"] < 0.195928:
            pred = "template_functions"
            rule = "Rule 3: dark images → multi-scale/dense"
        else:
            pred = "ATOL"
            rule = "Rule 4: default"

        predictions[ds] = {"predicted": pred, "rule": rule}

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Pre-compute TopoBenchmark assets")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip cheap feature computation (use existing if available)")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    lodo_dir = ASSETS_DIR / "top_performers_lodo"
    lodo_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pre-computing TopoBenchmark Assets")
    print("=" * 60)

    # 1. Load benchmark4 CSV → accuracy lookup
    print("\n[1/9] Building accuracy lookup from benchmark4 CSV...")
    rows = load_benchmark4_csv()
    accuracy_lookup = build_accuracy_lookup(rows)
    with open(ASSETS_DIR / "accuracy_lookup.json", "w") as f:
        json.dump(accuracy_lookup, f, indent=2)
    print(f"  Saved: {len(accuracy_lookup)} datasets, "
          f"{sum(len(v) for v in accuracy_lookup.values())} total entries")

    # 2. Build oracle
    print("\n[2/9] Building oracle (best descriptor per dataset)...")
    oracle = build_oracle(accuracy_lookup)
    with open(ASSETS_DIR / "oracle.json", "w") as f:
        json.dump(oracle, f, indent=2)
    for ds, info in sorted(oracle.items()):
        print(f"  {ds}: {info['descriptor']} ({info['accuracy']:.3f})")

    # 3. Build SBA
    print("\n[3/9] Building SBA (single best algorithm)...")
    sba = build_sba(accuracy_lookup)
    with open(ASSETS_DIR / "sba.json", "w") as f:
        json.dump(sba, f, indent=2)
    print(f"  SBA: {sba['descriptor']} (mean={sba['mean_accuracy']:.3f}, "
          f"{sba['n_datasets']}/{sba['total_datasets']} datasets)")

    # 4. Build TOP_PERFORMERS from benchmark4
    print("\n[4/9] Building top performers per object type...")
    top_performers = build_top_performers(rows, accuracy_lookup)
    with open(ASSETS_DIR / "top_performers.json", "w") as f:
        json.dump(top_performers, f, indent=2)
    for ot, rankings in top_performers.items():
        top3 = rankings[:3]
        top3_str = ", ".join(f"{r['descriptor']}({r['mean_accuracy']:.3f})" for r in top3)
        print(f"  {ot} ({len(rankings)} desc): {top3_str}")

    # 5. Build LODO variants
    print("\n[5/9] Building LODO variants of top performers...")
    lodo_variants = build_top_performers_lodo(rows, accuracy_lookup)
    for ds, tp in lodo_variants.items():
        with open(lodo_dir / f"{ds}.json", "w") as f:
            json.dump(tp, f, indent=2)
    print(f"  Saved {len(lodo_variants)} LODO variants")

    # 6. Compute cheap features
    cheap_features_path = ASSETS_DIR / "cheap_features.json"
    if args.skip_features and cheap_features_path.exists():
        print("\n[6/9] Loading existing cheap features (--skip-features)...")
        with open(cheap_features_path, "r") as f:
            cheap_features = json.load(f)
        print(f"  Loaded features for {len(cheap_features)} datasets")
    else:
        print("\n[6/9] Computing cheap features for all datasets...")
        cheap_features = compute_all_cheap_features(accuracy_lookup, n_samples=500)
        with open(cheap_features_path, "w") as f:
            json.dump(cheap_features, f, indent=2)
        n_ok = sum(1 for v in cheap_features.values() if v is not None)
        print(f"  Saved features for {n_ok}/{len(cheap_features)} datasets")

    # 7. Meta-learner LODO (GBR) + feature importance
    print("\n[7/9] Training GBR meta-learner in LODO fashion...")
    meta_lodo, feature_importance = build_meta_learner_lodo(accuracy_lookup, cheap_features)
    with open(ASSETS_DIR / "meta_learner_lodo.json", "w") as f:
        json.dump(meta_lodo, f, indent=2)
    with open(ASSETS_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    print(f"  Saved GBR LODO predictions for {len(meta_lodo)} datasets")
    for ds in list(meta_lodo.keys())[:5]:
        pred = meta_lodo[ds]
        print(f"    {ds}: predicted={pred['predicted']} (score={pred['predicted_score']:.3f})")

    # Print feature importance summary
    print(f"\n  Feature importance (top 10, averaged over {feature_importance['n_folds']} LODO folds):")
    for entry in feature_importance["ranked_features"][:10]:
        print(f"    {entry['rank']:2d}. {entry['feature']:<25s} {entry['mean_importance']:.4f} "
              f"(±{entry['std_importance']:.4f}, {entry['pct_total']:.1f}%)")
    print(f"\n  Category breakdown:")
    for cat, stats in feature_importance["category_importance"].items():
        print(f"    {cat:<18s} {stats['pct_total']:5.1f}% ({stats['n_features']} features)")

    # 8. Meta-learner LODO (RidgeCV) — simpler sanity check
    print("\n[8/9] Training RidgeCV meta-learner in LODO fashion...")
    ridge_lodo = build_meta_learner_ridge_lodo(accuracy_lookup, cheap_features)
    with open(ASSETS_DIR / "meta_learner_ridge_lodo.json", "w") as f:
        json.dump(ridge_lodo, f, indent=2)
    print(f"  Saved RidgeCV LODO predictions for {len(ridge_lodo)} datasets")
    # Compare GBR vs Ridge agreement
    agree = sum(1 for ds in meta_lodo if ds in ridge_lodo
                and meta_lodo[ds]["predicted"] == ridge_lodo[ds]["predicted"])
    total = sum(1 for ds in meta_lodo if ds in ridge_lodo)
    print(f"  GBR vs RidgeCV agreement: {agree}/{total} datasets ({agree/max(total,1):.0%})")

    # 9. Rule-based predictions
    print("\n[9/11] Computing rule-based predictions...")
    rule_preds = build_rule_based_predictions(cheap_features)
    with open(ASSETS_DIR / "rule_based_predictions.json", "w") as f:
        json.dump(rule_preds, f, indent=2)
    for ds, pred in sorted(rule_preds.items()):
        print(f"  {ds}: {pred['predicted']} ({pred['rule']})")

    # 10. Train/test meta-learner (5 exp4 train → 21 test, with descriptor one-hot)
    print("\n[10/11] Training train/test meta-learner (RidgeCV + descriptor one-hot)...")
    tt_predictions = build_meta_learner_train_test(accuracy_lookup, cheap_features)
    with open(ASSETS_DIR / "meta_learner_train_test.json", "w") as f:
        json.dump(tt_predictions, f, indent=2)
    # Also save as ridge variant (same model for now)
    with open(ASSETS_DIR / "meta_learner_ridge_train_test.json", "w") as f:
        json.dump(tt_predictions, f, indent=2)
    print(f"  Saved train/test predictions for {len(tt_predictions)} test datasets")
    for ds in list(tt_predictions.keys())[:5]:
        pred = tt_predictions[ds]
        print(f"    {ds}: predicted={pred['predicted']} (score={pred['predicted_score']:.3f})")

    # 11. Exp4 rankings (all 15 descriptors per type)
    print("\n[11/11] Building exp4-only rankings...")
    exp4_rankings = build_exp4_rankings()
    with open(ASSETS_DIR / "exp4_rankings.json", "w") as f:
        json.dump(exp4_rankings, f, indent=2)
    for ot, rankings in exp4_rankings.items():
        top3 = rankings[:3]
        top3_str = ", ".join(f"{r['descriptor']}({r['accuracy']:.3f})" for r in top3)
        print(f"  {ot}: {top3_str}")

    print("\n" + "=" * 60)
    print("All assets saved to:")
    print(f"  {ASSETS_DIR}")
    print("=" * 60)

    # Verify
    expected_files = [
        "accuracy_lookup.json", "oracle.json", "sba.json",
        "top_performers.json", "cheap_features.json",
        "meta_learner_lodo.json", "meta_learner_ridge_lodo.json",
        "feature_importance.json", "rule_based_predictions.json",
        "meta_learner_train_test.json", "meta_learner_ridge_train_test.json",
        "exp4_rankings.json",
    ]
    for f in expected_files:
        path = ASSETS_DIR / f
        size = path.stat().st_size if path.exists() else 0
        status = "OK" if size > 0 else "MISSING"
        print(f"  [{status}] {f} ({size:,} bytes)")

    lodo_count = len(list(lodo_dir.glob("*.json")))
    print(f"  [{'OK' if lodo_count > 0 else 'MISSING'}] top_performers_lodo/ ({lodo_count} files)")


if __name__ == "__main__":
    main()
