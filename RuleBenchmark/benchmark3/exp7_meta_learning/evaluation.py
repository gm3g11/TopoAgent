"""LODO (Leave-One-Dataset-Out) evaluation and metrics."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import N_FEATURES, DESCRIPTORS
from .models import MetaLearner


def compute_metrics(y_true, y_pred, descriptors):
    """Compute per-dataset metrics for a single LODO fold.

    Args:
        y_true: (n_desc,) true balanced_accuracy scores for this dataset
        y_pred: (n_desc,) predicted scores
        descriptors: list of descriptor names (same order as y_true/y_pred)

    Returns:
        dict with: regret, top1_hit, top2_hit, top3_hit, spearman_rho,
                   oracle_best, oracle_score, predicted_best, predicted_score
    """
    n = len(y_true)

    # Rank by predicted score (descending)
    pred_ranking = np.argsort(-y_pred)
    true_ranking = np.argsort(-y_true)

    # Oracle best (true best descriptor)
    oracle_idx = true_ranking[0]
    oracle_score = y_true[oracle_idx]
    oracle_best = descriptors[oracle_idx]

    # Predicted best
    pred_best_idx = pred_ranking[0]
    predicted_score = y_true[pred_best_idx]  # Actual score of predicted best
    predicted_best = descriptors[pred_best_idx]

    # Regret: oracle_score - actual_score_of_predicted_best
    regret = oracle_score - predicted_score

    # Top-k hit: is true best in top-k predicted?
    true_best_set = {true_ranking[0]}
    top1_hit = int(pred_ranking[0] in true_best_set)

    # For top-2, top-3: is predicted top-1 within top-k of true ranking?
    top2_hit = int(pred_best_idx in true_ranking[:2])
    top3_hit = int(pred_best_idx in true_ranking[:3])

    # Spearman correlation between predicted and true rankings
    if n >= 3:
        rho, _ = spearmanr(y_true, y_pred)
        rho = float(np.nan_to_num(rho, nan=0.0))
    else:
        rho = 0.0

    return {
        "regret": float(regret),
        "top1_hit": top1_hit,
        "top2_hit": top2_hit,
        "top3_hit": top3_hit,
        "spearman_rho": rho,
        "oracle_best": oracle_best,
        "oracle_score": float(oracle_score),
        "predicted_best": predicted_best,
        "predicted_score": float(predicted_score),
        "n_descriptors": n,
    }


def lodo_evaluation(X, y, meta_df, gbr_params=None, rf_params=None,
                    tree_params=None, n_rf_seeds=10):
    """Leave-One-Dataset-Out cross-validation.

    For each of 13 datasets:
      - Train on the other 12 datasets (~177-180 rows)
      - Predict on the held-out dataset (~12-15 rows)
      - Compute metrics

    Args:
        X: (n_rows, 40) training features
        y: (n_rows,) scores
        meta_df: DataFrame with 'dataset' and 'descriptor' columns

    Returns:
        list of dicts, one per dataset, with metrics + predictions
    """
    datasets = meta_df["dataset"].unique()
    results = []

    print(f"[evaluation] Running LODO CV with {len(datasets)} folds...")

    for i, held_out_dataset in enumerate(sorted(datasets)):
        # Split
        test_mask = meta_df["dataset"] == held_out_dataset
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_descriptors = meta_df.loc[test_mask, "descriptor"].values.tolist()

        # Fit model (scaler inside loop - no leakage)
        model = MetaLearner(
            gbr_params=gbr_params,
            rf_params=rf_params,
            tree_params=tree_params,
            n_rf_seeds=n_rf_seeds,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_rf, y_std_rf = model.predict_with_uncertainty(X_test)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, test_descriptors)
        metrics["dataset"] = held_out_dataset
        metrics["n_train"] = int(train_mask.sum())
        metrics["n_test"] = int(test_mask.sum())
        metrics["mean_uncertainty"] = float(y_std_rf.mean())

        # Store predictions for analysis
        metrics["predictions"] = {
            d: {"true": float(t), "pred": float(p), "std": float(s)}
            for d, t, p, s in zip(test_descriptors, y_test, y_pred, y_std_rf)
        }

        results.append(metrics)

        print(f"  [{i+1}/{len(datasets)}] {held_out_dataset}: "
              f"regret={metrics['regret']:.4f}, top1={metrics['top1_hit']}, "
              f"rho={metrics['spearman_rho']:.3f}, "
              f"oracle={metrics['oracle_best']}, pred={metrics['predicted_best']}")

    return results


def compute_aggregate_metrics(lodo_results):
    """Aggregate LODO results into summary metrics.

    Returns:
        dict with mean/std of all metrics + baselines
    """
    regrets = [r["regret"] for r in lodo_results]
    top1_hits = [r["top1_hit"] for r in lodo_results]
    top2_hits = [r["top2_hit"] for r in lodo_results]
    top3_hits = [r["top3_hit"] for r in lodo_results]
    rhos = [r["spearman_rho"] for r in lodo_results]

    summary = {
        "n_datasets": len(lodo_results),
        "mean_regret": float(np.mean(regrets)),
        "std_regret": float(np.std(regrets)),
        "median_regret": float(np.median(regrets)),
        "max_regret": float(np.max(regrets)),
        "top1_accuracy": float(np.mean(top1_hits)),
        "top2_accuracy": float(np.mean(top2_hits)),
        "top3_accuracy": float(np.mean(top3_hits)),
        "mean_spearman_rho": float(np.mean(rhos)),
        "std_spearman_rho": float(np.std(rhos)),
        "mean_uncertainty": float(np.mean([r["mean_uncertainty"] for r in lodo_results])),
    }

    return summary


def compute_baselines(lodo_results, performance_df):
    """Compute baseline strategies for comparison.

    Baselines:
        - random: Pick a descriptor uniformly at random
        - always_best_overall: Always pick the single descriptor that is best on average

    Args:
        lodo_results: list of per-dataset LODO results
        performance_df: Full performance DataFrame

    Returns:
        dict with baseline metrics
    """
    # Random baseline: expected regret = oracle - mean(scores)
    random_regrets = []
    for result in lodo_results:
        preds = result["predictions"]
        scores = [v["true"] for v in preds.values()]
        oracle = max(scores)
        random_expected = np.mean(scores)
        random_regrets.append(oracle - random_expected)

    # Always-best-overall: find descriptor with highest mean across training datasets
    # For each LODO fold, compute what the "always best" strategy would pick
    always_best_regrets = []
    for result in lodo_results:
        held_out = result["dataset"]
        # Training data: all datasets except held_out
        train_df = performance_df[performance_df["dataset"] != held_out]
        # Mean score per descriptor across training datasets
        desc_means = train_df.groupby("descriptor")["score"].mean()
        best_overall = desc_means.idxmax()

        # What score does this "always best" get on the held-out dataset?
        preds = result["predictions"]
        if best_overall in preds:
            actual_score = preds[best_overall]["true"]
        else:
            # Descriptor not available for this dataset (e.g., MURA)
            actual_score = np.mean([v["true"] for v in preds.values()])

        always_best_regrets.append(result["oracle_score"] - actual_score)

    # Always-ATOL: ATOL wins 6/13 datasets, so this is a strong fixed baseline
    always_atol_regrets = []
    for result in lodo_results:
        preds = result["predictions"]
        if "ATOL" in preds:
            actual_score = preds["ATOL"]["true"]
        else:
            actual_score = np.mean([v["true"] for v in preds.values()])
        always_atol_regrets.append(result["oracle_score"] - actual_score)

    baselines = {
        "random": {
            "mean_regret": float(np.mean(random_regrets)),
            "std_regret": float(np.std(random_regrets)),
        },
        "always_best_overall": {
            "mean_regret": float(np.mean(always_best_regrets)),
            "std_regret": float(np.std(always_best_regrets)),
        },
        "always_ATOL": {
            "mean_regret": float(np.mean(always_atol_regrets)),
            "std_regret": float(np.std(always_atol_regrets)),
        },
    }

    return baselines


def results_to_dataframe(lodo_results):
    """Convert LODO results to a DataFrame for CSV output.

    Returns:
        DataFrame with one row per dataset and columns for all metrics.
    """
    rows = []
    for r in lodo_results:
        row = {
            "dataset": r["dataset"],
            "regret": r["regret"],
            "top1_hit": r["top1_hit"],
            "top2_hit": r["top2_hit"],
            "top3_hit": r["top3_hit"],
            "spearman_rho": r["spearman_rho"],
            "oracle_best": r["oracle_best"],
            "oracle_score": r["oracle_score"],
            "predicted_best": r["predicted_best"],
            "predicted_score": r["predicted_score"],
            "n_descriptors": r["n_descriptors"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "mean_uncertainty": r["mean_uncertainty"],
        }
        rows.append(row)

    return pd.DataFrame(rows)
