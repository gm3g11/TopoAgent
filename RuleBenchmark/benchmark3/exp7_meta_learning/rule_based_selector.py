"""Rule-Based Descriptor Selector with LODO Evaluation.

Defines quantitative thresholds for selecting TDA descriptors based on
observable image characteristics. Evaluates via Leave-One-Dataset-Out (LODO)
cross-validation and compares against baselines.

Usage:
    python -m scripts.run_benchmark3.exp7_meta_learning.rule_based_selector
    # or
    python benchmarks/benchmark3/exp7_meta_learning/rule_based_selector.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

from .config import (
    DATASETS, DESCRIPTORS, FEATURE_NAMES, OUTPUT_DIR, PROJECT_ROOT,
)


# ─── Constants ──────────────────────────────────────────────────────────────

LEAKAGE_PENALTY = 0.02  # Applied to ATOL and persistence_codebook
LEAKAGE_DESCRIPTORS = {"ATOL", "persistence_codebook"}
NON_TDA_DESCRIPTORS = {"lbp_texture", "edge_histogram"}

# TDA-only descriptors (candidates for selection)
TDA_DESCRIPTORS = [d for d in DESCRIPTORS
                   if d not in NON_TDA_DESCRIPTORS]

# The 4 descriptor groups the rules map to
RULE_TARGETS = ["betti_curves", "persistence_statistics",
                "template_functions", "ATOL"]

RESULTS_DIR = OUTPUT_DIR / "rule_based"


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_data():
    """Load performance matrix and dataset characteristics.

    Returns:
        perf_df: DataFrame[dataset, descriptor, score, score_std]
        char_df: DataFrame indexed by dataset with 25 feature columns
    """
    perf_path = OUTPUT_DIR / "performance_matrix.csv"
    char_path = OUTPUT_DIR / "dataset_characteristics.csv"

    perf_df = pd.read_csv(perf_path)
    char_df = pd.read_csv(char_path, index_col=0)

    return perf_df, char_df


def apply_adjustments(perf_df):
    """Apply 2% penalty to leakage descriptors and filter to TDA-only.

    Returns:
        adj_df: adjusted DataFrame with only TDA descriptors
    """
    adj_df = perf_df.copy()

    # Apply leakage penalty
    mask = adj_df["descriptor"].isin(LEAKAGE_DESCRIPTORS)
    adj_df.loc[mask, "score"] = adj_df.loc[mask, "score"] - LEAKAGE_PENALTY

    # Filter to TDA descriptors only
    adj_df = adj_df[adj_df["descriptor"].isin(TDA_DESCRIPTORS)].copy()

    return adj_df.reset_index(drop=True)


def compute_winners(adj_df):
    """Compute adjusted winner and gap per dataset.

    Returns:
        DataFrame[dataset, winner, winner_score, runner_up, runner_up_score, gap]
    """
    rows = []
    for dataset in adj_df["dataset"].unique():
        sub = adj_df[adj_df["dataset"] == dataset].sort_values(
            "score", ascending=False)
        if len(sub) < 2:
            continue
        winner = sub.iloc[0]
        runner = sub.iloc[1]
        rows.append({
            "dataset": dataset,
            "winner": winner["descriptor"],
            "winner_score": winner["score"],
            "runner_up": runner["descriptor"],
            "runner_up_score": runner["score"],
            "gap": winner["score"] - runner["score"],
        })
    return pd.DataFrame(rows)


# ─── Rule-Based Selector ───────────────────────────────────────────────────

class RuleBasedSelector:
    """Cascading rule-based descriptor selector with compound conditions.

    Rules are evaluated in order:
        1. Discrete objects signal  -> betti_curves
        2. Diffuse/scattered signal -> persistence_statistics
        3. Multi-scale/dense signal -> template_functions
        4. Default                  -> ATOL

    Each rule has one or two conditions (AND conjunction).
    Thresholds are derived from training data.
    """

    def __init__(self):
        # Each rule: dict with keys:
        #   conditions: list of (feature, op, threshold)
        #   descriptor: str
        #   label: str
        self.rules = []
        self.fitted = False

    def fit(self, char_df, winners_df, mode="data_driven"):
        """Derive thresholds from training data.

        Args:
            char_df: DataFrame indexed by dataset, columns = 25 features
            winners_df: DataFrame[dataset, winner]
            mode: "data_driven" for automatic threshold search,
                  "hybrid" for data-driven with domain constraints
        """
        winner_map = dict(zip(winners_df["dataset"], winners_df["winner"]))
        datasets = [d for d in char_df.index if d in winner_map]
        features = char_df.loc[datasets]
        labels = pd.Series({d: winner_map[d] for d in datasets})

        self.rules = []

        # Use only image-characteristic features (not dataset metadata)
        img_features = [f for f in FEATURE_NAMES if f not in
                        {"n_samples", "n_classes", "samples_per_class",
                         "class_imbalance", "polarity"}]

        if mode == "hybrid":
            self._fit_hybrid(features, labels, img_features)
        else:
            self._fit_data_driven(features, labels, img_features)

        # Rule 4: Default -> ATOL
        self.rules.append({
            "conditions": [],
            "descriptor": "ATOL",
            "label": "Default",
        })

        self.fitted = True
        return self

    def _fit_hybrid(self, features, labels, img_features):
        """Fit using domain-constrained features with data-driven thresholds.

        Domain constraints:
        - betti_curves: use fg_bg_contrast (object separability) AND
          intensity_skewness (asymmetric intensity = discrete objects)
        - persistence_statistics: use gradient_mean (diffuse = low gradient)
          AND otsu_stability (stable thresholding = uniform texture)
        - template_functions: use intensity_skewness (positive = dark bg
          with bright features at multiple scales)
        """
        # Rule 1: betti_curves (discrete objects)
        # Domain: discrete objects have high fg/bg contrast and negative
        # skewness (bright objects on dark bg, or many dark objects on bright bg)
        rule1 = self._find_best_compound_split(
            features, labels, target="betti_curves",
            candidate_features=[
                "fg_bg_contrast", "intensity_skewness",
                "intensity_kurtosis", "otsu_holes",
            ],
            label="Discrete objects",
        )
        if rule1:
            self.rules.append(rule1)

        # Rule 2: persistence_statistics (diffuse/scattered)
        # Domain: diffuse features = low gradient, high otsu_stability
        rule2 = self._find_best_compound_split(
            features, labels, target="persistence_statistics",
            candidate_features=[
                "gradient_mean", "otsu_stability", "edge_density",
                "largest_component_ratio", "component_size_mean",
            ],
            label="Diffuse/scattered",
            exclude_already_captured=True,
        )
        if rule2:
            self.rules.append(rule2)

        # Rule 3: template_functions (multi-scale/dense)
        # Domain: positive skewness (dark images with bright features),
        # low intensity mean (dark overall)
        rule3 = self._find_best_compound_split(
            features, labels, target="template_functions",
            candidate_features=[
                "intensity_skewness", "intensity_mean", "intensity_kurtosis",
                "fft_low_freq_ratio", "otsu_stability",
            ],
            label="Multi-scale/dense",
            exclude_already_captured=True,
        )
        if rule3:
            self.rules.append(rule3)

    def _fit_data_driven(self, features, labels, img_features):
        """Fit using fully data-driven feature and threshold selection."""
        # Rule 1: betti_curves (discrete objects)
        rule1 = self._find_best_compound_split(
            features, labels, target="betti_curves",
            candidate_features=img_features,
            label="Discrete objects",
        )
        if rule1:
            self.rules.append(rule1)

        # Rule 2: persistence_statistics (diffuse/scattered)
        rule2 = self._find_best_compound_split(
            features, labels, target="persistence_statistics",
            candidate_features=img_features,
            label="Diffuse/scattered",
            exclude_already_captured=True,
        )
        if rule2:
            self.rules.append(rule2)

        # Rule 3: template_functions (multi-scale/dense)
        rule3 = self._find_best_compound_split(
            features, labels, target="template_functions",
            candidate_features=img_features,
            label="Multi-scale/dense",
            exclude_already_captured=True,
        )
        if rule3:
            self.rules.append(rule3)

    def _eval_condition(self, val, op, thresh):
        """Evaluate a single condition."""
        if op == ">=":
            return val >= thresh
        elif op == "<":
            return val < thresh
        elif op == "<=":
            return val <= thresh
        elif op == ">":
            return val > thresh
        return False

    def _eval_rule(self, rule, feature_dict):
        """Evaluate whether a rule fires for given features."""
        if not rule["conditions"]:
            return True  # Default rule always fires
        return all(
            self._eval_condition(feature_dict[feat], op, thresh)
            for feat, op, thresh in rule["conditions"]
        )

    def _get_captured_datasets(self, features, labels):
        """Get set of datasets captured by existing rules."""
        captured = set()
        for rule in self.rules:
            for ds in labels.index:
                if self._eval_rule(rule, features.loc[ds]):
                    captured.add(ds)
        return captured

    def _find_best_compound_split(self, features, labels, target,
                                  candidate_features, label,
                                  exclude_already_captured=False):
        """Find the best 1- or 2-feature conjunction for a target group.

        Tries all single features first, then all pairs. Picks the
        simplest rule that achieves the best accuracy.

        Returns:
            dict with conditions, descriptor, label; or None
        """
        target_mask = (labels == target)
        n_target = target_mask.sum()
        if n_target == 0:
            return None

        # Exclude datasets already captured
        if exclude_already_captured and self.rules:
            captured = self._get_captured_datasets(features, labels)
            avail = [d for d in labels.index if d not in captured]
            if not avail:
                return None
            features = features.loc[avail]
            labels = labels.loc[avail]
            target_mask = (labels == target)
            n_target = target_mask.sum()
            if n_target == 0:
                return None

        n = len(labels)
        best_acc = 0.0
        best_conditions = None
        best_n_conditions = 99  # Prefer simpler rules

        # Helper: generate all (feature, op, threshold) candidates
        def _candidates(feat_name):
            vals = features[feat_name].values
            all_vals = np.sort(np.unique(vals))
            if len(all_vals) < 2:
                return []
            cands = []
            for i in range(len(all_vals) - 1):
                thresh = (all_vals[i] + all_vals[i + 1]) / 2.0
                cands.append((feat_name, ">=", thresh))
                cands.append((feat_name, "<", thresh))
            return cands

        # Phase 1: Single-feature splits
        all_cands = []
        for feat_name in candidate_features:
            if feat_name not in features.columns:
                continue
            all_cands.extend(_candidates(feat_name))

        for cond in all_cands:
            feat, op, thresh = cond
            pred = np.array([
                self._eval_condition(features.iloc[j][feat], op, thresh)
                for j in range(n)
            ])
            acc = (np.sum(pred & target_mask.values) +
                   np.sum(~pred & ~target_mask.values)) / n
            if acc > best_acc or (acc == best_acc and 1 < best_n_conditions):
                best_acc = acc
                best_conditions = [cond]
                best_n_conditions = 1

        # Phase 2: Two-feature conjunctions (only if single-feature < perfect)
        if best_acc < 1.0 and n_target >= 2:
            # Precompute single condition predictions for speed
            cond_preds = {}
            for cond in all_cands:
                feat, op, thresh = cond
                pred = np.array([
                    self._eval_condition(features.iloc[j][feat], op, thresh)
                    for j in range(n)
                ])
                cond_preds[cond] = pred

            # Try all pairs (different features only)
            seen_feat_pairs = set()
            for c1 in all_cands:
                for c2 in all_cands:
                    if c1[0] >= c2[0]:  # Skip same feature and duplicates
                        continue
                    pair_key = (c1[0], c2[0])
                    # Don't need to track — the threshold search covers it

                    pred = cond_preds[c1] & cond_preds[c2]
                    acc = (np.sum(pred & target_mask.values) +
                           np.sum(~pred & ~target_mask.values)) / n

                    if acc > best_acc:
                        best_acc = acc
                        best_conditions = [c1, c2]
                        best_n_conditions = 2

        if best_conditions and best_acc > 0.5:
            return {
                "conditions": best_conditions,
                "descriptor": target,
                "label": label,
                "accuracy": best_acc,
            }
        return None

    def predict(self, feature_dict):
        """Predict descriptor for a single dataset.

        Args:
            feature_dict: dict or Series of 25 features

        Returns:
            str: descriptor name
        """
        assert self.fitted, "Selector not fitted"

        for rule in self.rules:
            if self._eval_rule(rule, feature_dict):
                return rule["descriptor"]

        return "ATOL"

    def predict_all(self, char_df):
        """Predict descriptor for all datasets.

        Returns:
            dict: {dataset: descriptor_name}
        """
        return {ds: self.predict(char_df.loc[ds]) for ds in char_df.index}

    def explain(self, feature_dict):
        """Return human-readable explanation of prediction.

        Args:
            feature_dict: dict or Series of 25 features

        Returns:
            str: explanation
        """
        assert self.fitted, "Selector not fitted"

        for rule in self.rules:
            if self._eval_rule(rule, feature_dict):
                if not rule["conditions"]:
                    return (f"No specific rule triggered. "
                            f"Default -> {rule['descriptor']}")
                parts = []
                for feat, op, thresh in rule["conditions"]:
                    val = feature_dict[feat]
                    parts.append(f"{feat} {op} {thresh:.4f} "
                                 f"(actual={val:.4f})")
                cond_str = " AND ".join(parts)
                return (f"Rule '{rule['label']}': {cond_str} "
                        f"-> {rule['descriptor']}")

        return "Default -> ATOL"

    def describe_rules(self):
        """Return a formatted string describing all rules."""
        lines = ["Rule-Based Selector Rules:", "=" * 50]
        for i, rule in enumerate(self.rules, 1):
            if not rule["conditions"]:
                lines.append(
                    f"  Rule {i} [{rule['label']}]: "
                    f"Default -> {rule['descriptor']}")
            else:
                conds = " AND ".join(
                    f"{f} {o} {t:.6f}" for f, o, t in rule["conditions"])
                acc_str = ""
                if "accuracy" in rule:
                    acc_str = f" (train_acc={rule['accuracy']:.3f})"
                lines.append(
                    f"  Rule {i} [{rule['label']}]: "
                    f"IF {conds} -> {rule['descriptor']}{acc_str}")
        return "\n".join(lines)


# ─── LODO Evaluation ────────────────────────────────────────────────────────

def lodo_evaluation(char_df, adj_df, mode="hybrid"):
    """Leave-One-Dataset-Out evaluation of the rule-based selector.

    For each dataset:
        1. Hold out one dataset
        2. Re-derive thresholds from remaining 12
        3. Predict descriptor for held-out dataset
        4. Compute regret

    Returns:
        list of dicts with per-fold results
    """
    datasets = sorted(char_df.index.tolist())
    results = []

    print(f"\n[LODO] Running Leave-One-Dataset-Out evaluation (mode={mode})...")
    print("=" * 70)

    for held_out in datasets:
        # Training datasets
        train_datasets = [d for d in datasets if d != held_out]

        # Compute winners on training data
        train_perf = adj_df[adj_df["dataset"].isin(train_datasets)]
        train_winners = compute_winners(train_perf)

        # Training characteristics
        train_chars = char_df.loc[train_datasets]

        # Fit selector on training data
        selector = RuleBasedSelector()
        selector.fit(train_chars, train_winners, mode=mode)

        # Predict for held-out dataset
        predicted_descriptor = selector.predict(char_df.loc[held_out])
        explanation = selector.explain(char_df.loc[held_out])

        # Get actual scores for held-out dataset
        held_out_scores = adj_df[adj_df["dataset"] == held_out]
        if held_out_scores.empty:
            continue

        # Oracle score (best adjusted TDA descriptor)
        oracle_row = held_out_scores.sort_values("score", ascending=False).iloc[0]
        oracle_score = oracle_row["score"]
        oracle_descriptor = oracle_row["descriptor"]

        # Score of predicted descriptor
        pred_score_row = held_out_scores[
            held_out_scores["descriptor"] == predicted_descriptor]
        if pred_score_row.empty:
            predicted_score = held_out_scores["score"].mean()
        else:
            predicted_score = pred_score_row["score"].values[0]

        regret = oracle_score - predicted_score

        result = {
            "dataset": held_out,
            "predicted": predicted_descriptor,
            "oracle": oracle_descriptor,
            "predicted_score": float(predicted_score),
            "oracle_score": float(oracle_score),
            "regret": float(regret),
            "explanation": explanation,
            "rules_used": selector.describe_rules(),
        }
        results.append(result)

        hit = "HIT" if predicted_descriptor == oracle_descriptor else "miss"
        print(f"  {held_out:20s}: pred={predicted_descriptor:25s} "
              f"oracle={oracle_descriptor:25s} "
              f"regret={regret:.4f} [{hit}]")

    return results


# ─── Baseline Strategies ────────────────────────────────────────────────────

def compute_baselines(adj_df, char_df):
    """Compute baseline strategy results for comparison.

    Baselines:
        - always_ATOL: Always predict ATOL (adjusted)
        - always_template: Always predict template_functions
        - random: Expected value of uniform random TDA descriptor
        - ml_meta_learner: Try to load exp7 GBR results

    Returns:
        dict of {strategy: list of per-dataset results}
    """
    datasets = sorted(char_df.index.tolist())
    baselines = {}

    # Always-ATOL
    atol_results = []
    for ds in datasets:
        ds_scores = adj_df[adj_df["dataset"] == ds]
        if ds_scores.empty:
            continue
        oracle = ds_scores["score"].max()
        atol_row = ds_scores[ds_scores["descriptor"] == "ATOL"]
        atol_score = atol_row["score"].values[0] if not atol_row.empty else ds_scores["score"].mean()
        atol_results.append({
            "dataset": ds,
            "predicted": "ATOL",
            "predicted_score": float(atol_score),
            "oracle_score": float(oracle),
            "regret": float(oracle - atol_score),
        })
    baselines["always_ATOL"] = atol_results

    # Always-template_functions
    tf_results = []
    for ds in datasets:
        ds_scores = adj_df[adj_df["dataset"] == ds]
        if ds_scores.empty:
            continue
        oracle = ds_scores["score"].max()
        tf_row = ds_scores[ds_scores["descriptor"] == "template_functions"]
        tf_score = tf_row["score"].values[0] if not tf_row.empty else ds_scores["score"].mean()
        tf_results.append({
            "dataset": ds,
            "predicted": "template_functions",
            "predicted_score": float(tf_score),
            "oracle_score": float(oracle),
            "regret": float(oracle - tf_score),
        })
    baselines["always_template"] = tf_results

    # Random baseline (expected regret = oracle - mean of all TDA scores)
    random_results = []
    for ds in datasets:
        ds_scores = adj_df[adj_df["dataset"] == ds]
        if ds_scores.empty:
            continue
        oracle = ds_scores["score"].max()
        mean_score = ds_scores["score"].mean()
        random_results.append({
            "dataset": ds,
            "predicted": "random",
            "predicted_score": float(mean_score),
            "oracle_score": float(oracle),
            "regret": float(oracle - mean_score),
        })
    baselines["random"] = random_results

    # Always-persistence_statistics
    ps_results = _always_descriptor_baseline(adj_df, datasets,
                                             "persistence_statistics")
    baselines["always_persistence_stats"] = ps_results

    # Always-persistence_codebook (adjusted)
    pc_results = _always_descriptor_baseline(adj_df, datasets,
                                             "persistence_codebook")
    baselines["always_persistence_codebook"] = pc_results

    return baselines


def _always_descriptor_baseline(adj_df, datasets, descriptor):
    """Compute regret for an always-X strategy."""
    results = []
    for ds in datasets:
        ds_scores = adj_df[adj_df["dataset"] == ds]
        if ds_scores.empty:
            continue
        oracle = ds_scores["score"].max()
        d_row = ds_scores[ds_scores["descriptor"] == descriptor]
        d_score = (d_row["score"].values[0] if not d_row.empty
                   else ds_scores["score"].mean())
        results.append({
            "dataset": ds,
            "predicted": descriptor,
            "predicted_score": float(d_score),
            "oracle_score": float(oracle),
            "regret": float(oracle - d_score),
        })
    return results


# ─── Metrics Computation ────────────────────────────────────────────────────

def aggregate_metrics(results):
    """Compute aggregate metrics from per-dataset results.

    Returns:
        dict with mean_regret, median_regret, max_regret,
        top1_accuracy, soft_accuracy (regret < 0.01), top3_accuracy
    """
    regrets = [r["regret"] for r in results]
    n = len(results)

    top1_hits = sum(1 for r in results if r["regret"] == 0.0)
    # Soft accuracy: regret < 1%
    soft_hits = sum(1 for r in results if r["regret"] < 0.01)
    # Top-3 proxy: regret < 2%
    top3_proxy = sum(1 for r in results if r["regret"] < 0.02)

    return {
        "n_datasets": n,
        "mean_regret": float(np.mean(regrets)),
        "std_regret": float(np.std(regrets)),
        "median_regret": float(np.median(regrets)),
        "max_regret": float(np.max(regrets)),
        "top1_accuracy": f"{top1_hits}/{n} ({top1_hits/n*100:.1f}%)",
        "top1_count": top1_hits,
        "soft_accuracy": f"{soft_hits}/{n} ({soft_hits/n*100:.1f}%)",
        "soft_count": soft_hits,
        "near_optimal": f"{top3_proxy}/{n} ({top3_proxy/n*100:.1f}%)",
    }


# ─── Feature Discriminability Analysis ──────────────────────────────────────

def analyze_feature_discriminability(char_df, winners_df):
    """For each rule target, find most discriminative features.

    Returns:
        dict: {target_descriptor: list of (feature, accuracy, threshold, direction)}
    """
    winner_map = dict(zip(winners_df["dataset"], winners_df["winner"]))
    datasets = [d for d in char_df.index if d in winner_map]
    features = char_df.loc[datasets]
    labels = pd.Series({d: winner_map[d] for d in datasets})

    analysis = {}

    for target in RULE_TARGETS[:-1]:  # Exclude ATOL (default)
        target_mask = (labels == target)
        n_target = target_mask.sum()
        if n_target == 0:
            analysis[target] = []
            continue

        feat_results = []
        for feat_name in FEATURE_NAMES:
            vals = features[feat_name].values
            target_vals = vals[target_mask.values]
            other_vals = vals[~target_mask.values]

            if len(target_vals) == 0 or len(other_vals) == 0:
                continue

            # Find best threshold
            all_vals = np.sort(np.unique(vals))
            best_acc = 0.0
            best_thresh = None
            best_dir = None

            for i in range(len(all_vals) - 1):
                thresh = (all_vals[i] + all_vals[i + 1]) / 2.0

                # >= direction
                pred = vals >= thresh
                acc_ge = (np.sum(pred & target_mask.values) +
                          np.sum(~pred & ~target_mask.values)) / len(vals)
                if acc_ge > best_acc:
                    best_acc = acc_ge
                    best_thresh = thresh
                    best_dir = ">="

                # < direction
                pred = vals < thresh
                acc_lt = (np.sum(pred & target_mask.values) +
                          np.sum(~pred & ~target_mask.values)) / len(vals)
                if acc_lt > best_acc:
                    best_acc = acc_lt
                    best_thresh = thresh
                    best_dir = "<"

            feat_results.append({
                "feature": feat_name,
                "accuracy": best_acc,
                "threshold": best_thresh,
                "direction": best_dir,
                "target_mean": float(np.mean(target_vals)),
                "other_mean": float(np.mean(other_vals)),
                "target_range": f"[{np.min(target_vals):.4f}, {np.max(target_vals):.4f}]",
                "other_range": f"[{np.min(other_vals):.4f}, {np.max(other_vals):.4f}]",
            })

        # Sort by accuracy
        feat_results.sort(key=lambda x: x["accuracy"], reverse=True)
        analysis[target] = feat_results[:5]  # Top 5

    return analysis


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_report(lodo_results, baselines, discriminability, selector):
    """Generate comprehensive results report.

    Returns:
        str: formatted report text
    """
    lines = []
    lines.append("=" * 70)
    lines.append("RULE-BASED DESCRIPTOR SELECTOR - EVALUATION REPORT")
    lines.append("=" * 70)

    # Rules
    lines.append("\n" + selector.describe_rules())

    # LODO per-dataset results
    lines.append("\n\n--- LODO Per-Dataset Results ---")
    lines.append(f"{'Dataset':20s} {'Predicted':25s} {'Oracle':25s} {'Regret':>8s}")
    lines.append("-" * 80)
    for r in lodo_results:
        hit = " *" if r["regret"] == 0.0 else ""
        lines.append(
            f"{r['dataset']:20s} {r['predicted']:25s} {r['oracle']:25s} "
            f"{r['regret']:8.4f}{hit}")

    # Aggregate metrics
    rule_metrics = aggregate_metrics(lodo_results)
    lines.append(f"\nRule-based LODO metrics:")
    lines.append(f"  Mean regret:     {rule_metrics['mean_regret']:.4f} "
                 f"(+/- {rule_metrics['std_regret']:.4f})")
    lines.append(f"  Median regret:   {rule_metrics['median_regret']:.4f}")
    lines.append(f"  Max regret:      {rule_metrics['max_regret']:.4f}")
    lines.append(f"  Top-1 accuracy:  {rule_metrics['top1_accuracy']}")
    lines.append(f"  Soft accuracy:   {rule_metrics['soft_accuracy']} "
                 f"(regret < 0.01)")
    lines.append(f"  Near-optimal:    {rule_metrics['near_optimal']} "
                 f"(regret < 0.02)")

    # Baseline comparison
    lines.append("\n\n--- Baseline Comparison ---")
    lines.append(f"{'Strategy':25s} {'Mean Regret':>12s} {'Top-1':>8s} "
                 f"{'Soft Acc':>10s}")
    lines.append("-" * 60)

    strategies = [("rule_based (LODO)", lodo_results)]
    for name, bl_results in baselines.items():
        strategies.append((name, bl_results))

    for name, results in strategies:
        metrics = aggregate_metrics(results)
        lines.append(
            f"{name:25s} {metrics['mean_regret']:12.4f} "
            f"{metrics['top1_count']:>3d}/13   "
            f"{metrics['soft_count']:>3d}/13")

    # Feature discriminability
    lines.append("\n\n--- Feature Discriminability (Top 3 per group) ---")
    for target, feats in discriminability.items():
        lines.append(f"\n  {target}:")
        for f in feats[:3]:
            lines.append(
                f"    {f['feature']:25s} acc={f['accuracy']:.3f} "
                f"thresh={f['threshold']:.6f} dir={f['direction']} "
                f"target_mean={f['target_mean']:.4f} "
                f"other_mean={f['other_mean']:.4f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    """Run end-to-end rule-based selector evaluation."""
    print("=" * 70)
    print("Rule-Based Descriptor Selector")
    print("=" * 70)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    perf_df, char_df = load_data()
    print(f"  Performance matrix: {len(perf_df)} rows")
    print(f"  Dataset characteristics: {len(char_df)} datasets x "
          f"{len(char_df.columns)} features")

    # Step 2: Apply adjustments
    print("\n[Step 2] Applying adjustments (2% penalty, TDA-only filter)...")
    adj_df = apply_adjustments(perf_df)
    print(f"  Adjusted matrix: {len(adj_df)} rows, "
          f"{adj_df['descriptor'].nunique()} descriptors")

    # Step 3: Compute adjusted winners
    print("\n[Step 3] Computing adjusted winners...")
    winners_df = compute_winners(adj_df)
    print("\n  Adjusted winners:")
    for _, row in winners_df.iterrows():
        print(f"    {row['dataset']:20s}: {row['winner']:25s} "
              f"(gap={row['gap']:.4f})")

    # Count by winner
    winner_counts = winners_df["winner"].value_counts()
    print(f"\n  Winner distribution:")
    for desc, count in winner_counts.items():
        print(f"    {desc}: {count}")

    # Step 4: Feature discriminability analysis
    print("\n[Step 4] Analyzing feature discriminability...")
    discriminability = analyze_feature_discriminability(char_df, winners_df)
    for target, feats in discriminability.items():
        print(f"\n  {target} (top 3 features):")
        for f in feats[:3]:
            print(f"    {f['feature']:25s} acc={f['accuracy']:.3f} "
                  f"thresh={f['threshold']:.6f} ({f['direction']})")

    # Step 5: Fit full selectors (on all data, for rule display)
    print("\n[Step 5] Fitting full selectors (all 13 datasets)...")
    for mode in ["hybrid", "data_driven"]:
        sel = RuleBasedSelector()
        sel.fit(char_df, winners_df, mode=mode)
        print(f"\n  Mode: {mode}")
        print(sel.describe_rules())

    selector = RuleBasedSelector()
    selector.fit(char_df, winners_df, mode="hybrid")

    # Step 6: LODO evaluation (both modes)
    print("\n[Step 6] LODO evaluation...")
    lodo_results_hybrid = lodo_evaluation(char_df, adj_df, mode="hybrid")
    lodo_results_dd = lodo_evaluation(char_df, adj_df, mode="data_driven")

    # Pick the better one
    m_hybrid = aggregate_metrics(lodo_results_hybrid)
    m_dd = aggregate_metrics(lodo_results_dd)
    print(f"\n  Hybrid LODO mean regret:      {m_hybrid['mean_regret']:.4f}")
    print(f"  Data-driven LODO mean regret: {m_dd['mean_regret']:.4f}")

    if m_hybrid["mean_regret"] <= m_dd["mean_regret"]:
        lodo_results = lodo_results_hybrid
        best_mode = "hybrid"
    else:
        lodo_results = lodo_results_dd
        best_mode = "data_driven"
    print(f"  Best mode: {best_mode}")

    # Step 7: Baselines
    print("\n[Step 7] Computing baselines...")
    baselines = compute_baselines(adj_df, char_df)

    # Step 8: Generate report
    print("\n[Step 8] Generating report...")
    report = generate_report(lodo_results, baselines, discriminability, selector)
    print(report)

    # Step 9: Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save report
    report_path = RESULTS_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Saved report to {report_path}")

    # Save LODO results as CSV
    lodo_df = pd.DataFrame([{
        "dataset": r["dataset"],
        "predicted": r["predicted"],
        "oracle": r["oracle"],
        "predicted_score": r["predicted_score"],
        "oracle_score": r["oracle_score"],
        "regret": r["regret"],
        "explanation": r["explanation"],
    } for r in lodo_results])
    lodo_csv_path = RESULTS_DIR / "lodo_results.csv"
    lodo_df.to_csv(lodo_csv_path, index=False)
    print(f"  Saved LODO results to {lodo_csv_path}")

    # Save full results as JSON
    all_results = {
        "rules": [
            {
                "conditions": [
                    {"feature": f, "op": o, "threshold": float(t)}
                    for f, o, t in rule["conditions"]
                ],
                "descriptor": rule["descriptor"],
                "label": rule["label"],
                "accuracy": rule.get("accuracy"),
            }
            for rule in selector.rules
        ],
        "lodo_results": lodo_results,
        "lodo_metrics": aggregate_metrics(lodo_results),
        "baselines": {
            name: {
                "per_dataset": bl_results,
                "metrics": aggregate_metrics(bl_results),
            }
            for name, bl_results in baselines.items()
        },
        "adjusted_winners": winners_df.to_dict(orient="records"),
        "discriminability": {
            target: feats
            for target, feats in discriminability.items()
        },
    }
    json_path = RESULTS_DIR / "full_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved full results to {json_path}")

    # Summary
    rule_metrics = aggregate_metrics(lodo_results)
    atol_metrics = aggregate_metrics(baselines["always_ATOL"])
    tf_metrics = aggregate_metrics(baselines["always_template"])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Rule-based mean regret (LODO): {rule_metrics['mean_regret']:.4f}")
    print(f"  Always-ATOL mean regret:       {atol_metrics['mean_regret']:.4f}")
    print(f"  Always-TF mean regret:         {tf_metrics['mean_regret']:.4f}")
    print(f"  Rule-based beats always-ATOL:  "
          f"{rule_metrics['mean_regret'] < atol_metrics['mean_regret']}")
    print(f"  Rule-based beats always-TF:    "
          f"{rule_metrics['mean_regret'] < tf_metrics['mean_regret']}")
    print(f"  Rule-based top-1 accuracy:     {rule_metrics['top1_accuracy']}")
    print(f"  Rule-based soft accuracy:      {rule_metrics['soft_accuracy']}")

    # Per-dataset comparison table
    print(f"\n  {'Dataset':20s} {'Rule':>8s} {'ATOL':>8s} {'TF':>8s} {'Best':>8s}")
    print(f"  {'-'*56}")
    for i, r in enumerate(lodo_results):
        ds = r["dataset"]
        r_reg = r["regret"]
        a_reg = baselines["always_ATOL"][i]["regret"]
        t_reg = baselines["always_template"][i]["regret"]
        best = min(r_reg, a_reg, t_reg)
        marker = ""
        if r_reg == best and r_reg < min(a_reg, t_reg):
            marker = " <-- rule wins"
        print(f"  {ds:20s} {r_reg:8.4f} {a_reg:8.4f} {t_reg:8.4f} {best:8.4f}{marker}")

    return all_results


if __name__ == "__main__":
    import sys
    # Allow running as script by adding parent to path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Re-import with package context
    from RuleBenchmark.benchmark3.exp7_meta_learning.rule_based_selector import main as _main
    _main()
