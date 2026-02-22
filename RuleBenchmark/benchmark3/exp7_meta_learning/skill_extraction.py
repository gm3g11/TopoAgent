"""Generate TOPOAGENT_SKILL.md from trained meta-learner results."""

import numpy as np
from pathlib import Path


def generate_skill_document(aggregate_metrics, baselines, importance_pairs,
                            tree_rules, lodo_results, performance_df,
                            output_path):
    """Generate the TOPOAGENT_SKILL.md document.

    Args:
        aggregate_metrics: dict from compute_aggregate_metrics()
        baselines: dict from compute_baselines()
        importance_pairs: list of (name, importance) from get_feature_importance()
        tree_rules: str from get_tree_rules()
        lodo_results: list of per-dataset dicts
        performance_df: DataFrame with [dataset, descriptor, score]
        output_path: Path to write the skill document
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # ─── Header ──────────────────────────────────────────────────────────
    lines.append("# TOPOAGENT_SKILL: Adaptive Descriptor Selection")
    lines.append("")
    lines.append("Meta-learned rules for selecting the best TDA descriptor for a new medical image dataset.")
    lines.append("Trained on 13 datasets x 15 descriptors using Leave-One-Dataset-Out CV.")
    lines.append("")

    # ─── Performance Summary ─────────────────────────────────────────────
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Top-1 Accuracy | {aggregate_metrics['top1_accuracy']*100:.1f}% |")
    lines.append(f"| Top-2 Accuracy | {aggregate_metrics['top2_accuracy']*100:.1f}% |")
    lines.append(f"| Top-3 Accuracy | {aggregate_metrics['top3_accuracy']*100:.1f}% |")
    lines.append(f"| Mean Regret | {aggregate_metrics['mean_regret']:.4f} |")
    lines.append(f"| Mean Spearman rho | {aggregate_metrics['mean_spearman_rho']:.3f} |")
    lines.append("")

    # ─── Baselines Comparison ────────────────────────────────────────────
    lines.append("## vs Baselines")
    lines.append("")
    lines.append("| Strategy | Mean Regret |")
    lines.append("|----------|-------------|")
    lines.append(f"| **Meta-Learner (ours)** | **{aggregate_metrics['mean_regret']:.4f}** |")
    lines.append(f"| Always-Best-Overall | {baselines['always_best_overall']['mean_regret']:.4f} |")
    lines.append(f"| Random | {baselines['random']['mean_regret']:.4f} |")
    lines.append("")

    # ─── Quick Reference Table ───────────────────────────────────────────
    lines.append("## Quick Reference: Best Descriptor by Dataset")
    lines.append("")
    lines.append("| Dataset | Oracle Best | Predicted Best | Regret |")
    lines.append("|---------|-------------|----------------|--------|")
    for r in sorted(lodo_results, key=lambda x: x["dataset"]):
        lines.append(f"| {r['dataset']} | {r['oracle_best']} | {r['predicted_best']} | {r['regret']:.4f} |")
    lines.append("")

    # ─── Decision Rules (from DecisionTree) ──────────────────────────────
    lines.append("## Decision Rules (Interpretable Tree)")
    lines.append("")
    lines.append("```")
    lines.append(tree_rules.strip())
    lines.append("```")
    lines.append("")

    # ─── Feature Importance ──────────────────────────────────────────────
    lines.append("## Feature Importance (Top 15)")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    for rank, (name, imp) in enumerate(importance_pairs[:15], 1):
        lines.append(f"| {rank} | {name} | {imp:.4f} |")
    lines.append("")

    # ─── Descriptor Performance Summary ──────────────────────────────────
    lines.append("## Descriptor Performance Across Datasets")
    lines.append("")
    desc_stats = performance_df.groupby("descriptor")["score"].agg(["mean", "std", "min", "max"])
    desc_stats = desc_stats.sort_values("mean", ascending=False)
    lines.append("| Descriptor | Mean | Std | Min | Max |")
    lines.append("|------------|------|-----|-----|-----|")
    for desc, row in desc_stats.iterrows():
        lines.append(f"| {desc} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |")
    lines.append("")

    # ─── Per-Dataset Detailed Results ────────────────────────────────────
    lines.append("## Per-Dataset Predictions")
    lines.append("")
    for r in sorted(lodo_results, key=lambda x: x["dataset"]):
        lines.append(f"### {r['dataset']}")
        lines.append(f"- Oracle: {r['oracle_best']} ({r['oracle_score']:.4f})")
        lines.append(f"- Predicted: {r['predicted_best']} ({r['predicted_score']:.4f})")
        lines.append(f"- Regret: {r['regret']:.4f}")
        lines.append(f"- Spearman rho: {r['spearman_rho']:.3f}")
        lines.append("")

        # Top-3 predicted
        preds = r["predictions"]
        sorted_preds = sorted(preds.items(), key=lambda x: x[1]["pred"], reverse=True)
        lines.append("| Rank | Descriptor | Pred | True |")
        lines.append("|------|------------|------|------|")
        for rank, (desc, vals) in enumerate(sorted_preds[:5], 1):
            marker = " *" if desc == r["oracle_best"] else ""
            lines.append(f"| {rank} | {desc}{marker} | {vals['pred']:.4f} | {vals['true']:.4f} |")
        lines.append("")

    # ─── Usage Example ───────────────────────────────────────────────────
    lines.append("## Usage Example")
    lines.append("")
    lines.append("```python")
    lines.append("from exp7_meta_learning.features import compute_cheap_features_stable")
    lines.append("from exp7_meta_learning.models import MetaLearner")
    lines.append("import joblib")
    lines.append("")
    lines.append("# 1. Compute dataset characteristics")
    lines.append("features = compute_cheap_features_stable(images, labels)")
    lines.append("")
    lines.append("# 2. Load trained meta-learner")
    lines.append("model = joblib.load('results/benchmark3/exp7/models/gbr_final.pkl')")
    lines.append("")
    lines.append("# 3. Predict best descriptor")
    lines.append("# Build input: 25 features + one-hot for each descriptor candidate")
    lines.append("# Select descriptor with highest predicted score")
    lines.append("```")
    lines.append("")

    # Write file
    content = "\n".join(lines)
    output_path.write_text(content)
    print(f"[skill_extraction] Wrote {output_path} ({len(lines)} lines)")
