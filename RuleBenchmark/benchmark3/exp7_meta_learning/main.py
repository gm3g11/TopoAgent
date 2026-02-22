"""Exp7: Meta-Learning for Descriptor Selection.

CLI entry point. Trains a scoring model to predict which TDA descriptor
will perform best on a new medical image dataset using 25 cheap features.
"""

import argparse
import json
import time
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from .config import (
    EXP1_CSV, OUTPUT_DIR, MODELS_DIR, CACHE_DIR,
    DATASETS, DESCRIPTORS, FEATURE_NAMES, N_FEATURES,
    GBR_PARAMS, RF_PARAMS, TREE_PARAMS, N_RF_SEEDS,
    N_REPEATS, SAMPLES_PER_REPEAT, DEFAULT_N_SAMPLES, SEED,
)
from .data_loader import load_performance_matrix, load_all_dataset_features, build_training_data
from .models import MetaLearner
from .evaluation import (
    lodo_evaluation, compute_aggregate_metrics, compute_baselines,
    results_to_dataframe,
)
from .skill_extraction import generate_skill_document


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp7: Meta-Learning for Descriptor Selection"
    )
    parser.add_argument(
        "--recompute-features", action="store_true",
        help="Force recomputation of dataset characteristics (ignore cache)"
    )
    parser.add_argument(
        "--n-image-samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of images to load per dataset (default: {DEFAULT_N_SAMPLES})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Use cached features only, skip image loading (fail if no cache)"
    )
    parser.add_argument(
        "--csv-path", type=str, default=str(EXP1_CSV),
        help=f"Path to benchmark3_all_results.csv (default: {EXP1_CSV})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0_total = time.time()

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    print("=" * 70)
    print("Exp7: Meta-Learning for Descriptor Selection")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  Seed: {args.seed}")
    print(f"  N image samples: {args.n_image_samples}")
    print(f"  Skip features: {args.skip_features}")
    print(f"  Recompute features: {args.recompute_features}")
    print()

    # ─── Step 1: Load performance matrix ─────────────────────────────────
    print("=" * 70)
    print("Step 1: Load performance matrix")
    print("=" * 70)

    performance_df = load_performance_matrix(args.csv_path)
    print(f"  Shape: {performance_df.shape}")
    print(f"  Score range: [{performance_df['score'].min():.4f}, {performance_df['score'].max():.4f}]")
    print()

    # Save performance matrix
    perf_path = output_dir / "performance_matrix.csv"
    performance_df.to_csv(perf_path, index=False)
    print(f"  Saved: {perf_path}")

    # ─── Step 2: Load/compute dataset features ───────────────────────────
    print()
    print("=" * 70)
    print("Step 2: Dataset characterization features")
    print("=" * 70)

    cache_path = output_dir / "dataset_characteristics.csv"

    if args.skip_features:
        if not cache_path.exists():
            print(f"ERROR: --skip-features set but cache not found at {cache_path}")
            sys.exit(1)
        print(f"  Loading from cache: {cache_path}")

    dataset_features = load_all_dataset_features(
        n_samples=args.n_image_samples,
        n_repeats=N_REPEATS,
        samples_per_repeat=SAMPLES_PER_REPEAT,
        cache_path=cache_path,
        recompute=args.recompute_features,
        seed=args.seed,
    )

    # Also save to output_dir (may be same as cache)
    chars_path = output_dir / "dataset_characteristics.csv"
    if not chars_path.exists():
        df_chars = pd.DataFrame(dataset_features).T
        df_chars.index.name = "dataset"
        df_chars.to_csv(chars_path)
    print(f"  Features computed for {len(dataset_features)} datasets")

    # ─── Step 3: Build training data ─────────────────────────────────────
    print()
    print("=" * 70)
    print("Step 3: Build training matrix")
    print("=" * 70)

    X, y, meta_df = build_training_data(dataset_features, performance_df, DESCRIPTORS)
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  y stats: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")

    # ─── Step 4: LODO evaluation ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("Step 4: Leave-One-Dataset-Out evaluation")
    print("=" * 70)

    lodo_results = lodo_evaluation(
        X, y, meta_df,
        gbr_params=GBR_PARAMS,
        rf_params=RF_PARAMS,
        tree_params=TREE_PARAMS,
        n_rf_seeds=N_RF_SEEDS,
    )

    # Aggregate metrics
    aggregate = compute_aggregate_metrics(lodo_results)
    baselines = compute_baselines(lodo_results, performance_df)

    print()
    print("  ─── Aggregate Results ───")
    print(f"  Top-1 Accuracy: {aggregate['top1_accuracy']*100:.1f}%")
    print(f"  Top-2 Accuracy: {aggregate['top2_accuracy']*100:.1f}%")
    print(f"  Top-3 Accuracy: {aggregate['top3_accuracy']*100:.1f}%")
    print(f"  Mean Regret:    {aggregate['mean_regret']:.4f} (std={aggregate['std_regret']:.4f})")
    print(f"  Mean Spearman:  {aggregate['mean_spearman_rho']:.3f}")
    print()
    print("  ─── Baselines ───")
    print(f"  Random:              mean_regret={baselines['random']['mean_regret']:.4f}")
    print(f"  Always-Best-Overall: mean_regret={baselines['always_best_overall']['mean_regret']:.4f}")
    print(f"  Always-ATOL:         mean_regret={baselines['always_ATOL']['mean_regret']:.4f}")

    # Save LODO results
    lodo_df = results_to_dataframe(lodo_results)
    lodo_path = output_dir / "lodo_results.csv"
    lodo_df.to_csv(lodo_path, index=False)
    print(f"\n  Saved: {lodo_path}")

    # Save aggregate metrics
    aggregate_with_baselines = {**aggregate, "baselines": baselines}
    metrics_path = output_dir / "aggregate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(aggregate_with_baselines, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # ─── Step 5: Train final model on all data ───────────────────────────
    print()
    print("=" * 70)
    print("Step 5: Train final model on all data")
    print("=" * 70)

    final_model = MetaLearner(
        gbr_params=GBR_PARAMS,
        rf_params=RF_PARAMS,
        tree_params=TREE_PARAMS,
        n_rf_seeds=N_RF_SEEDS,
    )
    final_model.fit(X, y)

    # Save models
    joblib.dump(final_model.gbr, models_dir / "gbr_final.pkl")
    joblib.dump(final_model.tree, models_dir / "tree_final.pkl")
    joblib.dump(final_model.rf_ensemble, models_dir / "rf_ensemble_final.pkl")
    joblib.dump(final_model.scaler, models_dir / "scaler_final.pkl")
    print(f"  Saved models to {models_dir}")

    # ─── Step 6: Feature importance + tree rules ─────────────────────────
    print()
    print("=" * 70)
    print("Step 6: Feature importance & decision rules")
    print("=" * 70)

    all_feature_names = final_model.get_all_feature_names(DESCRIPTORS)
    importance_pairs = final_model.get_feature_importance(all_feature_names)

    # Save importance
    imp_df = pd.DataFrame(importance_pairs, columns=["feature", "importance"])
    imp_path = output_dir / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")

    print("\n  Top 10 features:")
    for rank, (name, imp) in enumerate(importance_pairs[:10], 1):
        print(f"    {rank:2d}. {name:30s} {imp:.4f}")

    # Tree rules
    tree_rules = final_model.get_tree_rules(all_feature_names)
    rules_path = output_dir / "tree_rules.txt"
    rules_path.write_text(tree_rules)
    print(f"\n  Saved: {rules_path}")
    print(f"\n  Decision tree rules (first 20 lines):")
    for line in tree_rules.split("\n")[:20]:
        print(f"    {line}")

    # ─── Step 7: Generate TOPOAGENT_SKILL.md ─────────────────────────────
    print()
    print("=" * 70)
    print("Step 7: Generate TOPOAGENT_SKILL.md")
    print("=" * 70)

    skill_path = output_dir / "TOPOAGENT_SKILL.md"
    generate_skill_document(
        aggregate_metrics=aggregate,
        baselines=baselines,
        importance_pairs=importance_pairs,
        tree_rules=tree_rules,
        lodo_results=lodo_results,
        performance_df=performance_df,
        output_path=skill_path,
    )

    # ─── Done ────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0_total
    print()
    print("=" * 70)
    print(f"Exp7 complete. Total time: {elapsed_total:.1f}s")
    print("=" * 70)
    print(f"\nOutputs in {output_dir}:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            print(f"  {p.relative_to(output_dir)} ({size:,} bytes)")


if __name__ == "__main__":
    main()
