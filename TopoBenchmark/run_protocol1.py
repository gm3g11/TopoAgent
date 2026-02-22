#!/usr/bin/env python3
"""Run TopoBenchmark Protocol 1: Per-Dataset Strategic Selection.

For each dataset, the agent receives context (description, object type,
color mode) and selects ONE descriptor. We look up the pre-computed
Benchmark4 accuracy for evaluation.

Usage:
    # Run on all 26 datasets with default settings
    python TopoBenchmark/run_protocol1.py

    # Run on specific datasets
    python TopoBenchmark/run_protocol1.py --datasets BloodMNIST,DermaMNIST

    # Run with specific ablation config
    python TopoBenchmark/run_protocol1.py --config no_skills

    # Run baselines only (no LLM needed)
    python TopoBenchmark/run_protocol1.py --baselines-only

    # Analyze existing results
    python TopoBenchmark/run_protocol1.py --analyze-only
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from TopoBenchmark.ground_truth import load_ground_truth
from TopoBenchmark.baselines import run_all_baselines, print_baseline_table
from TopoBenchmark.config import DATASET_DESCRIPTIONS, ABLATION_CONFIGS
from TopoBenchmark.agent_runner import (
    run_protocol1,
    extract_selections_from_results,
    load_protocol1_results,
)
from TopoBenchmark.metrics import evaluate_selections
from TopoBenchmark.analyze import full_analysis, comparison_table
from TopoBenchmark.baselines import DATASET_OBJECT_TYPES


RESULTS_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "protocol1"


def main():
    parser = argparse.ArgumentParser(
        description="TopoBenchmark Protocol 1: Per-Dataset Strategic Selection",
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated list of datasets (default: all 26)",
    )
    parser.add_argument(
        "--config", type=str, default="full",
        choices=list(ABLATION_CONFIGS.keys()),
        help="Ablation config (default: full)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="LLM model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=4,
        help="Max reasoning rounds (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/topobenchmark/protocol1/)",
    )
    parser.add_argument(
        "--baselines-only", action="store_true",
        help="Only run baselines (no LLM needed)",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Only analyze existing results",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (default: from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="API base URL (default: from OPENAI_BASE_URL env var)",
    )
    parser.add_argument(
        "--mode", type=str, default="direct",
        choices=["direct", "workflow"],
        help="direct: single LLM call (fast). workflow: full agent pipeline.",
    )

    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth from benchmark4...")
    gt = load_ground_truth()
    print(f"  {gt.n_results} results, {gt.n_datasets} datasets, {gt.n_descriptors} descriptors")
    print(f"  Oracle MBA: {gt.mba:.4f}")
    print()

    # ---------------------------------------------------------------
    # Baselines (always computed)
    # ---------------------------------------------------------------
    print("Running baselines...")
    baseline_results = run_all_baselines(gt)
    print()
    print(print_baseline_table(baseline_results))
    print(f"{'Oracle':<30s} {gt.mba:>8.4f} {'100.0%':>8s} {'100.0%':>10s} {'0.0000':>8s}")
    print()

    if args.baselines_only:
        # Save baselines
        output_dir = Path(args.output) if args.output else RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "baselines.json", "w") as f:
            json.dump(baseline_results, f, indent=2, default=str)
        print(f"Baselines saved to {output_dir / 'baselines.json'}")
        return

    # ---------------------------------------------------------------
    # Analyze existing results
    # ---------------------------------------------------------------
    if args.analyze_only:
        results = load_protocol1_results(
            config_name=args.config,
            model_name=args.model,
        )
        if not results:
            print("No existing results found. Run without --analyze-only first.")
            return
        report = full_analysis(results, gt)
        print(report)
        return

    # ---------------------------------------------------------------
    # Run agent
    # ---------------------------------------------------------------
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
        # Validate
        for d in datasets:
            if d not in DATASET_DESCRIPTIONS:
                print(f"ERROR: Unknown dataset '{d}'")
                print(f"Available: {list(DATASET_DESCRIPTIONS.keys())}")
                return

    output_dir = Path(args.output) if args.output else RESULTS_DIR

    print(f"Running Protocol 1 with config='{args.config}', model='{args.model}'")
    print(f"Datasets: {len(datasets) if datasets else 26}")
    print()

    agent_results = run_protocol1(
        datasets=datasets,
        model_name=args.model,
        config_name=args.config,
        output_dir=output_dir,
        mode=args.mode,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    selections = extract_selections_from_results(agent_results)
    if selections:
        metrics = evaluate_selections(
            selections, gt, DATASET_OBJECT_TYPES, label="TopoAgent",
        )
        print(f"\nTopoAgent ({args.config}):")
        print(f"  MBA:     {metrics['mba']:.4f} [{metrics['mba_ci_lower']:.3f}, {metrics['mba_ci_upper']:.3f}]")
        print(f"  DSA:     {metrics['dsa']:.1%}")
        print(f"  Top-3:   {metrics['dsa_top3']:.1%}")
        print(f"  Regret:  {metrics['regret']:.4f}")
        print(f"  Oracle:  {gt.mba:.4f}")
        print()

        # Full analysis
        report = full_analysis(agent_results, gt, save_dir=output_dir / "analysis")
        print(report)
    else:
        print("No successful selections. Check agent logs for errors.")


if __name__ == "__main__":
    main()
