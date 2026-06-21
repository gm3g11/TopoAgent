#!/usr/bin/env python3
"""TopoBenchmark v2 — Redesigned evaluation with zero circularity.

Train/test split:
  - Train (5 datasets): BloodMNIST, PathMNIST, RetinaMNIST, DermaMNIST, OrganAMNIST
    These are the exp4 pilot datasets — the ONLY empirical knowledge source.
  - Test (21 datasets): All remaining benchmark4 datasets.
    Never used for any rule/parameter derivation.

8 Methods:
  1. Oracle         — best descriptor per dataset (from benchmark4)
  2. TopoAgent      — exp4 rankings + math descriptions + cheap features (Claude)
  3. TopoAgent-zero — math descriptions + cheap features only, NO rankings (Claude)
  4. LLM zero-shot  — dataset description + descriptor list only (GPT-4o)
  5. Meta-learner   — RidgeCV trained on 5 exp4 datasets with descriptor one-hot
  6. ObjType default— top-1 descriptor per type from exp4
  7. SBA            — always persistence_statistics
  8. Random         — uniform random

Usage:
    # Full run (requires Claude API key for TopoAgent methods)
    python scripts/run_topobenchmark_v2.py

    # Non-LLM methods only (fast, no API key needed)
    python scripts/run_topobenchmark_v2.py --no-llm

    # Smoke test on 2 datasets
    python scripts/run_topobenchmark_v2.py --smoke-test

    # LODO supplementary (all 26 datasets)
    python scripts/run_topobenchmark_v2.py --lodo

    # Multiple LLM runs for variance
    python scripts/run_topobenchmark_v2.py --n-runs 3
"""

import argparse
import json
import sys
import random
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from topoagent.benchmark_runner import (
    TopoBenchmarkRunner,
    TRAIN_DATASETS,
    TEST_DATASETS,
    EXTENDED_DESCRIPTORS,
)
from topoagent.skills.rules_data import DATASET_TO_OBJECT_TYPE, OBJECT_TYPES

OUTPUT_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "v2"


def get_model(model_name: str = "claude"):
    """Get LangChain chat model."""
    if model_name == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.0, max_tokens=1024)
    elif model_name == "gpt4o":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=1024)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_main_experiment(runner, datasets, methods, n_runs=1, seed=42):
    """Run main train/test experiment.

    Args:
        runner: TopoBenchmarkRunner instance
        datasets: List of test dataset names
        methods: List of method names
        n_runs: Number of runs for LLM methods (for variance estimation)
        seed: Random seed

    Returns:
        Dict with all results and summary
    """
    all_results = []
    llm_methods = [m for m in methods if m not in runner.__class__.__dict__
                   and m not in {"oracle", "random", "meta_learner_ridge_train_test",
                                 "meta_learner_train_test", "object_type_default",
                                 "sba", "rule_based"}]

    # Separate LLM and non-LLM methods
    from topoagent.benchmark_runner import NO_LLM_METHODS
    non_llm = [m for m in methods if m in NO_LLM_METHODS or m.startswith("fixed_")]
    llm = [m for m in methods if m not in non_llm]

    for ds_idx, dataset in enumerate(datasets, 1):
        print(f"\n{'='*60}")
        print(f"[{ds_idx}/{len(datasets)}] Dataset: {dataset}")
        print(f"{'='*60}")

        # Non-LLM methods: single run
        for method in non_llm:
            if method not in methods:
                continue
            print(f"  {method}...", end=" ", flush=True)
            try:
                result = runner.evaluate_dataset(
                    dataset_name=dataset, method=method, seed=seed,
                )
                result["run"] = 0
                print(f"acc={result['accuracy']:.1%}, desc={result['descriptor_selected']}")
                all_results.append(result)
            except Exception as e:
                print(f"FAILED: {e}")
                all_results.append({
                    "dataset": dataset, "method": method, "error": str(e),
                    "accuracy": 0.0, "run": 0,
                })

        # LLM methods: n_runs
        for method in llm:
            if method not in methods:
                continue
            for run_idx in range(n_runs):
                run_label = f" (run {run_idx+1}/{n_runs})" if n_runs > 1 else ""
                print(f"  {method}{run_label}...", end=" ", flush=True)
                try:
                    result = runner.evaluate_dataset(
                        dataset_name=dataset, method=method,
                        seed=seed + run_idx,
                    )
                    result["run"] = run_idx
                    print(f"acc={result['accuracy']:.1%}, desc={result['descriptor_selected']}")
                    all_results.append(result)
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results.append({
                        "dataset": dataset, "method": method, "error": str(e),
                        "accuracy": 0.0, "run": run_idx,
                    })

    return all_results


def compute_metrics(results, datasets):
    """Compute all metrics from results.

    Returns:
        Dict with per_method metrics, accuracy_matrix, descriptor_matrix,
        per_type breakdown, and pairwise comparisons.
    """
    by_method = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_method[r["method"]].append(r)

    # SBA baseline
    sba_per_ds = {}
    for r in by_method.get("sba", []):
        sba_per_ds[r["dataset"]] = r["accuracy"]

    oracle_per_ds = {}
    for r in by_method.get("oracle", []):
        oracle_per_ds[r["dataset"]] = r["accuracy"]

    sba_mean = np.mean(list(sba_per_ds.values())) if sba_per_ds else None
    oracle_mean = np.mean(list(oracle_per_ds.values())) if oracle_per_ds else None

    per_method = {}
    for method, method_results in by_method.items():
        # For LLM methods with multiple runs, average per dataset first
        ds_accs = defaultdict(list)
        for r in method_results:
            ds_accs[r["dataset"]].append(r["accuracy"])

        # Mean accuracy per dataset (averaging over runs)
        mean_per_ds = {ds: np.mean(accs) for ds, accs in ds_accs.items()}
        accuracies = list(mean_per_ds.values())

        regrets = []
        for ds, acc in mean_per_ds.items():
            oracle_acc = oracle_per_ds.get(ds, acc)
            regrets.append(oracle_acc - acc)

        mean_acc = float(np.mean(accuracies))

        # Gap closed
        gap_closed = None
        if sba_mean is not None and oracle_mean is not None and (oracle_mean - sba_mean) > 0.005:
            gap_closed = (mean_acc - sba_mean) / (oracle_mean - sba_mean) * 100

        # Win rate vs SBA
        wins = sum(1 for ds in mean_per_ds if mean_per_ds[ds] >= sba_per_ds.get(ds, 0))
        n_compared = sum(1 for ds in mean_per_ds if ds in sba_per_ds)
        win_rate = float(wins / n_compared) if n_compared > 0 else None

        # Descriptor choices
        desc_choices = defaultdict(list)
        for r in method_results:
            desc_choices[r["dataset"]].append(r.get("descriptor_selected", "?"))

        per_method[method] = {
            "n_datasets": len(mean_per_ds),
            "mean_accuracy": mean_acc,
            "std_accuracy": float(np.std(accuracies)),
            "mean_regret": float(np.mean(regrets)),
            "max_regret": float(np.max(regrets)) if regrets else 0,
            "win_rate_vs_sba": win_rate,
            "gap_closed": gap_closed,
            "n_runs": max(len(v) for v in ds_accs.values()) if ds_accs else 1,
        }

    # Accuracy matrix: {method: {dataset: accuracy}}
    acc_matrix = defaultdict(dict)
    desc_matrix = defaultdict(dict)
    for r in results:
        if "error" not in r:
            # Use first run for matrix (or average?)
            if r["dataset"] not in acc_matrix[r["method"]]:
                acc_matrix[r["method"]][r["dataset"]] = r["accuracy"]
                desc_matrix[r["method"]][r["dataset"]] = r.get("descriptor_selected", "?")

    # Per object type breakdown
    per_type = {}
    for ot in OBJECT_TYPES:
        ot_datasets = [ds for ds in datasets if DATASET_TO_OBJECT_TYPE.get(ds) == ot]
        if not ot_datasets:
            continue

        ot_metrics = {}
        for method, method_results in by_method.items():
            ot_results = [r for r in method_results if r["dataset"] in ot_datasets]
            if not ot_results:
                continue
            ds_accs = defaultdict(list)
            for r in ot_results:
                ds_accs[r["dataset"]].append(r["accuracy"])
            mean_per_ds = {ds: np.mean(accs) for ds, accs in ds_accs.items()}
            ot_metrics[method] = {
                "n_datasets": len(mean_per_ds),
                "mean_accuracy": float(np.mean(list(mean_per_ds.values()))),
            }
        per_type[ot] = ot_metrics

    return {
        "per_method": per_method,
        "accuracy_matrix": dict(acc_matrix),
        "descriptor_matrix": dict(desc_matrix),
        "per_type": per_type,
    }


def compute_pairwise_tests(results, datasets, method_pairs=None):
    """Compute Wilcoxon signed-rank tests for pairwise method comparisons.

    Args:
        results: List of result dicts
        datasets: List of dataset names
        method_pairs: List of (method_a, method_b) tuples to compare

    Returns:
        List of test result dicts
    """
    from scipy.stats import wilcoxon

    # Build per-dataset accuracy for each method
    method_accs = defaultdict(dict)
    for r in results:
        if "error" not in r:
            ds = r["dataset"]
            method = r["method"]
            if ds not in method_accs[method]:
                method_accs[method][ds] = r["accuracy"]

    if method_pairs is None:
        all_methods = sorted(method_accs.keys())
        method_pairs = [(a, b) for i, a in enumerate(all_methods)
                        for b in all_methods[i+1:]]

    test_results = []
    for method_a, method_b in method_pairs:
        if method_a not in method_accs or method_b not in method_accs:
            continue

        # Get paired accuracies
        paired_ds = [ds for ds in datasets
                     if ds in method_accs[method_a] and ds in method_accs[method_b]]
        if len(paired_ds) < 5:
            continue

        a_accs = [method_accs[method_a][ds] for ds in paired_ds]
        b_accs = [method_accs[method_b][ds] for ds in paired_ds]
        diffs = [a - b for a, b in zip(a_accs, b_accs)]

        if all(d == 0 for d in diffs):
            test_results.append({
                "method_a": method_a, "method_b": method_b,
                "n": len(paired_ds), "statistic": 0, "p_value": 1.0,
                "mean_diff": 0.0, "significant": False,
            })
            continue

        try:
            stat, p_val = wilcoxon(a_accs, b_accs, alternative="two-sided")
            test_results.append({
                "method_a": method_a,
                "method_b": method_b,
                "n": len(paired_ds),
                "statistic": float(stat),
                "p_value": float(p_val),
                "mean_diff": float(np.mean(diffs)),
                "significant": p_val < 0.05,
            })
        except Exception as e:
            test_results.append({
                "method_a": method_a, "method_b": method_b,
                "n": len(paired_ds), "error": str(e),
            })

    return test_results


def format_results_table(metrics, datasets):
    """Format results as a clean markdown table."""
    per_method = metrics["per_method"]

    METHOD_ORDER = [
        "oracle", "topoagent", "topoagent_zero", "zeroshot",
        "meta_learner_ridge_train_test", "object_type_default",
        "sba", "random",
    ]
    METHOD_NAMES = {
        "oracle": "Oracle",
        "topoagent": "TopoAgent",
        "topoagent_zero": "TopoAgent-zero",
        "zeroshot": "LLM zero-shot",
        "meta_learner_ridge_train_test": "Meta-learner",
        "object_type_default": "ObjType default",
        "sba": "SBA",
        "random": "Random",
    }

    ordered = [m for m in METHOD_ORDER if m in per_method]
    # Add any methods not in the order
    ordered += [m for m in per_method if m not in ordered]

    lines = []
    lines.append(f"{'Method':<24} {'Mean Acc':>10} {'Mean Regret':>12} {'Win vs SBA':>12} {'Gap Closed':>12}")
    lines.append("-" * 74)

    for method in ordered:
        stats = per_method[method]
        name = METHOD_NAMES.get(method, method)
        acc = f"{stats['mean_accuracy']:.1%}"
        regret = f"{stats['mean_regret']:.3f}"
        win = f"{stats['win_rate_vs_sba']:.0%}" if stats['win_rate_vs_sba'] is not None else "--"
        gap = f"{stats['gap_closed']:.1f}%" if stats['gap_closed'] is not None else "--"

        lines.append(f"{name:<24} {acc:>10} {regret:>12} {win:>12} {gap:>12}")

    return "\n".join(lines)


def format_per_type_table(metrics):
    """Format per-object-type breakdown."""
    per_type = metrics.get("per_type", {})
    if not per_type:
        return "(no per-type data)"

    METHOD_NAMES = {
        "oracle": "Oracle", "topoagent": "TopoAgent",
        "topoagent_zero": "TAgent-zero", "zeroshot": "LLM zero",
        "meta_learner_ridge_train_test": "Meta-learn",
        "object_type_default": "ObjType", "sba": "SBA", "random": "Random",
    }

    # Collect all methods across types
    all_methods = set()
    for ot_metrics in per_type.values():
        all_methods.update(ot_metrics.keys())
    methods = sorted(all_methods)

    lines = []
    header = f"{'Object Type':<18}"
    for m in methods:
        name = METHOD_NAMES.get(m, m[:10])
        header += f" {name:>10}"
    header += f" {'N':>4}"
    lines.append(header)
    lines.append("-" * len(header))

    for ot in OBJECT_TYPES:
        if ot not in per_type:
            continue
        row = f"{ot:<18}"
        ot_metrics = per_type[ot]
        n_ds = 0
        for m in methods:
            if m in ot_metrics:
                acc = ot_metrics[m]["mean_accuracy"]
                row += f" {acc:>9.1%}"
                n_ds = ot_metrics[m]["n_datasets"]
            else:
                row += f" {'--':>10}"
        row += f" {n_ds:>4}"
        lines.append(row)

    return "\n".join(lines)


def format_descriptor_choices(metrics, datasets):
    """Show which descriptor each method chose for each dataset."""
    desc_matrix = metrics.get("descriptor_matrix", {})
    if not desc_matrix:
        return "(no descriptor data)"

    lines = []
    header = f"{'Dataset':<24}"
    methods = sorted(desc_matrix.keys())
    for m in methods:
        header += f" {m[:12]:>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for ds in datasets:
        row = f"{ds:<24}"
        for m in methods:
            desc = desc_matrix.get(m, {}).get(ds, "--")
            # Shorten descriptor names
            short = desc.replace("persistence_", "p_").replace("euler_characteristic_", "ec_")
            row += f" {short[:14]:>14}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TopoBenchmark v2 Evaluation")
    parser.add_argument("--no-llm", action="store_true",
                        help="Run only non-LLM methods (no API key needed)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test on 2 datasets")
    parser.add_argument("--lodo", action="store_true",
                        help="Run LODO supplementary instead of main experiment")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of LLM runs for variance (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: auto-generated)")
    parser.add_argument("--model", type=str, default="claude",
                        choices=["claude", "gpt4o"],
                        help="LLM model for TopoAgent/zero-shot")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine experiment mode
    if args.lodo:
        experiment_mode = "lodo"
        datasets = TRAIN_DATASETS + TEST_DATASETS  # All 26
        print("=" * 60)
        print("TopoBenchmark v2 — LODO Supplementary (all 26 datasets)")
        print("=" * 60)
    else:
        experiment_mode = "train_test"
        datasets = TEST_DATASETS  # 21 test only
        print("=" * 60)
        print("TopoBenchmark v2 — Main Experiment (21 test datasets)")
        print("=" * 60)

    if args.smoke_test:
        datasets = datasets[:2]
        print(f"SMOKE TEST: using {datasets}")

    print(f"Experiment mode: {experiment_mode}")
    print(f"Datasets: {len(datasets)}")
    print(f"LLM runs: {args.n_runs}")

    # Define methods
    if args.lodo:
        if args.no_llm:
            methods = ["oracle", "meta_learner", "meta_learner_ridge",
                       "object_type_default", "sba", "random"]
        else:
            methods = ["oracle", "topoagent", "meta_learner", "meta_learner_ridge",
                       "object_type_default", "sba", "random"]
    else:
        if args.no_llm:
            methods = ["oracle", "meta_learner_ridge_train_test",
                       "object_type_default", "sba", "random"]
        else:
            methods = [
                "oracle", "topoagent", "topoagent_zero", "zeroshot",
                "meta_learner_ridge_train_test", "object_type_default",
                "sba", "random",
            ]

    print(f"Methods: {methods}")

    # Initialize model (only if LLM methods are included)
    model = None
    needs_llm = any(m not in {"oracle", "random", "meta_learner", "meta_learner_ridge",
                               "meta_learner_train_test", "meta_learner_ridge_train_test",
                               "object_type_default", "sba", "rule_based"}
                    and not m.startswith("fixed_")
                    for m in methods)
    if needs_llm:
        print(f"\nInitializing LLM: {args.model}")
        model = get_model(args.model)

    # Initialize runner
    runner = TopoBenchmarkRunner(
        model=model,
        lookup_mode=True,
        experiment_mode=experiment_mode,
    )

    # Run experiment
    print(f"\nRunning {len(methods)} methods on {len(datasets)} datasets...")
    all_results = run_main_experiment(
        runner, datasets, methods,
        n_runs=args.n_runs, seed=args.seed,
    )

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)

    metrics = compute_metrics(all_results, datasets)

    # Format and print results
    print("\n## Main Results Table")
    print(format_results_table(metrics, datasets))

    print("\n## Per Object Type Breakdown")
    print(format_per_type_table(metrics))

    print("\n## Descriptor Choices")
    print(format_descriptor_choices(metrics, datasets))

    # Pairwise tests (only if enough datasets)
    if len(datasets) >= 5:
        print("\n## Pairwise Wilcoxon Signed-Rank Tests")
        key_pairs = [
            ("topoagent", "meta_learner_ridge_train_test"),
            ("topoagent", "topoagent_zero"),
            ("topoagent_zero", "zeroshot"),
            ("meta_learner_ridge_train_test", "object_type_default"),
            ("topoagent", "sba"),
            ("topoagent_zero", "sba"),
        ]
        # Filter to pairs where both methods exist
        key_pairs = [(a, b) for a, b in key_pairs
                     if a in metrics["per_method"] and b in metrics["per_method"]]
        if key_pairs:
            try:
                tests = compute_pairwise_tests(all_results, datasets, key_pairs)
                for t in tests:
                    if "error" in t:
                        print(f"  {t['method_a']} vs {t['method_b']}: ERROR ({t['error']})")
                    else:
                        sig = "*" if t["significant"] else ""
                        print(f"  {t['method_a']} vs {t['method_b']}: "
                              f"p={t['p_value']:.4f}{sig}, "
                              f"mean_diff={t['mean_diff']:+.3f}, n={t['n']}")
            except ImportError:
                print("  (scipy not available — skipping statistical tests)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "lodo" if args.lodo else "train_test"
    llm_tag = "no_llm" if args.no_llm else args.model
    output_path = args.output or str(
        OUTPUT_DIR / f"topobenchmark_v2_{mode_tag}_{llm_tag}_{timestamp}.json"
    )

    output = {
        "config": {
            "experiment_mode": experiment_mode,
            "datasets": datasets,
            "methods": methods,
            "n_runs": args.n_runs,
            "seed": args.seed,
            "model": args.model if needs_llm else None,
            "smoke_test": args.smoke_test,
            "timestamp": datetime.now().isoformat(),
            "train_datasets": TRAIN_DATASETS,
            "test_datasets": TEST_DATASETS,
        },
        "results": all_results,
        "metrics": metrics,
    }

    # Remove non-serializable interaction objects
    for r in output["results"]:
        r.pop("llm_interaction", None)
        r.pop("_interaction", None)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Print key comparisons summary
    pm = metrics["per_method"]
    print("\n## Key Comparisons")
    comparisons = [
        ("TopoAgent vs Meta-learner", "topoagent", "meta_learner_ridge_train_test",
         "LLM reasoning + domain knowledge vs statistical learning (same exp4 data)"),
        ("TopoAgent vs TopoAgent-zero", "topoagent", "topoagent_zero",
         "Does empirical data from exp4 help the LLM?"),
        ("TopoAgent-zero vs LLM zero-shot", "topoagent_zero", "zeroshot",
         "Does structured TDA knowledge help vs general LLM knowledge?"),
        ("Meta-learner vs ObjType default", "meta_learner_ridge_train_test", "object_type_default",
         "Does dataset-specific feature analysis beat simple type-based lookup?"),
        ("All vs SBA", "topoagent", "sba",
         "Is intelligent selection better than always picking persistence_statistics?"),
    ]
    for name, a, b, question in comparisons:
        if a in pm and b in pm:
            diff = pm[a]["mean_accuracy"] - pm[b]["mean_accuracy"]
            print(f"  {name}: {diff:+.1%} ({pm[a]['mean_accuracy']:.1%} vs {pm[b]['mean_accuracy']:.1%})")
            print(f"    Q: {question}")

    print("\nDone.")


if __name__ == "__main__":
    main()
