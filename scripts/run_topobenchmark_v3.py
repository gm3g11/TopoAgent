#!/usr/bin/env python3
"""TopoBenchmark v3 — Multi-descriptor fusion via predict-verify-portfolio.

Extends v2 with:
  - TopoAgent-portfolio: 3-step LLM reasoning → descriptor fusion
  - TopoAgent-single-PV: predict-verify without fusion (ablation)
  - Oracle-pair: best 2-descriptor fusion per dataset
  - Random-pair: random 2-descriptor fusion baseline
  - Fusion gain analysis
  - Reasoning trace logging
  - Sequential vs independent evaluation modes

11 Methods:
  1. Oracle-single    — best descriptor per dataset (from benchmark4)
  2. Oracle-pair      — best pair per dataset (from fusion_accuracy_lookup)
  3. TopoAgent-portfolio — 3-step predict→verify→portfolio (NEW, our main method)
  4. TopoAgent-single — exp4 rankings + math descriptions (v2 method)
  5. TopoAgent-single-PV — predict-verify, single descriptor (ablation)
  6. TopoAgent-zero   — math descriptions only, NO rankings
  7. LLM zero-shot    — dataset description + descriptor list only
  8. Meta-learner     — RidgeCV on 5 exp4 datasets
  9. ObjType default  — top-1 per type from exp4
  10. SBA             — always persistence_statistics
  11. Random-pair     — random 2-descriptor concat

Usage:
    # Full run (requires Claude API key)
    python scripts/run_topobenchmark_v3.py

    # Non-LLM methods only
    python scripts/run_topobenchmark_v3.py --no-llm

    # Smoke test on 2 datasets
    python scripts/run_topobenchmark_v3.py --smoke-test

    # Multiple LLM runs for variance
    python scripts/run_topobenchmark_v3.py --n-runs 3

    # Sequential mode (agent learns from previous datasets)
    python scripts/run_topobenchmark_v3.py --sequential
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
    NO_LLM_METHODS,
)
from topoagent.skills.rules_data import DATASET_TO_OBJECT_TYPE, OBJECT_TYPES

OUTPUT_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "v3"


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


def run_experiment(runner, datasets, methods, n_runs=1, seed=42, sequential=False):
    """Run experiment with all methods on all datasets.

    Args:
        runner: TopoBenchmarkRunner instance
        datasets: List of test dataset names
        methods: List of method names
        n_runs: Number of runs for LLM methods
        seed: Random seed
        sequential: If True, process datasets by object type (agent learns)

    Returns:
        List of result dicts
    """
    all_results = []

    # Determine LLM vs non-LLM methods
    non_llm = [m for m in methods if m in NO_LLM_METHODS or m.startswith("fixed_")]
    llm = [m for m in methods if m not in non_llm]

    # Optionally reorder datasets by object type for sequential learning
    if sequential:
        type_order = ["discrete_cells", "glands_lumens", "organ_shape",
                       "vessel_trees", "surface_lesions"]
        datasets = sorted(datasets,
                          key=lambda d: type_order.index(DATASET_TO_OBJECT_TYPE.get(d, "organ_shape")))
        print(f"Sequential mode: datasets ordered by object type")

    for ds_idx, dataset in enumerate(datasets, 1):
        print(f"\n{'='*60}")
        print(f"[{ds_idx}/{len(datasets)}] Dataset: {dataset}")
        print(f"{'='*60}")

        # Non-LLM methods: single run
        for method in non_llm:
            print(f"  {method}...", end=" ", flush=True)
            try:
                result = runner.evaluate_dataset(
                    dataset_name=dataset, method=method, seed=seed,
                )
                result["run"] = 0
                result["sequential"] = sequential
                portfolio_str = "+".join(result.get("portfolio", [result.get("descriptor_selected", "?")]))
                print(f"acc={result['accuracy']:.1%}, desc={portfolio_str}")
                all_results.append(result)
            except Exception as e:
                print(f"FAILED: {e}")
                all_results.append({
                    "dataset": dataset, "method": method, "error": str(e),
                    "accuracy": 0.0, "run": 0,
                })

        # LLM methods: n_runs
        for method in llm:
            for run_idx in range(n_runs):
                run_label = f" (run {run_idx+1}/{n_runs})" if n_runs > 1 else ""
                print(f"  {method}{run_label}...", end=" ", flush=True)
                try:
                    result = runner.evaluate_dataset(
                        dataset_name=dataset, method=method,
                        seed=seed + run_idx,
                    )
                    result["run"] = run_idx
                    result["sequential"] = sequential
                    portfolio_str = "+".join(result.get("portfolio", [result.get("descriptor_selected", "?")]))
                    print(f"acc={result['accuracy']:.1%}, desc={portfolio_str}")
                    all_results.append(result)
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results.append({
                        "dataset": dataset, "method": method, "error": str(e),
                        "accuracy": 0.0, "run": run_idx,
                    })

    return all_results


def compute_metrics(results, datasets):
    """Compute all metrics including fusion-specific analysis."""
    by_method = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_method[r["method"]].append(r)

    # Build per-dataset lookups
    sba_per_ds = {}
    for r in by_method.get("sba", []):
        sba_per_ds[r["dataset"]] = r["accuracy"]
    oracle_per_ds = {}
    for r in by_method.get("oracle", []):
        oracle_per_ds[r["dataset"]] = r["accuracy"]
    oracle_pair_per_ds = {}
    for r in by_method.get("oracle_pair", []):
        oracle_pair_per_ds[r["dataset"]] = r["accuracy"]

    sba_mean = np.mean(list(sba_per_ds.values())) if sba_per_ds else None
    oracle_mean = np.mean(list(oracle_per_ds.values())) if oracle_per_ds else None
    oracle_pair_mean = np.mean(list(oracle_pair_per_ds.values())) if oracle_pair_per_ds else None

    per_method = {}
    for method, method_results in by_method.items():
        ds_accs = defaultdict(list)
        for r in method_results:
            ds_accs[r["dataset"]].append(r["accuracy"])

        mean_per_ds = {ds: np.mean(accs) for ds, accs in ds_accs.items()}
        accuracies = list(mean_per_ds.values())

        regrets_single = []
        regrets_pair = []
        for ds, acc in mean_per_ds.items():
            oracle_acc = oracle_per_ds.get(ds, acc)
            regrets_single.append(oracle_acc - acc)
            oracle_pair_acc = oracle_pair_per_ds.get(ds, acc)
            regrets_pair.append(oracle_pair_acc - acc)

        mean_acc = float(np.mean(accuracies))

        # Gap closed vs SBA → Oracle-single
        gap_closed_single = None
        if sba_mean is not None and oracle_mean is not None and (oracle_mean - sba_mean) > 0.005:
            gap_closed_single = (mean_acc - sba_mean) / (oracle_mean - sba_mean) * 100

        # Gap closed vs SBA → Oracle-pair
        gap_closed_pair = None
        if sba_mean is not None and oracle_pair_mean is not None and (oracle_pair_mean - sba_mean) > 0.005:
            gap_closed_pair = (mean_acc - sba_mean) / (oracle_pair_mean - sba_mean) * 100

        # Win rate vs SBA
        wins = sum(1 for ds in mean_per_ds if mean_per_ds[ds] >= sba_per_ds.get(ds, 0))
        n_compared = sum(1 for ds in mean_per_ds if ds in sba_per_ds)
        win_rate = float(wins / n_compared) if n_compared > 0 else None

        # Beats oracle-single rate
        beats_oracle_single = sum(
            1 for ds in mean_per_ds
            if mean_per_ds[ds] > oracle_per_ds.get(ds, float('inf'))
        )

        # Fusion statistics
        n_portfolio = sum(1 for r in method_results if r.get("is_portfolio"))
        fusion_strategies = [r.get("fusion_strategy") for r in method_results if r.get("fusion_strategy")]

        per_method[method] = {
            "n_datasets": len(mean_per_ds),
            "mean_accuracy": mean_acc,
            "std_accuracy": float(np.std(accuracies)),
            "mean_regret_vs_single": float(np.mean(regrets_single)),
            "mean_regret_vs_pair": float(np.mean(regrets_pair)),
            "win_rate_vs_sba": win_rate,
            "gap_closed_vs_single": gap_closed_single,
            "gap_closed_vs_pair": gap_closed_pair,
            "beats_oracle_single": beats_oracle_single,
            "n_portfolio": n_portfolio,
            "fusion_strategies": fusion_strategies,
            "n_runs": max(len(v) for v in ds_accs.values()) if ds_accs else 1,
        }

    # Accuracy matrix
    acc_matrix = defaultdict(dict)
    desc_matrix = defaultdict(dict)
    for r in results:
        if "error" not in r:
            if r["dataset"] not in acc_matrix[r["method"]]:
                acc_matrix[r["method"]][r["dataset"]] = r["accuracy"]
                desc_matrix[r["method"]][r["dataset"]] = r.get("descriptor_selected", "?")

    # Per-type breakdown
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

    # Fusion gain analysis
    fusion_analysis = analyze_fusion_gains(results, oracle_per_ds)

    return {
        "per_method": per_method,
        "accuracy_matrix": dict(acc_matrix),
        "descriptor_matrix": dict(desc_matrix),
        "per_type": per_type,
        "fusion_analysis": fusion_analysis,
    }


def analyze_fusion_gains(results, oracle_per_ds):
    """Analyze when fusion helps vs hurts.

    Compares portfolio methods against their primary descriptor alone.
    """
    portfolio_results = [r for r in results
                         if r.get("is_portfolio") and "error" not in r]

    if not portfolio_results:
        return {"message": "No portfolio results to analyze"}

    gains = []
    for r in portfolio_results:
        portfolio = r.get("portfolio", [])
        if len(portfolio) < 2:
            continue

        # We need the primary descriptor's standalone accuracy
        # (from single-descriptor oracle_per_ds or from accuracy_lookup)
        ds = r["dataset"]
        fusion_acc = r["accuracy"]
        oracle_acc = oracle_per_ds.get(ds, fusion_acc)

        gains.append({
            "dataset": ds,
            "method": r["method"],
            "object_type": r.get("object_type", "?"),
            "portfolio": portfolio,
            "fusion_accuracy": fusion_acc,
            "oracle_single_accuracy": oracle_acc,
            "beats_oracle": fusion_acc > oracle_acc + 0.001,
            "fusion_strategy": r.get("fusion_strategy", "?"),
        })

    n_beats_oracle = sum(1 for g in gains if g["beats_oracle"])

    return {
        "n_portfolio_experiments": len(gains),
        "n_beats_oracle_single": n_beats_oracle,
        "pct_beats_oracle": (n_beats_oracle / len(gains) * 100) if gains else 0,
        "per_dataset": gains,
    }


def format_results_table(metrics, datasets):
    """Format results as a markdown table."""
    per_method = metrics["per_method"]

    METHOD_ORDER = [
        "oracle_pair", "oracle",
        "topoagent_portfolio", "topoagent", "topoagent_single_pv",
        "topoagent_zero", "zeroshot",
        "meta_learner_ridge_train_test", "object_type_default",
        "sba", "random_pair", "random",
    ]
    METHOD_NAMES = {
        "oracle_pair": "Oracle-pair",
        "oracle": "Oracle-single",
        "topoagent_portfolio": "TopoAgent-portfolio",
        "topoagent": "TopoAgent-single",
        "topoagent_single_pv": "TopoAgent-PV",
        "topoagent_zero": "TopoAgent-zero",
        "zeroshot": "LLM zero-shot",
        "meta_learner_ridge_train_test": "Meta-learner",
        "object_type_default": "ObjType default",
        "sba": "SBA",
        "random_pair": "Random-pair",
        "random": "Random",
    }

    ordered = [m for m in METHOD_ORDER if m in per_method]
    ordered += [m for m in per_method if m not in ordered]

    lines = []
    lines.append(f"{'Method':<24} {'Acc':>8} {'Regret↓':>10} {'Win/SBA':>10} "
                 f"{'Gap% ↑':>8} {'Beat Oracle':>12}")
    lines.append("-" * 78)

    for method in ordered:
        stats = per_method[method]
        name = METHOD_NAMES.get(method, method)
        acc = f"{stats['mean_accuracy']:.1%}"
        regret = f"{stats['mean_regret_vs_single']:.3f}"
        win = f"{stats['win_rate_vs_sba']:.0%}" if stats['win_rate_vs_sba'] is not None else "--"
        gap = f"{stats['gap_closed_vs_single']:.1f}%" if stats['gap_closed_vs_single'] is not None else "--"
        beats = f"{stats['beats_oracle_single']}/{stats['n_datasets']}"

        lines.append(f"{name:<24} {acc:>8} {regret:>10} {win:>10} {gap:>8} {beats:>12}")

    return "\n".join(lines)


def format_per_type_table(metrics):
    """Format per-object-type breakdown."""
    per_type = metrics.get("per_type", {})
    if not per_type:
        return "(no per-type data)"

    METHOD_NAMES = {
        "oracle_pair": "O-pair", "oracle": "Oracle",
        "topoagent_portfolio": "TA-port", "topoagent": "TA-sing",
        "topoagent_single_pv": "TA-PV", "topoagent_zero": "TA-zero",
        "zeroshot": "LLM-0", "meta_learner_ridge_train_test": "MetaL",
        "object_type_default": "ObjType", "sba": "SBA",
        "random_pair": "Rnd-pr", "random": "Random",
    }

    all_methods = set()
    for ot_metrics in per_type.values():
        all_methods.update(ot_metrics.keys())
    methods = sorted(all_methods)

    lines = []
    header = f"{'Object Type':<18}"
    for m in methods:
        name = METHOD_NAMES.get(m, m[:7])
        header += f" {name:>7}"
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
                row += f" {acc:>6.1%}"
                n_ds = ot_metrics[m]["n_datasets"]
            else:
                row += f" {'--':>7}"
        row += f" {n_ds:>4}"
        lines.append(row)

    return "\n".join(lines)


def format_fusion_analysis(metrics):
    """Format fusion gain analysis."""
    analysis = metrics.get("fusion_analysis", {})
    if "message" in analysis:
        return analysis["message"]

    lines = []
    lines.append(f"Portfolio experiments: {analysis['n_portfolio_experiments']}")
    lines.append(f"Beats oracle-single: {analysis['n_beats_oracle_single']} "
                 f"({analysis['pct_beats_oracle']:.1f}%)")

    lines.append("\nPer-dataset fusion outcomes:")
    for g in analysis.get("per_dataset", []):
        portfolio_str = "+".join(g["portfolio"])
        marker = "BEATS" if g["beats_oracle"] else "below"
        lines.append(f"  {g['dataset']:24s}  {portfolio_str:40s}  "
                     f"fusion={g['fusion_accuracy']:.3f}  oracle={g['oracle_single_accuracy']:.3f}  "
                     f"{marker}")

    return "\n".join(lines)


def format_reasoning_traces(results):
    """Format reasoning quality analysis from predict-verify traces."""
    pv_results = [r for r in results
                  if r.get("method") in ("topoagent_portfolio", "topoagent_single_pv")
                  and "error" not in r]

    if not pv_results:
        return "(no predict-verify reasoning traces)"

    lines = []
    n_confirm = 0
    n_revise = 0
    n_override = 0
    prediction_accuracy = []  # Was predicted #1 in exp4 top-3?

    for r in pv_results:
        interactions = r.get("llm_interactions", [])
        # Extract verify action if available
        verify = None
        for interaction in interactions:
            if hasattr(interaction, 'step') and 'verify' in interaction.step:
                try:
                    verify = json.loads(interaction.response)
                except (json.JSONDecodeError, AttributeError):
                    pass

        if verify:
            action = verify.get("action", "?")
            if action == "confirm":
                n_confirm += 1
            elif action == "revise":
                n_revise += 1
            elif action == "keep_override":
                n_override += 1

    total = n_confirm + n_revise + n_override
    if total > 0:
        lines.append(f"Predict-verify traces: {total}")
        lines.append(f"  Confirmed: {n_confirm} ({n_confirm/total*100:.0f}%)")
        lines.append(f"  Revised: {n_revise} ({n_revise/total*100:.0f}%)")
        lines.append(f"  Override (kept prediction): {n_override} ({n_override/total*100:.0f}%)")

    return "\n".join(lines) if lines else "(no verify actions parsed)"


def main():
    parser = argparse.ArgumentParser(description="TopoBenchmark v3 Evaluation")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="claude",
                        choices=["claude", "gpt4o"])
    parser.add_argument("--sequential", action="store_true",
                        help="Process datasets by object type for sequential learning")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = TEST_DATASETS
    if args.smoke_test:
        datasets = datasets[:2]

    print("=" * 60)
    print("TopoBenchmark v3 — Multi-Descriptor Fusion")
    print("=" * 60)
    print(f"Datasets: {len(datasets)}")
    print(f"LLM runs: {args.n_runs}")
    print(f"Sequential: {args.sequential}")

    # Define methods
    if args.no_llm:
        methods = [
            "oracle", "oracle_pair",
            "meta_learner_ridge_train_test", "object_type_default",
            "sba", "random_pair", "random",
        ]
    else:
        methods = [
            "oracle", "oracle_pair",
            "topoagent_portfolio", "topoagent", "topoagent_single_pv",
            "topoagent_zero", "zeroshot",
            "meta_learner_ridge_train_test", "object_type_default",
            "sba", "random_pair", "random",
        ]

    print(f"Methods: {methods}")

    # Initialize model
    model = None
    needs_llm = any(m not in NO_LLM_METHODS and not m.startswith("fixed_")
                    for m in methods)
    if needs_llm:
        print(f"\nInitializing LLM: {args.model}")
        model = get_model(args.model)

    # Initialize runner
    runner = TopoBenchmarkRunner(
        model=model,
        lookup_mode=True,
        experiment_mode="train_test",
    )

    # Run experiment
    print(f"\nRunning {len(methods)} methods on {len(datasets)} datasets...")
    all_results = run_experiment(
        runner, datasets, methods,
        n_runs=args.n_runs, seed=args.seed,
        sequential=args.sequential,
    )

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)

    metrics = compute_metrics(all_results, datasets)

    # Print results
    print("\n## Main Results Table")
    print(format_results_table(metrics, datasets))

    print("\n## Per Object Type Breakdown")
    print(format_per_type_table(metrics))

    print("\n## Fusion Gain Analysis")
    print(format_fusion_analysis(metrics))

    print("\n## Reasoning Quality")
    print(format_reasoning_traces(all_results))

    # Key comparisons
    pm = metrics["per_method"]
    print("\n## Key Comparisons")
    comparisons = [
        ("Portfolio vs Oracle-single",
         "topoagent_portfolio", "oracle",
         "Can LLM-guided fusion beat single-descriptor oracle?"),
        ("Portfolio vs Oracle-pair",
         "topoagent_portfolio", "oracle_pair",
         "How close does LLM portfolio get to perfect fusion?"),
        ("Portfolio vs Single (predict-verify effect)",
         "topoagent_portfolio", "topoagent",
         "Does multi-descriptor fusion beat single descriptor?"),
        ("Single-PV vs Single (predict-verify helps?)",
         "topoagent_single_pv", "topoagent",
         "Does predict-verify improve single descriptor selection?"),
        ("Single vs Meta-learner",
         "topoagent", "meta_learner_ridge_train_test",
         "LLM reasoning vs statistical learning (same knowledge)"),
        ("Random-pair vs Oracle-single",
         "random_pair", "oracle",
         "Does naive fusion already beat single oracle?"),
    ]
    for name, a, b, question in comparisons:
        if a in pm and b in pm:
            diff = pm[a]["mean_accuracy"] - pm[b]["mean_accuracy"]
            print(f"  {name}: {diff:+.1%} "
                  f"({pm[a]['mean_accuracy']:.1%} vs {pm[b]['mean_accuracy']:.1%})")
            print(f"    Q: {question}")

    # Pairwise tests
    if len(datasets) >= 5:
        print("\n## Pairwise Wilcoxon Tests")
        key_pairs = [
            ("topoagent_portfolio", "oracle"),
            ("topoagent_portfolio", "topoagent"),
            ("topoagent_single_pv", "topoagent"),
            ("topoagent", "meta_learner_ridge_train_test"),
            ("topoagent_portfolio", "sba"),
        ]
        key_pairs = [(a, b) for a, b in key_pairs
                     if a in metrics["per_method"] and b in metrics["per_method"]]
        if key_pairs:
            try:
                from scripts.run_topobenchmark_v2 import compute_pairwise_tests
                tests = compute_pairwise_tests(all_results, datasets, key_pairs)
                for t in tests:
                    if "error" in t:
                        print(f"  {t['method_a']} vs {t['method_b']}: ERROR")
                    else:
                        sig = "*" if t["significant"] else ""
                        print(f"  {t['method_a']} vs {t['method_b']}: "
                              f"p={t['p_value']:.4f}{sig}, diff={t['mean_diff']:+.3f}")
            except ImportError:
                print("  (could not import pairwise test function)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seq_tag = "sequential" if args.sequential else "independent"
    llm_tag = "no_llm" if args.no_llm else args.model
    output_path = args.output or str(
        OUTPUT_DIR / f"topobenchmark_v3_{seq_tag}_{llm_tag}_{timestamp}.json"
    )

    output = {
        "config": {
            "version": "v3",
            "datasets": datasets,
            "methods": methods,
            "n_runs": args.n_runs,
            "seed": args.seed,
            "model": args.model if needs_llm else None,
            "sequential": args.sequential,
            "smoke_test": args.smoke_test,
            "timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
        "metrics": metrics,
    }

    # Remove non-serializable objects
    for r in output["results"]:
        r.pop("llm_interaction", None)
        r.pop("llm_interactions", None)
        r.pop("_interaction", None)
        r.pop("_interactions", None)
        # Also clean nested selection data
        for key in ("predict_result", "verify_result", "portfolio_result"):
            if key in r:
                r.pop(key, None)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
