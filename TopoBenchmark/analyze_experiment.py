#!/usr/bin/env python3
"""Analyze TopoBenchmark Experiment Results.

Aggregates per-dataset JSONs into:
  1. Main comparison table (26 rows x 4 methods)
  2. Summary statistics (mean accuracy per method)
  3. Per-object-type breakdown
  4. Statistical tests (Wilcoxon, bootstrap CI)
  5. Descriptor frequency analysis

Usage:
    python TopoBenchmark/analyze_experiment.py
    python TopoBenchmark/analyze_experiment.py --input results/topobenchmark/experiment/
    python TopoBenchmark/analyze_experiment.py --latex
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from TopoBenchmark.metrics import wilcoxon_test, bootstrap_ci
from topoagent.skills.rules_data import DATASET_TO_OBJECT_TYPE, DATASET_COLOR_MODE

ALL_DATASETS = list(DATASET_TO_OBJECT_TYPE.keys())
METHODS = ["topoagent", "gpt4o", "medrax", "rule_based"]
METHOD_LABELS = {
    "topoagent": "TopoAgent",
    "gpt4o": "GPT-4o",
    "medrax": "MedRAX",
    "rule_based": "Rule-based",
}


def load_results(input_dir: Path) -> dict:
    """Load all result JSONs into {method: {dataset: result_dict}}."""
    data = {m: {} for m in METHODS}
    for method in METHODS:
        for dataset in ALL_DATASETS:
            path = input_dir / f"{dataset}_{method}.json"
            if path.exists():
                with open(path) as f:
                    data[method][dataset] = json.load(f)
    return data


def print_main_table(data: dict, metric: str = "best_balanced_accuracy"):
    """Print the main comparison table: 26 datasets x 4 methods."""
    print("\n" + "=" * 130)
    print("  MAIN COMPARISON TABLE")
    print("=" * 130)

    header = (f"  {'Dataset':<22} {'ObjType':<16} "
              f"{'TopoAgent':>25} {'GPT-4o':>25} "
              f"{'MedRAX':>25} {'Rule-based':>25}")
    print(header)
    print("  " + "-" * 128)

    for dataset in ALL_DATASETS:
        obj_type = DATASET_TO_OBJECT_TYPE[dataset]
        row = f"  {dataset:<22} {obj_type:<16} "

        for method in METHODS:
            r = data[method].get(dataset)
            if r:
                desc = r.get("descriptor", "?")[:15]
                acc = r.get(metric, 0)
                row += f"{desc:>15} {acc:>8.4f} "
            else:
                row += f"{'---':>15} {'---':>8} "

        print(row)


def compute_summary(data: dict, metric: str = "best_balanced_accuracy") -> dict:
    """Compute summary statistics for each method."""
    summary = {}
    for method in METHODS:
        accs = []
        for dataset in ALL_DATASETS:
            r = data[method].get(dataset)
            if r and r.get(metric, 0) > 0:
                accs.append(r[metric])

        if accs:
            mean, ci_low, ci_high = bootstrap_ci(accs, n_boot=2000, ci=0.95)
            summary[method] = {
                "mean": mean,
                "std": float(np.std(accs)),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_datasets": len(accs),
                "min": float(np.min(accs)),
                "max": float(np.max(accs)),
                "median": float(np.median(accs)),
            }
        else:
            summary[method] = {"mean": 0, "std": 0, "n_datasets": 0}

    return summary


def print_summary_table(data: dict, metric: str = "best_balanced_accuracy"):
    """Print summary statistics per method."""
    summary = compute_summary(data, metric)

    print("\n" + "=" * 90)
    print(f"  SUMMARY ({metric})")
    print("=" * 90)
    print(f"  {'Method':<15} {'Mean':>8} {'Std':>8} {'95% CI':>18} "
          f"{'Min':>8} {'Median':>8} {'Max':>8} {'N':>4}")
    print("  " + "-" * 88)

    for method in METHODS:
        s = summary[method]
        if s["n_datasets"] > 0:
            print(f"  {METHOD_LABELS[method]:<15} {s['mean']:>8.4f} {s['std']:>8.4f} "
                  f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}] "
                  f"{s['min']:>8.4f} {s['median']:>8.4f} {s['max']:>8.4f} {s['n_datasets']:>4}")
        else:
            print(f"  {METHOD_LABELS[method]:<15} {'---':>8}")


def print_per_object_type(data: dict, metric: str = "best_balanced_accuracy"):
    """Print per-object-type breakdown."""
    print("\n" + "=" * 100)
    print("  PER-OBJECT-TYPE BREAKDOWN")
    print("=" * 100)

    object_types = sorted(set(DATASET_TO_OBJECT_TYPE.values()))

    print(f"  {'Object Type':<18} {'N':>3} "
          f"{'TopoAgent':>12} {'GPT-4o':>12} {'MedRAX':>12} {'Rule-based':>12}")
    print("  " + "-" * 75)

    for obj_type in object_types:
        datasets_of_type = [d for d in ALL_DATASETS
                           if DATASET_TO_OBJECT_TYPE[d] == obj_type]
        n = len(datasets_of_type)

        row = f"  {obj_type:<18} {n:>3} "
        for method in METHODS:
            accs = []
            for ds in datasets_of_type:
                r = data[method].get(ds)
                if r and r.get(metric, 0) > 0:
                    accs.append(r[metric])
            if accs:
                row += f"{np.mean(accs):>12.4f}"
            else:
                row += f"{'---':>12}"
        print(row)


def print_statistical_tests(data: dict, metric: str = "best_balanced_accuracy"):
    """Print pairwise Wilcoxon signed-rank tests."""
    print("\n" + "=" * 80)
    print("  STATISTICAL TESTS (Wilcoxon signed-rank, one-sided: row > col)")
    print("=" * 80)

    # Build per-dataset accuracy dicts
    method_accs = {}
    for method in METHODS:
        method_accs[method] = {}
        for dataset in ALL_DATASETS:
            r = data[method].get(dataset)
            if r and r.get(metric, 0) > 0:
                method_accs[method][dataset] = r[metric]

    # Header
    print(f"  {'':>15}", end="")
    for m2 in METHODS:
        print(f"  {METHOD_LABELS[m2]:>15}", end="")
    print()
    print("  " + "-" * 78)

    for m1 in METHODS:
        print(f"  {METHOD_LABELS[m1]:>15}", end="")
        for m2 in METHODS:
            if m1 == m2:
                print(f"  {'---':>15}", end="")
            else:
                stat, p = wilcoxon_test(method_accs[m1], method_accs[m2])
                sig = ""
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                print(f"  {f'p={p:.4f}{sig}':>15}", end="")
        print()


def print_descriptor_frequency(data: dict):
    """Print descriptor selection frequency per method."""
    print("\n" + "=" * 80)
    print("  DESCRIPTOR SELECTION FREQUENCY")
    print("=" * 80)

    for method in METHODS:
        descriptors = []
        for dataset in ALL_DATASETS:
            r = data[method].get(dataset)
            if r:
                descriptors.append(r.get("descriptor", "?"))

        freq = Counter(descriptors)
        n_unique = len(freq)
        print(f"\n  {METHOD_LABELS[method]} ({len(descriptors)} datasets, "
              f"{n_unique} unique descriptors):")
        for desc, count in freq.most_common():
            pct = 100 * count / len(descriptors) if descriptors else 0
            print(f"    {desc:<30} {count:>3} ({pct:>5.1f}%)")


def print_agreement_matrix(data: dict):
    """Print pairwise agreement between methods."""
    print("\n" + "=" * 80)
    print("  PAIRWISE AGREEMENT (same descriptor selected)")
    print("=" * 80)

    print(f"  {'':>15}", end="")
    for m2 in METHODS:
        print(f"  {METHOD_LABELS[m2]:>15}", end="")
    print()
    print("  " + "-" * 78)

    for m1 in METHODS:
        print(f"  {METHOD_LABELS[m1]:>15}", end="")
        for m2 in METHODS:
            agree = 0
            total = 0
            for dataset in ALL_DATASETS:
                r1 = data[m1].get(dataset)
                r2 = data[m2].get(dataset)
                if r1 and r2:
                    total += 1
                    if r1.get("descriptor") == r2.get("descriptor"):
                        agree += 1
            if total > 0:
                pct = 100 * agree / total
                print(f"  {f'{agree}/{total} ({pct:.0f}%)':>15}", end="")
            else:
                print(f"  {'---':>15}", end="")
        print()


def print_win_loss(data: dict, metric: str = "best_balanced_accuracy"):
    """Print win/loss/tie counts for TopoAgent vs each baseline."""
    print("\n" + "=" * 80)
    print("  WIN / LOSS / TIE (TopoAgent vs. baselines)")
    print("=" * 80)

    for baseline in ["gpt4o", "medrax", "rule_based"]:
        wins = losses = ties = 0
        win_datasets = []
        loss_datasets = []
        for dataset in ALL_DATASETS:
            r_agent = data["topoagent"].get(dataset)
            r_base = data[baseline].get(dataset)
            if r_agent and r_base:
                a = r_agent.get(metric, 0)
                b = r_base.get(metric, 0)
                if abs(a - b) < 1e-6:
                    ties += 1
                elif a > b:
                    wins += 1
                    win_datasets.append(f"{dataset}(+{a-b:.4f})")
                else:
                    losses += 1
                    loss_datasets.append(f"{dataset}({b-a:.4f})")

        total = wins + losses + ties
        print(f"\n  vs {METHOD_LABELS[baseline]}: "
              f"{wins}W / {losses}L / {ties}T (of {total})")
        if win_datasets:
            print(f"    Wins:   {', '.join(win_datasets[:8])}"
                  f"{'...' if len(win_datasets) > 8 else ''}")
        if loss_datasets:
            print(f"    Losses: {', '.join(loss_datasets[:8])}"
                  f"{'...' if len(loss_datasets) > 8 else ''}")


def generate_latex_table(data: dict, metric: str = "best_balanced_accuracy") -> str:
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Descriptor selection comparison across 26 medical image datasets.}")
    lines.append(r"\label{tab:experiment}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Object Type & TopoAgent & GPT-4o & MedRAX & Rule-based \\")
    lines.append(r"\midrule")

    for dataset in ALL_DATASETS:
        obj_type = DATASET_TO_OBJECT_TYPE[dataset].replace("_", r"\_")
        row_vals = []
        best_val = 0
        for method in METHODS:
            r = data[method].get(dataset)
            val = r.get(metric, 0) if r else 0
            row_vals.append(val)
            best_val = max(best_val, val)

        cells = []
        for val in row_vals:
            if val > 0:
                s = f"{val:.3f}"
                if abs(val - best_val) < 1e-6:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            else:
                cells.append("---")

        ds_name = dataset.replace("_", r"\_")
        lines.append(f"{ds_name} & {obj_type} & {' & '.join(cells)} \\\\")

    lines.append(r"\midrule")

    # Summary row
    summary = compute_summary(data, metric)
    cells = []
    best_mean = max(s["mean"] for s in summary.values() if s.get("mean", 0) > 0)
    for method in METHODS:
        s = summary[method]
        if s["n_datasets"] > 0:
            val = f"{s['mean']:.3f}"
            if abs(s["mean"] - best_mean) < 1e-4:
                val = r"\textbf{" + val + "}"
            cells.append(val)
        else:
            cells.append("---")

    lines.append(r"\textit{Mean} & & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_summary_json(data: dict, output_dir: Path,
                      metric: str = "best_balanced_accuracy"):
    """Save consolidated summary to JSON."""
    summary = compute_summary(data, metric)

    # Per-dataset results
    per_dataset = {}
    for dataset in ALL_DATASETS:
        per_dataset[dataset] = {
            "object_type": DATASET_TO_OBJECT_TYPE[dataset],
            "color_mode": DATASET_COLOR_MODE.get(dataset, "grayscale"),
        }
        for method in METHODS:
            r = data[method].get(dataset)
            if r:
                per_dataset[dataset][method] = {
                    "descriptor": r.get("descriptor"),
                    "balanced_accuracy": r.get("best_balanced_accuracy", 0),
                    "accuracy": r.get("best_accuracy", 0),
                    "best_classifier": r.get("best_classifier"),
                    "feature_dim": r.get("feature_dim", 0),
                }

    # Descriptor frequencies
    desc_freq = {}
    for method in METHODS:
        freq = Counter()
        for dataset in ALL_DATASETS:
            r = data[method].get(dataset)
            if r:
                freq[r.get("descriptor", "?")] += 1
        desc_freq[method] = dict(freq.most_common())

    out = {
        "summary": summary,
        "per_dataset": per_dataset,
        "descriptor_frequency": desc_freq,
        "metric": metric,
        "n_datasets": len(ALL_DATASETS),
    }

    path = output_dir / "experiment_summary.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved summary to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TopoBenchmark experiment results")
    parser.add_argument("--input", type=str,
                        default="results/topobenchmark/experiment/",
                        help="Input directory with per-dataset JSONs")
    parser.add_argument("--metric", type=str, default="best_balanced_accuracy",
                        choices=["best_balanced_accuracy", "best_accuracy"],
                        help="Primary metric (default: best_balanced_accuracy)")
    parser.add_argument("--latex", action="store_true",
                        help="Generate LaTeX table")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    # Count available results
    n_files = sum(1 for f in input_dir.glob("*_*.json")
                  if not f.name.startswith("experiment_"))
    print(f"Loading results from: {input_dir}")
    print(f"Found {n_files} result files")

    data = load_results(input_dir)

    # Count per method
    for method in METHODS:
        n = len(data[method])
        print(f"  {METHOD_LABELS[method]}: {n}/{len(ALL_DATASETS)} datasets")

    # Print all analyses
    print_main_table(data, args.metric)
    print_summary_table(data, args.metric)
    print_per_object_type(data, args.metric)
    print_statistical_tests(data, args.metric)
    print_descriptor_frequency(data)
    print_agreement_matrix(data)
    print_win_loss(data, args.metric)

    # Save summary JSON
    save_summary_json(data, input_dir, args.metric)

    # Optional LaTeX
    if args.latex:
        latex = generate_latex_table(data, args.metric)
        latex_path = input_dir / "experiment_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to: {latex_path}")
        print("\n--- LaTeX Preview ---")
        print(latex[:500])


if __name__ == "__main__":
    main()
