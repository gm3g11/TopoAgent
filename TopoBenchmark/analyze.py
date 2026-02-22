"""Analysis and Visualization for TopoBenchmark.

Generates:
- Main comparison table (baselines + agent × {MBA, DSA, Regret})
- Ablation table
- Per-object-type breakdown
- Descriptor selection heatmap
- Accuracy gap waterfall chart
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ground_truth import GroundTruth, load_ground_truth
from .baselines import DATASET_OBJECT_TYPES, run_all_baselines
from .metrics import (
    evaluate_selections,
    wilcoxon_test,
    bootstrap_ci,
    mcnemar_test,
)
from .agent_runner import extract_selections_from_results, load_protocol1_results


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "topobenchmark"
ANALYSIS_DIR = RESULTS_DIR / "analysis"


def comparison_table(
    agent_results: Dict[str, Dict],
    ground_truth: Optional[GroundTruth] = None,
    baseline_results: Optional[Dict[str, Dict]] = None,
) -> str:
    """Generate main comparison table: baselines + agent × {MBA, DSA, Regret}.

    Args:
        agent_results: Protocol 1 results from agent_runner
        ground_truth: GroundTruth (loaded if None)
        baseline_results: Pre-computed baselines (run if None)

    Returns:
        Formatted table string
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()
    if baseline_results is None:
        baseline_results = run_all_baselines(ground_truth)

    # Compute agent metrics
    selections = extract_selections_from_results(agent_results)
    agent_metrics = evaluate_selections(
        selections, ground_truth, DATASET_OBJECT_TYPES, label="TopoAgent",
    )

    # Build table
    lines = []
    header = (f"{'Method':<32s} {'MBA':>8s} {'95% CI':>15s} {'DSA':>8s} "
              f"{'Top3':>8s} {'Regret':>8s} {'p-val':>8s}")
    lines.append(header)
    lines.append("=" * len(header))

    # Baselines
    for name, metrics in baseline_results.items():
        mba = metrics.get("mba", metrics.get("mba_mean", 0.0))
        ci_lo = metrics.get("mba_ci_lower", 0.0)
        ci_hi = metrics.get("mba_ci_upper", 0.0)
        dsa = metrics.get("dsa", 0.0)
        dsa3 = metrics.get("dsa_top3", 0.0)
        regret = metrics.get("regret", 0.0)
        label = metrics.get("label", name)

        # Wilcoxon vs agent
        b_accs = metrics.get("per_dataset_accuracy", {})
        a_accs = agent_metrics.get("per_dataset_accuracy", {})
        _, p_val = wilcoxon_test(a_accs, b_accs)

        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "  N/A"
        ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]"

        lines.append(
            f"{label:<32s} {mba:>8.4f} {ci_str:>15s} {dsa:>7.1%} "
            f"{dsa3:>7.1%} {regret:>8.4f} {p_str:>8s}"
        )

    # Agent row
    lines.append("-" * len(header))
    mba = agent_metrics["mba"]
    ci_str = f"[{agent_metrics['mba_ci_lower']:.3f},{agent_metrics['mba_ci_upper']:.3f}]"
    lines.append(
        f"{'TopoAgent':<32s} {mba:>8.4f} {ci_str:>15s} "
        f"{agent_metrics['dsa']:>7.1%} {agent_metrics['dsa_top3']:>7.1%} "
        f"{agent_metrics['regret']:>8.4f} {'  ---':>8s}"
    )

    # Oracle
    lines.append("-" * len(header))
    lines.append(
        f"{'Oracle':<32s} {ground_truth.mba:>8.4f} {'':>15s} "
        f"{'100.0%':>8s} {'100.0%':>8s} {'0.0000':>8s} {'':>8s}"
    )

    return "\n".join(lines)


def ablation_table(
    ablation_results: Dict[str, Dict[str, Dict]],
    ground_truth: Optional[GroundTruth] = None,
) -> str:
    """Generate ablation comparison table.

    Args:
        ablation_results: {config_name: {dataset: result_dict}}
        ground_truth: GroundTruth

    Returns:
        Formatted table string
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    lines = []
    header = f"{'Config':<25s} {'MBA':>8s} {'DSA':>8s} {'Top3':>8s} {'Regret':>8s}"
    lines.append(header)
    lines.append("=" * len(header))

    for config_name, results in ablation_results.items():
        selections = extract_selections_from_results(results)
        if not selections:
            lines.append(f"{config_name:<25s} {'N/A':>8s}")
            continue

        metrics = evaluate_selections(
            selections, ground_truth, DATASET_OBJECT_TYPES, label=config_name,
        )
        lines.append(
            f"{config_name:<25s} {metrics['mba']:>8.4f} {metrics['dsa']:>7.1%} "
            f"{metrics['dsa_top3']:>7.1%} {metrics['regret']:>8.4f}"
        )

    return "\n".join(lines)


def per_object_type_table(
    selections: Dict[str, str],
    ground_truth: Optional[GroundTruth] = None,
) -> str:
    """Per-object-type MBA breakdown.

    Args:
        selections: {dataset: descriptor}
        ground_truth: GroundTruth

    Returns:
        Formatted table string
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    from .metrics import compute_per_object_type_mba
    ot_mba = compute_per_object_type_mba(
        selections, ground_truth, DATASET_OBJECT_TYPES,
    )

    # Also compute oracle per object type
    oracle_selections = {
        ds: ground_truth.get_oracle_descriptor(ds)
        for ds in selections.keys()
        if ground_truth.get_oracle_descriptor(ds)
    }
    oracle_ot_mba = compute_per_object_type_mba(
        oracle_selections, ground_truth, DATASET_OBJECT_TYPES,
    )

    # Count datasets per type
    type_counts = {}
    for ds in selections:
        ot = DATASET_OBJECT_TYPES.get(ds, "unknown")
        type_counts[ot] = type_counts.get(ot, 0) + 1

    lines = []
    header = f"{'Object Type':<20s} {'N':>4s} {'Agent MBA':>10s} {'Oracle MBA':>11s} {'Gap':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for ot in sorted(ot_mba.keys()):
        agent = ot_mba[ot]
        oracle = oracle_ot_mba.get(ot, 0.0)
        gap = oracle - agent
        n = type_counts.get(ot, 0)
        lines.append(f"{ot:<20s} {n:>4d} {agent:>10.4f} {oracle:>11.4f} {gap:>8.4f}")

    return "\n".join(lines)


def per_dataset_detail_table(
    selections: Dict[str, str],
    ground_truth: Optional[GroundTruth] = None,
) -> str:
    """Detailed per-dataset results showing agent choice vs oracle.

    Returns:
        Formatted table string
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    lines = []
    header = (f"{'Dataset':<25s} {'Agent Desc':<25s} {'Agent Acc':>10s} "
              f"{'Oracle Desc':<25s} {'Oracle Acc':>10s} {'Regret':>8s}")
    lines.append(header)
    lines.append("=" * len(header))

    for dataset in sorted(selections.keys()):
        agent_desc = selections[dataset]
        agent_acc = ground_truth.get_accuracy(dataset, agent_desc)
        oracle_desc = ground_truth.get_oracle_descriptor(dataset)
        oracle_acc = ground_truth.get_oracle_accuracy(dataset)

        if agent_acc is None or oracle_acc is None:
            continue

        regret = oracle_acc - agent_acc
        match = "*" if agent_desc == oracle_desc else " "

        lines.append(
            f"{dataset:<25s} {agent_desc:<25s} {agent_acc:>10.4f} "
            f"{oracle_desc:<25s} {oracle_acc:>10.4f} {regret:>7.4f}{match}"
        )

    return "\n".join(lines)


def selection_frequency_table(
    selections: Dict[str, str],
) -> str:
    """Show which descriptors the agent selected and how often.

    Returns:
        Formatted table string
    """
    from .metrics import compute_selection_distribution
    dist = compute_selection_distribution(selections)

    lines = []
    header = f"{'Descriptor':<30s} {'Count':>6s} {'Pct':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    total = sum(dist.values())
    for desc, count in dist.items():
        pct = count / total if total > 0 else 0
        lines.append(f"{desc:<30s} {count:>6d} {pct:>7.1%}")

    return "\n".join(lines)


def full_analysis(
    agent_results: Dict[str, Dict],
    ground_truth: Optional[GroundTruth] = None,
    save_dir: Optional[Path] = None,
) -> str:
    """Run complete analysis and save results.

    Args:
        agent_results: Protocol 1 results
        ground_truth: GroundTruth
        save_dir: Where to save analysis files

    Returns:
        Full analysis report as string
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()
    if save_dir is None:
        save_dir = ANALYSIS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    selections = extract_selections_from_results(agent_results)

    sections = []

    # 1. Main comparison table
    sections.append("=" * 80)
    sections.append("MAIN COMPARISON TABLE")
    sections.append("=" * 80)
    comp = comparison_table(agent_results, ground_truth)
    sections.append(comp)

    # 2. Per-dataset detail
    sections.append("\n" + "=" * 80)
    sections.append("PER-DATASET DETAIL")
    sections.append("=" * 80)
    sections.append(per_dataset_detail_table(selections, ground_truth))

    # 3. Per-object-type breakdown
    sections.append("\n" + "=" * 80)
    sections.append("PER-OBJECT-TYPE BREAKDOWN")
    sections.append("=" * 80)
    sections.append(per_object_type_table(selections, ground_truth))

    # 4. Selection frequency
    sections.append("\n" + "=" * 80)
    sections.append("DESCRIPTOR SELECTION FREQUENCY")
    sections.append("=" * 80)
    sections.append(selection_frequency_table(selections))

    report = "\n".join(sections)

    # Save report
    report_path = save_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Analysis report saved to {report_path}")

    # Save metrics as JSON
    metrics = evaluate_selections(
        selections, ground_truth, DATASET_OBJECT_TYPES, label="TopoAgent",
    )
    metrics_path = save_dir / "agent_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze TopoBenchmark results")
    parser.add_argument("--config", default="full", help="Config name")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--results-dir", default=None, help="Protocol 1 results dir")
    args = parser.parse_args()

    # Load results
    results = load_protocol1_results(
        results_dir=Path(args.results_dir) if args.results_dir else None,
        config_name=args.config,
        model_name=args.model,
    )

    if not results:
        print("No results found. Run run_protocol1.py first.")
    else:
        report = full_analysis(results)
        print(report)
