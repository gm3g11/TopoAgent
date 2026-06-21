#!/usr/bin/env python
"""Analyze and Visualize TopoAgent Experiment Results.

This script provides:
- Statistical significance tests (McNemar's test, paired t-test)
- Accuracy comparison bar charts
- Confusion matrix heatmaps
- Ablation study visualization
- Cross-dataset comparison plots
- Markdown table generation

Usage:
    # Analyze all results in directory
    python scripts/analyze_results.py --results-dir results/experiments

    # Generate specific visualizations
    python scripts/analyze_results.py --results-file results/exp1.json --plot-accuracy

    # Generate markdown tables
    python scripts/analyze_results.py --results-dir results/experiments --tables
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(path: str) -> Dict[str, Any]:
    """Load results from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Results dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """Load all result files from a directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dictionary of filename -> results
    """
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("**/*.json"):
        try:
            results[json_file.stem] = load_results(str(json_file))
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def mcnemar_test(correct1: List[bool], correct2: List[bool]) -> Dict[str, float]:
    """Perform McNemar's test for paired binary classifications.

    Args:
        correct1: Boolean list of correct predictions for method 1
        correct2: Boolean list of correct predictions for method 2

    Returns:
        Dictionary with test statistic and p-value
    """
    try:
        from scipy.stats import chi2

        # Build contingency table
        # b: correct1 wrong, correct2 right
        # c: correct1 right, correct2 wrong
        b = sum(1 for c1, c2 in zip(correct1, correct2) if not c1 and c2)
        c = sum(1 for c1, c2 in zip(correct1, correct2) if c1 and not c2)

        # McNemar statistic with continuity correction
        if b + c == 0:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "b": b,
            "c": c
        }

    except ImportError:
        print("Warning: scipy not installed. Cannot perform McNemar's test.")
        return {"error": "scipy not installed"}


def paired_ttest(values1: List[float], values2: List[float]) -> Dict[str, float]:
    """Perform paired t-test.

    Args:
        values1: Values for method 1
        values2: Values for method 2

    Returns:
        Dictionary with test statistic and p-value
    """
    try:
        from scipy.stats import ttest_rel

        statistic, p_value = ttest_rel(values1, values2)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }

    except ImportError:
        print("Warning: scipy not installed. Cannot perform t-test.")
        return {"error": "scipy not installed"}


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute confidence interval for mean.

    Args:
        values: List of values
        confidence: Confidence level

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    try:
        from scipy.stats import t

        n = len(values)
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(n)
        t_val = t.ppf((1 + confidence) / 2, n - 1)

        margin = t_val * std_err

        return float(mean), float(mean - margin), float(mean + margin)

    except ImportError:
        mean = np.mean(values)
        std = np.std(values)
        return float(mean), float(mean - 1.96*std/np.sqrt(len(values))), float(mean + 1.96*std/np.sqrt(len(values)))


def plot_accuracy_comparison(
    results: Dict[str, Any],
    output_path: str,
    title: str = "Method Comparison"
) -> None:
    """Create bar chart comparing method accuracies.

    Args:
        results: Results dictionary with methods
        output_path: Path to save figure
        title: Chart title
    """
    try:
        import matplotlib.pyplot as plt

        methods = []
        accuracies = []
        f1_scores = []

        for name, data in results.get("methods", {}).items():
            if "error" not in data:
                methods.append(name)
                metrics = data.get("metrics", {})
                accuracies.append(metrics.get("accuracy", 0) * 100)
                f1_scores.append(metrics.get("macro_f1", 0) * 100)

        if not methods:
            print("No valid methods to plot")
            return

        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', color='coral')

        ax.set_ylabel('Score (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved accuracy comparison to: {output_path}")

    except ImportError:
        print("Warning: matplotlib not installed. Cannot create plots.")


def plot_confusion_matrix(
    confusion_matrix: List[List[int]],
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """Create confusion matrix heatmap.

    Args:
        confusion_matrix: NxN confusion matrix
        class_names: List of class names
        output_path: Path to save figure
        title: Chart title
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(confusion_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax)

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved confusion matrix to: {output_path}")

    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Cannot create heatmap.")


def plot_ablation_study(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """Create ablation study visualization.

    Args:
        results: Ablation study results
        output_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        configs = []
        accuracies = []
        baseline_acc = None

        for config_name, data in results.get("configs", {}).items():
            if "error" not in data:
                configs.append(config_name)
                acc = data.get("results", {}).get("metrics", {}).get("accuracy", 0)
                accuracies.append(acc * 100)
                if config_name == "full_v2":
                    baseline_acc = acc * 100

        if not configs:
            print("No valid configs to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if acc == baseline_acc else 'coral' for acc in accuracies]
        bars = ax.bar(configs, accuracies, color=colors)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Ablation Study: Component Contribution')
        ax.set_ylim(0, 100)

        # Add baseline reference line
        if baseline_acc is not None:
            ax.axhline(y=baseline_acc, color='green', linestyle='--', label=f'Full v2: {baseline_acc:.1f}%')
            ax.legend()

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            diff = acc - baseline_acc if baseline_acc else 0
            diff_str = f"\n({diff:+.1f}%)" if baseline_acc and acc != baseline_acc else ""
            ax.annotate(f'{acc:.1f}%{diff_str}',
                       xy=(bar.get_x() + bar.get_width() / 2, acc),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved ablation study plot to: {output_path}")

    except ImportError:
        print("Warning: matplotlib not installed. Cannot create plots.")


def plot_cross_dataset(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """Create cross-dataset comparison plot.

    Args:
        results: Cross-dataset results
        output_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        datasets = results.get("datasets", {})
        if not datasets:
            print("No datasets to plot")
            return

        # Get all methods
        all_methods = set()
        for ds_results in datasets.values():
            all_methods.update(ds_results.keys())

        methods = sorted(all_methods)
        ds_names = list(datasets.keys())

        # Build data matrix
        data = np.zeros((len(ds_names), len(methods)))
        for i, ds_name in enumerate(ds_names):
            for j, method in enumerate(methods):
                acc = datasets[ds_name].get(method, {}).get("accuracy", 0)
                data[i, j] = acc * 100

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(ds_names))
        width = 0.8 / len(methods)

        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        for j, method in enumerate(methods):
            offset = (j - len(methods)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[:, j], width, label=method, color=colors[j])

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Cross-Dataset Generalization')
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-dataset plot to: {output_path}")

    except ImportError:
        print("Warning: matplotlib not installed. Cannot create plots.")


def generate_markdown_table(
    results: Dict[str, Any],
    table_type: str = "comparison"
) -> str:
    """Generate markdown table from results.

    Args:
        results: Results dictionary
        table_type: Type of table to generate

    Returns:
        Markdown table string
    """
    if table_type == "comparison":
        return _generate_comparison_table(results)
    elif table_type == "ablation":
        return _generate_ablation_table(results)
    elif table_type == "cross_dataset":
        return _generate_cross_dataset_table(results)
    else:
        return ""


def _generate_comparison_table(results: Dict[str, Any]) -> str:
    """Generate comparison table markdown."""
    lines = [
        "## Baseline Comparison Results",
        "",
        "| Method | Accuracy | Macro F1 | Avg Time |",
        "|--------|----------|----------|----------|"
    ]

    for name, data in results.get("methods", {}).items():
        if "error" in data:
            lines.append(f"| {name} | ERROR | - | - |")
            continue

        metrics = data.get("metrics", {})
        acc = metrics.get("accuracy", 0) * 100
        f1 = metrics.get("macro_f1", 0) * 100

        timing = data.get("timing", {})
        if isinstance(timing, dict):
            t = timing.get("avg_per_sample", data.get("avg_time", 0))
        else:
            t = data.get("avg_time", 0)

        if t > 1:
            time_str = f"{t:.2f}s"
        else:
            time_str = f"{t*1000:.0f}ms"

        lines.append(f"| {name} | {acc:.1f}% | {f1:.1f}% | {time_str} |")

    return "\n".join(lines)


def _generate_ablation_table(results: Dict[str, Any]) -> str:
    """Generate ablation table markdown."""
    lines = [
        "## Ablation Study Results",
        "",
        "| Configuration | Accuracy | Diff vs Full | Rounds | Time |",
        "|---------------|----------|--------------|--------|------|"
    ]

    baseline_acc = None
    for config_name, data in results.get("configs", {}).items():
        if config_name == "full_v2" and "results" in data:
            baseline_acc = data["results"].get("metrics", {}).get("accuracy", 0)
            break

    for config_name, data in results.get("configs", {}).items():
        if "error" in data:
            lines.append(f"| {config_name} | ERROR | - | - | - |")
            continue

        r = data.get("results", {})
        acc = r.get("metrics", {}).get("accuracy", 0) * 100
        avg_rounds = r.get("rounds", {}).get("avg", 0)
        avg_time = r.get("timing", {}).get("avg_per_sample", 0)

        diff = ""
        if baseline_acc is not None:
            diff_val = acc - baseline_acc * 100
            diff = f"{diff_val:+.1f}%"

        lines.append(f"| {config_name} | {acc:.1f}% | {diff} | {avg_rounds:.1f} | {avg_time:.2f}s |")

    return "\n".join(lines)


def _generate_cross_dataset_table(results: Dict[str, Any]) -> str:
    """Generate cross-dataset table markdown."""
    datasets = results.get("datasets", {})

    # Get all methods
    all_methods = set()
    for ds_results in datasets.values():
        all_methods.update(ds_results.keys())
    methods = sorted(all_methods)

    # Header
    header = "| Dataset |"
    separator = "|---------|"
    for method in methods:
        header += f" {method[:12]} |"
        separator += "------------|"

    lines = [
        "## Cross-Dataset Generalization",
        "",
        header,
        separator
    ]

    # Rows
    for ds_name, ds_results in datasets.items():
        row = f"| {ds_name} |"
        for method in methods:
            acc = ds_results.get(method, {}).get("accuracy", 0) * 100
            row += f" {acc:.1f}% |"
        lines.append(row)

    # Average row
    avg_row = "| **Average** |"
    for method in methods:
        accs = []
        for ds_results in datasets.values():
            if "accuracy" in ds_results.get(method, {}):
                accs.append(ds_results[method]["accuracy"])
        if accs:
            avg_row += f" **{np.mean(accs)*100:.1f}%** |"
        else:
            avg_row += " - |"
    lines.append(avg_row)

    return "\n".join(lines)


def analyze_results(
    results_dir: str,
    output_dir: str,
    create_plots: bool = True,
    create_tables: bool = True
) -> Dict[str, Any]:
    """Analyze all results and generate outputs.

    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save analysis outputs
        create_plots: Whether to create visualization plots
        create_tables: Whether to create markdown tables

    Returns:
        Analysis summary
    """
    all_results = load_all_results(results_dir)

    if not all_results:
        print(f"No results found in {results_dir}")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(exist_ok=True)
    (output_path / "tables").mkdir(exist_ok=True)

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "results_analyzed": list(all_results.keys()),
        "outputs": []
    }

    for name, results in all_results.items():
        print(f"\nAnalyzing: {name}")

        # Determine result type and generate appropriate outputs
        if "methods" in results:
            # Baseline comparison results
            if create_plots:
                plot_path = output_path / "figures" / f"{name}_accuracy.png"
                plot_accuracy_comparison(results, str(plot_path), title=f"Baseline Comparison - {name}")
                analysis["outputs"].append(str(plot_path))

            if create_tables:
                table = generate_markdown_table(results, "comparison")
                table_path = output_path / "tables" / f"{name}_comparison.md"
                with open(table_path, 'w') as f:
                    f.write(table)
                analysis["outputs"].append(str(table_path))
                print(f"  Generated table: {table_path}")

            # Confusion matrix if available
            for method_name, method_data in results.get("methods", {}).items():
                metrics = method_data.get("metrics", {})
                if "confusion_matrix" in metrics:
                    cm_path = output_path / "figures" / f"{name}_{method_name}_cm.png"
                    plot_confusion_matrix(
                        metrics["confusion_matrix"],
                        metrics.get("class_names", []),
                        str(cm_path),
                        title=f"Confusion Matrix - {method_name}"
                    )
                    analysis["outputs"].append(str(cm_path))

        elif "configs" in results:
            # Ablation study results
            if create_plots:
                plot_path = output_path / "figures" / f"{name}_ablation.png"
                plot_ablation_study(results, str(plot_path))
                analysis["outputs"].append(str(plot_path))

            if create_tables:
                table = generate_markdown_table(results, "ablation")
                table_path = output_path / "tables" / f"{name}_ablation.md"
                with open(table_path, 'w') as f:
                    f.write(table)
                analysis["outputs"].append(str(table_path))
                print(f"  Generated table: {table_path}")

        elif "datasets" in results:
            # Cross-dataset results
            if create_plots:
                plot_path = output_path / "figures" / f"{name}_cross_dataset.png"
                plot_cross_dataset(results, str(plot_path))
                analysis["outputs"].append(str(plot_path))

            if create_tables:
                table = generate_markdown_table(results, "cross_dataset")
                table_path = output_path / "tables" / f"{name}_cross_dataset.md"
                with open(table_path, 'w') as f:
                    f.write(table)
                analysis["outputs"].append(str(table_path))
                print(f"  Generated table: {table_path}")

    # Save analysis summary
    summary_path = output_path / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis summary saved to: {summary_path}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze TopoAgent Experiment Results")

    parser.add_argument("--results-dir", "-r", type=str, default="results/experiments/raw",
                       help="Directory containing result JSON files")
    parser.add_argument("--results-file", "-f", type=str,
                       help="Single result file to analyze")
    parser.add_argument("--output-dir", "-o", type=str, default="results/experiments",
                       help="Output directory for analysis")

    # Analysis options
    parser.add_argument("--plots", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--tables", action="store_true",
                       help="Generate markdown tables")
    parser.add_argument("--all", action="store_true",
                       help="Generate all outputs")

    # Specific visualizations
    parser.add_argument("--plot-accuracy", action="store_true",
                       help="Plot accuracy comparison")
    parser.add_argument("--plot-confusion", action="store_true",
                       help="Plot confusion matrices")
    parser.add_argument("--plot-ablation", action="store_true",
                       help="Plot ablation study")

    # Statistical tests
    parser.add_argument("--significance", action="store_true",
                       help="Perform statistical significance tests")

    args = parser.parse_args()

    if args.all:
        args.plots = True
        args.tables = True

    if args.results_file:
        # Analyze single file
        results = load_results(args.results_file)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.plot_accuracy and "methods" in results:
            plot_accuracy_comparison(
                results,
                str(output_dir / "accuracy_comparison.png")
            )

        if args.plot_ablation and "configs" in results:
            plot_ablation_study(
                results,
                str(output_dir / "ablation_study.png")
            )

        if args.tables:
            if "methods" in results:
                table = generate_markdown_table(results, "comparison")
                print(table)
            elif "configs" in results:
                table = generate_markdown_table(results, "ablation")
                print(table)
            elif "datasets" in results:
                table = generate_markdown_table(results, "cross_dataset")
                print(table)

    else:
        # Analyze all results in directory
        analyze_results(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            create_plots=args.plots,
            create_tables=args.tables
        )


if __name__ == "__main__":
    main()
