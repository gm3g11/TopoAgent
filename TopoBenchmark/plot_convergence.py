"""
Plot convergence analysis results for the paper.

Generates:
1. Main figure: 3 datasets × convergence curves with confidence bands
   - Subplot inset: Spearman rho ranking stability
2. Supplementary table: all 26 datasets with convergence_n

Usage:
    python TopoBenchmark/plot_convergence.py
    python TopoBenchmark/plot_convergence.py --datasets BloodMNIST,Kvasir,APTOS2019
    python TopoBenchmark/plot_convergence.py --all-table   # supplementary table only
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from RuleBenchmark.benchmark4.config import DATASETS

CONVERGENCE_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "convergence"
FIGURE_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "convergence" / "figures"

# Color palette for descriptors (colorblind-friendly)
DESCRIPTOR_COLORS = {
    'persistence_image': '#1f77b4',
    'persistence_landscapes': '#ff7f0e',
    'betti_curves': '#2ca02c',
    'persistence_silhouette': '#d62728',
    'persistence_entropy': '#9467bd',
    'persistence_statistics': '#8c564b',
    'tropical_coordinates': '#e377c2',
    'persistence_codebook': '#7f7f7f',
    'ATOL': '#bcbd22',
    'template_functions': '#17becf',
    'minkowski_functionals': '#aec7e8',
    'euler_characteristic_curve': '#ffbb78',
    'euler_characteristic_transform': '#98df8a',
    'edge_histogram': '#ff9896',
    'lbp_texture': '#c5b0d5',
}

# Short descriptor names for legend
SHORT_NAMES = {
    'persistence_image': 'PI',
    'persistence_landscapes': 'PL',
    'betti_curves': 'BC',
    'persistence_silhouette': 'PS',
    'persistence_entropy': 'PE',
    'persistence_statistics': 'PStat',
    'tropical_coordinates': 'TC',
    'persistence_codebook': 'PCB',
    'ATOL': 'ATOL',
    'template_functions': 'TF',
    'minkowski_functionals': 'MF',
    'euler_characteristic_curve': 'ECC',
    'euler_characteristic_transform': 'ECT',
    'edge_histogram': 'EH',
    'lbp_texture': 'LBP',
}


def load_convergence(dataset: str) -> Optional[Dict]:
    """Load convergence results for a dataset.

    Prefers extension files (higher n_max) over base files.
    """
    for suffix in ['_convergence_n10000.json', '_convergence_n5712.json', '_convergence.json']:
        path = CONVERGENCE_DIR / f"{dataset}{suffix}"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def plot_convergence_panel(
    ax,
    data: Dict,
    dataset: str,
    show_ylabel: bool = True,
    show_legend: bool = False,
):
    """Plot convergence curves for one dataset on a given axes."""
    descriptors = data['descriptors_tested']
    n_values = data['n_values']
    convergence = data.get('convergence', {})
    conv_n = convergence.get('convergence_n')

    for desc in descriptors:
        desc_results = data['results'].get(desc, {})
        ns = []
        means = []
        stds = []

        for n in n_values:
            entry = desc_results.get(str(n), {})
            m = entry.get('mean')
            s = entry.get('std')
            if m is not None:
                ns.append(n)
                means.append(m * 100)  # Convert to percentage
                stds.append((s or 0) * 100)

        if not ns:
            continue

        ns = np.array(ns)
        means = np.array(means)
        stds = np.array(stds)

        color = DESCRIPTOR_COLORS.get(desc, '#333333')
        short = SHORT_NAMES.get(desc, desc[:6])

        ax.plot(ns, means, '-o', color=color, label=short,
                markersize=4, linewidth=1.5)
        ax.fill_between(ns, means - stds, means + stds,
                        alpha=0.15, color=color)

    # Convergence line
    if conv_n is not None:
        ax.axvline(conv_n, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.annotate(f'n*={conv_n}', xy=(conv_n, ax.get_ylim()[0]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='gray')

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Sample size (n)', fontsize=10)
    if show_ylabel:
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=10)
    ax.set_title(dataset, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(labelsize=9)

    if show_legend:
        ax.legend(fontsize=7, loc='lower right', ncol=1, framealpha=0.8)


def plot_ranking_inset(
    ax,
    data: Dict,
):
    """Plot Spearman rho ranking stability as an inset."""
    ranking = data.get('convergence', {}).get('ranking_spearman', {})
    rho_per_n = ranking.get('rho_per_n', {})

    if not rho_per_n:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
        return

    ns = []
    rhos = []
    for n_str, rho in sorted(rho_per_n.items(), key=lambda x: int(x[0])):
        if rho is not None:
            ns.append(int(n_str))
            rhos.append(rho)

    if not ns:
        return

    ax.plot(ns, rhos, '-s', color='#2c3e50', markersize=3, linewidth=1.2)
    ax.axhline(0.9, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(r'$\rho$', fontsize=8)
    ax.set_xlabel('n', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, alpha=0.2)


def make_main_figure(
    datasets: List[str],
    output_path: Optional[Path] = None,
):
    """Generate the main convergence figure for the paper.

    3 panels (one per dataset) + ranking stability insets.
    """
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), squeeze=False)
    axes = axes[0]

    for i, dataset in enumerate(datasets):
        data = load_convergence(dataset)
        if data is None:
            axes[i].text(0.5, 0.5, f'{dataset}\n(no data)',
                         transform=axes[i].transAxes, ha='center', va='center')
            continue

        plot_convergence_panel(
            axes[i], data, dataset,
            show_ylabel=(i == 0),
            show_legend=True,
        )

        # Add ranking stability inset
        inset_ax = axes[i].inset_axes([0.55, 0.08, 0.4, 0.3])
        plot_ranking_inset(inset_ax, data)

    plt.tight_layout()

    if output_path is None:
        output_path = FIGURE_DIR / "convergence_main.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Main figure saved to {output_path}")


def make_supplementary_table(output_path: Optional[Path] = None):
    """Generate supplementary table: all datasets with convergence_n."""
    if output_path is None:
        output_path = FIGURE_DIR / "convergence_table.tex"

    rows = []
    for dataset in sorted(DATASETS.keys()):
        data = load_convergence(dataset)
        cfg = DATASETS[dataset]
        n_max = cfg['n_samples']
        n_classes = cfg['n_classes']

        if data is not None:
            conv = data.get('convergence', {})
            conv_n = conv.get('convergence_n')
            ranking = conv.get('ranking_spearman', {})
            rank_n = ranking.get('ranking_convergence_n')

            # Get accuracies at convergence point
            descs = data.get('descriptors_tested', [])
            acc_at_max = []
            for desc in descs:
                entry = data['results'].get(desc, {}).get(str(n_max), {})
                m = entry.get('mean')
                if m is not None:
                    acc_at_max.append(m)
            top_acc = max(acc_at_max) * 100 if acc_at_max else None
        else:
            conv_n = None
            rank_n = None
            top_acc = None

        rows.append({
            'dataset': dataset,
            'n_classes': n_classes,
            'n_max': n_max,
            'conv_n': conv_n,
            'rank_n': rank_n,
            'top_acc': top_acc,
        })

    # Print markdown table
    print(f"\n  Convergence Summary Table ({len(rows)} datasets)")
    print(f"  {'Dataset':<25} {'#Cls':>5} {'n_max':>6} {'conv_n':>7} {'rank_n':>7} {'Top Acc':>8}")
    print(f"  {'-'*60}")
    for r in rows:
        conv_str = str(r['conv_n']) if r['conv_n'] else '-'
        rank_str = str(r['rank_n']) if r['rank_n'] else '-'
        acc_str = f"{r['top_acc']:.1f}%" if r['top_acc'] else '-'
        print(f"  {r['dataset']:<25} {r['n_classes']:>5} {r['n_max']:>6} "
              f"{conv_str:>7} {rank_str:>7} {acc_str:>8}")

    # LaTeX table
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Convergence analysis results for all 26 datasets. "
                "$n^*$ is the smallest sample size where accuracy converges "
                "(within 1\\% of $n_{\\max}$ with $<$2\\% cross-seed variance). "
                "$n^*_{\\text{rank}}$ is where descriptor rankings stabilize "
                "(Spearman $\\rho > 0.9$).}\n")
        f.write("\\label{tab:convergence}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & \\#Cls & $n_{\\max}$ & $n^*$ & $n^*_{\\text{rank}}$ "
                "& Top Acc \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            conv_str = str(r['conv_n']) if r['conv_n'] else '--'
            rank_str = str(r['rank_n']) if r['rank_n'] else '--'
            acc_str = f"{r['top_acc']:.1f}\\%" if r['top_acc'] else '--'
            ds_name = r['dataset'].replace('_', '\\_')
            f.write(f"{ds_name} & {r['n_classes']} & {r['n_max']} & "
                    f"{conv_str} & {rank_str} & {acc_str} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\n  LaTeX table saved to {output_path}")


def make_dataset_table(output_path: Optional[Path] = None):
    """Generate a dataset overview table with sample counts and metadata."""
    if output_path is None:
        output_path = FIGURE_DIR / "dataset_table.tex"

    # Actual full training set sizes (counted from original data sources)
    ACTUAL_SIZES = {
        'BloodMNIST': 11959, 'TissueMNIST': 165466, 'PathMNIST': 89996,
        'OCTMNIST': 97477, 'OrganAMNIST': 34561, 'OrganCMNIST': 12975,
        'OrganSMNIST': 13932, 'DermaMNIST': 7007, 'BreastMNIST': 546,
        'RetinaMNIST': 1080, 'PneumoniaMNIST': 4708,
        'ISIC2019': 25331, 'Kvasir': 4000, 'BrainTumorMRI': 5712,
        'MURA': 40005, 'BreakHis': 7909, 'NCT_CRC_HE': 100000,
        'MalariaCell': 55116, 'IDRiD': 516, 'PCam': 262144,
        'LC25000': 25000, 'SIPaKMeD': 5015, 'AML_Cytomorphology': 21546,
        'APTOS2019': 3296, 'GasHisSDB': 33284, 'Chaoyang': 6160,
    }

    # Try to load frozen config for frozen n values
    frozen_config_path = Path(__file__).parent / "frozen_dataset_config.json"
    frozen = {}
    if frozen_config_path.exists():
        with open(frozen_config_path) as f:
            frozen = json.load(f).get('datasets', {})

    rows = []
    for dataset in sorted(DATASETS.keys()):
        cfg = DATASETS[dataset]
        frozen_info = frozen.get(dataset, {})
        rows.append({
            'dataset': dataset,
            'source': cfg['source'],
            'object_type': cfg['object_type'],
            'n_classes': cfg['n_classes'],
            'dataset_size': ACTUAL_SIZES.get(dataset, cfg['n_samples']),
            'n': frozen_info.get('n', cfg['n_samples']),
            'image_size': cfg['image_size'],
            'color_mode': cfg['color_mode'],
        })

    # Print markdown table
    print(f"\n  Dataset Overview Table ({len(rows)} datasets)")
    print(f"  {'Dataset':<25} {'Source':<12} {'Object Type':<18} {'#Cls':>5} "
          f"{'Size':>8} {'n':>6} {'ImgSize':>8} {'Color':>6}")
    print(f"  {'-'*94}")
    total_n = 0
    for r in rows:
        img_str = str(r['image_size']) if r['image_size'] != 'variable' else 'var'
        color_str = 'RGB' if r['color_mode'] == 'per_channel' else 'gray'
        print(f"  {r['dataset']:<25} {r['source']:<12} {r['object_type']:<18} "
              f"{r['n_classes']:>5} {r['dataset_size']:>8} {r['n']:>6} "
              f"{img_str:>8} {color_str:>6}")
        total_n += r['n']
    print(f"  {'-'*94}")
    print(f"  Total benchmark samples: {total_n}")

    # Short object type names for LaTeX
    ot_short = {
        'discrete_cells': 'Cells',
        'glands_lumens': 'Glands',
        'organ_shape': 'Organ',
        'surface_lesions': 'Lesion',
        'vessel_trees': 'Vessel',
    }

    # LaTeX table
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Overview of the 26 datasets in TopoBenchmark. "
                "Size is the full training set. "
                "$n$ is the sample size used after convergence analysis. "
                "Color: G = grayscale, RGB = per-channel.}\n")
        f.write("\\label{tab:datasets}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llrrrrl}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Object Type & \\#Cls & Size & $n$ "
                "& Img Size & Color \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            ds_name = r['dataset'].replace('_', '\\_')
            ot = ot_short.get(r['object_type'], r['object_type'])
            img_str = str(r['image_size']) if r['image_size'] != 'variable' else 'var.'
            color_str = 'RGB' if r['color_mode'] == 'per_channel' else 'G'
            f.write(f"{ds_name} & {ot} & {r['n_classes']} & {r['dataset_size']} & "
                    f"{r['n']} & {img_str} & {color_str} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\n  LaTeX dataset table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot convergence results')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated datasets for main figure '
                             '(default: BloodMNIST,DermaMNIST,OrganAMNIST)')
    parser.add_argument('--all-table', action='store_true',
                        help='Only generate supplementary table')
    parser.add_argument('--dataset-table', action='store_true',
                        help='Generate dataset overview table')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    global FIGURE_DIR
    if args.output_dir:
        FIGURE_DIR = Path(args.output_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset_table:
        make_dataset_table()
        return

    if args.all_table:
        make_supplementary_table()
        return

    # Default showcase datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        # Pick datasets that tell different convergence stories:
        # BloodMNIST: Fast convergence (n*=3000, well below n_max=5000)
        # DermaMNIST: Needed extension (n*=7500 via n10000 run)
        # OrganAMNIST: Full dataset used (n*=5000=n_max, still improving)
        datasets = ['BloodMNIST', 'DermaMNIST', 'OrganAMNIST']

    print("=" * 60)
    print("  Convergence Plot Generation")
    print("=" * 60)

    # Check which datasets have data
    available = []
    for ds in datasets:
        data = load_convergence(ds)
        if data is not None:
            available.append(ds)
        else:
            print(f"  WARNING: No convergence data for {ds}")

    if not available:
        # Fall back to whatever we have
        for ds in sorted(DATASETS.keys()):
            if load_convergence(ds) is not None:
                available.append(ds)
                if len(available) >= 3:
                    break
        if not available:
            print("  ERROR: No convergence results found.")
            return
        print(f"  Falling back to available datasets: {available}")

    # Main figure
    print(f"\n  Generating main figure for: {available}")
    make_main_figure(available)

    # Supplementary table
    print(f"\n  Generating supplementary table...")
    make_supplementary_table()

    # Dataset overview table
    print(f"\n  Generating dataset overview table...")
    make_dataset_table()

    print("\n  Done!")


if __name__ == '__main__':
    main()
