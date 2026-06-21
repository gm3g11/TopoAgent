#!/usr/bin/env python
"""
Consolidate PH statistics from all 26 datasets into paper-ready tables.

Reads results/benchmark3/exp1/*_results.json and produces:
  - results/paper_ph_stats_26.json  (machine-readable)
  - results/paper_ph_stats_26.tex   (LaTeX tables)

Usage:
    python scripts/consolidate_ph_stats.py
    python scripts/consolidate_ph_stats.py --check   # just check which are missing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import numpy as np

# Full 26-dataset taxonomy
DATASET_INFO = {
    # discrete_cells (6)
    'BloodMNIST':         {'object_type': 'discrete_cells',  'color_mode': 'per_channel', 'n_classes': 8},
    'MalariaCell':        {'object_type': 'discrete_cells',  'color_mode': 'per_channel', 'n_classes': 2},
    'TissueMNIST':        {'object_type': 'discrete_cells',  'color_mode': 'grayscale',   'n_classes': 8},
    'PCam':               {'object_type': 'discrete_cells',  'color_mode': 'per_channel', 'n_classes': 2},
    'SIPaKMeD':           {'object_type': 'discrete_cells',  'color_mode': 'per_channel', 'n_classes': 5},
    'AML_Cytomorphology': {'object_type': 'discrete_cells',  'color_mode': 'per_channel', 'n_classes': 15},
    # glands_lumens (6)
    'PathMNIST':          {'object_type': 'glands_lumens',   'color_mode': 'per_channel', 'n_classes': 9},
    'Kvasir':             {'object_type': 'glands_lumens',   'color_mode': 'per_channel', 'n_classes': 8},
    'BreakHis':           {'object_type': 'glands_lumens',   'color_mode': 'per_channel', 'n_classes': 8},
    'NCT_CRC_HE':         {'object_type': 'glands_lumens',   'color_mode': 'per_channel', 'n_classes': 9},
    'LC25000':            {'object_type': 'glands_lumens',   'color_mode': 'per_channel', 'n_classes': 5},
    'Chaoyang':           {'object_type': 'glands_lumens',   'color_mode': 'per_channel', 'n_classes': 4},
    # vessel_trees (3)
    'RetinaMNIST':        {'object_type': 'vessel_trees',    'color_mode': 'per_channel', 'n_classes': 5},
    'APTOS2019':          {'object_type': 'vessel_trees',    'color_mode': 'per_channel', 'n_classes': 5},
    'IDRiD':              {'object_type': 'vessel_trees',    'color_mode': 'per_channel', 'n_classes': 5},
    # surface_lesions (3)
    'ISIC2019':           {'object_type': 'surface_lesions', 'color_mode': 'per_channel', 'n_classes': 8},
    'DermaMNIST':         {'object_type': 'surface_lesions', 'color_mode': 'per_channel', 'n_classes': 7},
    'GasHisSDB':          {'object_type': 'surface_lesions', 'color_mode': 'per_channel', 'n_classes': 2},
    # organ_shape (8)
    'OrganAMNIST':        {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 11},
    'OCTMNIST':           {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 4},
    'BrainTumorMRI':      {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 4},
    'MURA':               {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 14},
    'OrganCMNIST':        {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 11},
    'OrganSMNIST':        {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 11},
    'PneumoniaMNIST':     {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 2},
    'BreastMNIST':        {'object_type': 'organ_shape',     'color_mode': 'grayscale',   'n_classes': 2},
}

OBJECT_TYPE_ORDER = [
    'discrete_cells', 'glands_lumens', 'vessel_trees',
    'surface_lesions', 'organ_shape',
]

OBJECT_TYPE_LABELS = {
    'discrete_cells': 'Discrete cells',
    'glands_lumens': 'Glands \\& lumens',
    'vessel_trees': 'Vessel trees',
    'surface_lesions': 'Surface lesions',
    'organ_shape': 'Organ shape',
}


def load_all_ph_stats(exp1_dir):
    """Load avg_h0_bars and avg_h1_bars from all result JSONs."""
    stats = {}
    missing = []

    for ds in sorted(DATASET_INFO.keys()):
        result_file = exp1_dir / f"{ds}_results.json"
        if not result_file.exists():
            missing.append(ds)
            continue

        with open(result_file) as f:
            data = json.load(f)

        meta = data.get('metadata', data)
        h0 = meta.get('avg_h0_bars')
        h1 = meta.get('avg_h1_bars')

        if h0 is None or h1 is None:
            missing.append(ds)
            continue

        stats[ds] = {
            'avg_h0': float(h0),
            'avg_h1': float(h1),
            'ratio': round(float(h1) / float(h0), 2) if h0 > 0 else None,
            'n_samples': meta.get('n_samples'),
            'n_classes': meta.get('n_classes'),
            'object_type': DATASET_INFO[ds]['object_type'],
        }

    return stats, missing


def aggregate_by_type(stats):
    """Compute per-object-type aggregates."""
    agg = {}
    for ot in OBJECT_TYPE_ORDER:
        ds_in_type = [ds for ds, s in stats.items() if s['object_type'] == ot]
        if not ds_in_type:
            continue

        h0_vals = [stats[ds]['avg_h0'] for ds in ds_in_type]
        h1_vals = [stats[ds]['avg_h1'] for ds in ds_in_type]
        ratios = [stats[ds]['ratio'] for ds in ds_in_type if stats[ds]['ratio'] is not None]

        agg[ot] = {
            'datasets': ds_in_type,
            'n_datasets': len(ds_in_type),
            'mean_h0': round(np.mean(h0_vals)),
            'mean_h1': round(np.mean(h1_vals)),
            'std_h0': round(float(np.std(h0_vals)), 1),
            'std_h1': round(float(np.std(h1_vals)), 1),
            'beta_ratio': round(float(np.mean(ratios)), 2),
            'total_pairs': round(np.mean(h0_vals)) + round(np.mean(h1_vals)),
        }

    return agg


def fmt_num(n):
    """Format number with thousands separator for LaTeX."""
    n = int(round(n))
    if n >= 1000:
        return f"{n:,}".replace(",", "{,}")
    return str(n)


def generate_latex_per_dataset(stats):
    """Generate per-dataset table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Persistent homology statistics for all 26 datasets (sublevel filtration, $n{\leq}5{,}000$).}")
    lines.append(r"\label{tab:ph-stats-26}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Object Type} & \textbf{Dataset} & $N$ & $\bar\beta_0$ & $\bar\beta_1$ & $\beta_1/\beta_0$ \\")
    lines.append(r"\midrule")

    for ot in OBJECT_TYPE_ORDER:
        ds_in_type = sorted([ds for ds, s in stats.items() if s['object_type'] == ot])
        if not ds_in_type:
            continue

        label = OBJECT_TYPE_LABELS[ot]
        if len(ds_in_type) > 1:
            lines.append(f"\\multirow{{{len(ds_in_type)}}}{{*}}{{{label}}}")
        else:
            lines.append(f"{label}")

        for i, ds in enumerate(ds_in_type):
            s = stats[ds]
            ds_display = ds.replace('_', r'\_')
            n_samp = fmt_num(s['n_samples']) if s['n_samples'] else '---'
            h0 = fmt_num(s['avg_h0'])
            h1 = fmt_num(s['avg_h1'])
            ratio = f"{s['ratio']:.2f}" if s['ratio'] is not None else '---'
            prefix = "  & " if len(ds_in_type) > 1 else ""
            lines.append(f"  {prefix}{ds_display} & {n_samp} & {h0} & {h1} & {ratio} \\\\")

        # Add midrule between object types (not after last)
        if ot != OBJECT_TYPE_ORDER[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_aggregated(agg):
    """Generate aggregated table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Aggregated persistent homology profiles per object type (26 datasets).}")
    lines.append(r"\label{tab:ph-stats-aggregated-26}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Object Type} & $k$ & $\overline{\beta_0}$ & $\overline{\beta_1}$ & $\beta_1/\beta_0$ & \textbf{Total} & $\sigma_{\beta_0}$ \\")
    lines.append(r"\midrule")

    for ot in OBJECT_TYPE_ORDER:
        if ot not in agg:
            continue
        a = agg[ot]
        label = OBJECT_TYPE_LABELS[ot]
        lines.append(
            f"  {label} & {a['n_datasets']} & {fmt_num(a['mean_h0'])} & "
            f"{fmt_num(a['mean_h1'])} & {a['beta_ratio']:.2f} & "
            f"{fmt_num(a['total_pairs'])} & {a['std_h0']:.0f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true',
                        help='Only check which datasets are missing')
    parser.add_argument('--exp1-dir', type=str,
                        default='results/benchmark3/exp1')
    args = parser.parse_args()

    exp1_dir = Path(args.exp1_dir)
    stats, missing = load_all_ph_stats(exp1_dir)

    if args.check or missing:
        print(f"Found: {len(stats)}/26 datasets")
        if missing:
            print(f"Missing ({len(missing)}): {missing}")
        else:
            print("All 26 datasets have PH stats!")
        if args.check:
            return

    if missing:
        print(f"\nWARNING: {len(missing)} datasets missing. Run compute_ph_stats_missing.py first.")
        print(f"Generating tables with {len(stats)} datasets.\n")

    # Aggregate
    agg = aggregate_by_type(stats)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  PH Statistics Summary ({len(stats)} datasets)")
    print(f"{'='*80}")
    print(f"{'Object Type':<20} {'k':>3} {'Mean H0':>10} {'Mean H1':>10} {'B1/B0':>7} {'Total':>8}")
    print("-" * 65)
    for ot in OBJECT_TYPE_ORDER:
        if ot not in agg:
            continue
        a = agg[ot]
        print(f"{ot:<20} {a['n_datasets']:>3} {a['mean_h0']:>10,} {a['mean_h1']:>10,} "
              f"{a['beta_ratio']:>7.2f} {a['total_pairs']:>8,}")
    print()

    # Per-dataset detail
    print(f"{'Dataset':<25} {'Type':<18} {'N':>6} {'H0':>8} {'H1':>8} {'Ratio':>7}")
    print("-" * 80)
    for ot in OBJECT_TYPE_ORDER:
        ds_list = sorted([ds for ds, s in stats.items() if s['object_type'] == ot])
        for ds in ds_list:
            s = stats[ds]
            print(f"{ds:<25} {ot:<18} {s['n_samples']:>6} {s['avg_h0']:>8.0f} "
                  f"{s['avg_h1']:>8.0f} {s['ratio']:>7.2f}")
        print()

    # Save JSON
    output_json = Path('results/paper_ph_stats_26.json')
    output_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'n_datasets': len(stats),
        'per_dataset': stats,
        'aggregated': agg,
    }
    with open(output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {output_json}")

    # Save LaTeX
    output_tex = Path('results/paper_ph_stats_26.tex')
    tex_content = []
    tex_content.append("% Auto-generated by scripts/consolidate_ph_stats.py")
    tex_content.append(f"% {len(stats)} datasets\n")
    tex_content.append(generate_latex_per_dataset(stats))
    tex_content.append("\n")
    tex_content.append(generate_latex_aggregated(agg))
    with open(output_tex, 'w') as f:
        f.write("\n".join(tex_content) + "\n")
    print(f"Saved: {output_tex}")


if __name__ == '__main__':
    main()
