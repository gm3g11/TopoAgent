#!/usr/bin/env python
"""
Benchmark4: Consolidation and Analysis.

Reads all 390 raw JSON results and produces:
  1. Master CSV: 26 datasets × 15 descriptors × 6 classifiers
  2. Classifier comparison: mean balanced accuracy per classifier
  3. Descriptor rankings: per object type
  4. Rule generalization: do exp4 rules hold on new datasets?
  5. Object type validation: within-type consistency
  6. Summary markdown report

Usage:
    python benchmarks/benchmark4/analyze_results.py
    python benchmarks/benchmark4/analyze_results.py --partial   # analyze whatever is available
    python benchmarks/benchmark4/analyze_results.py --csv-only  # just produce CSV
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

from RuleBenchmark.benchmark4.config import (
    DATASETS, ALL_DATASETS, ALL_DESCRIPTORS, ACTIVE_CLASSIFIERS,
    OBJECT_TYPE_DATASETS, RAW_RESULTS_PATH, SUMMARY_PATH,
    DESCRIPTORS, N_DATASETS, N_DESCRIPTORS, N_CLASSIFIERS,
    GRAYSCALE_DATASETS, RGB_DATASETS,
)


# =============================================================================
# 1. Load all raw results
# =============================================================================

def load_all_results(partial=False):
    """Load all raw JSON result files.

    Returns:
        results: dict[(dataset, descriptor)] -> parsed JSON dict
        missing: list of (dataset, descriptor) pairs with no results
    """
    results = {}
    missing = []
    errors = []

    for dataset in ALL_DATASETS:
        for desc in ALL_DESCRIPTORS:
            path = RAW_RESULTS_PATH / f"{dataset}_{desc}.json"
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if 'error' in data:
                        errors.append((dataset, desc, data['error']))
                    else:
                        results[(dataset, desc)] = data
                except Exception as e:
                    errors.append((dataset, desc, str(e)))
            else:
                missing.append((dataset, desc))

    total = N_DATASETS * N_DESCRIPTORS
    print(f"Loaded: {len(results)} / {total} results")
    print(f"Missing: {len(missing)}, Errors: {len(errors)}")

    if not partial and missing:
        print(f"\nWARNING: {len(missing)} missing results. Use --partial to analyze anyway.")

    if errors:
        print(f"\nErrors:")
        for d, desc, err in errors[:10]:
            print(f"  {d} x {desc}: {err[:80]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return results, missing, errors


# =============================================================================
# 2. Master CSV
# =============================================================================

def build_master_csv(results):
    """Build master CSV with all results.

    Columns: dataset, descriptor, object_type, color_mode, n_samples, n_classes,
             feature_dim, clf1_mean, clf1_std, clf2_mean, clf2_std, ...
    """
    rows = []
    classifiers = ACTIVE_CLASSIFIERS

    for (dataset, desc), data in sorted(results.items()):
        row = {
            'dataset': dataset,
            'descriptor': desc,
            'object_type': data['object_type'],
            'color_mode': data['color_mode'],
            'n_samples': data['n_samples'],
            'n_classes': data['n_classes'],
            'feature_dim': data.get('actual_feature_dim', data.get('total_feature_dim', 0)),
            'n_folds': data['n_folds'],
        }

        for clf_name in classifiers:
            if clf_name in data['classifiers']:
                clf_data = data['classifiers'][clf_name]
                row[f'{clf_name}_mean'] = clf_data['balanced_accuracy_mean']
                row[f'{clf_name}_std'] = clf_data['balanced_accuracy_std']
                row[f'{clf_name}_time'] = clf_data.get('time_seconds', 0)
            else:
                row[f'{clf_name}_mean'] = None
                row[f'{clf_name}_std'] = None
                row[f'{clf_name}_time'] = None

        # Best classifier for this (dataset, descriptor)
        best_clf = None
        best_acc = -1
        for clf_name in classifiers:
            acc = row.get(f'{clf_name}_mean')
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_clf = clf_name
        row['best_classifier'] = best_clf
        row['best_accuracy'] = best_acc

        rows.append(row)

    return rows


def write_csv(rows, output_path):
    """Write rows to CSV."""
    if not rows:
        print("No rows to write")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all column names from first row
    columns = list(rows[0].keys())
    with open(output_path, 'w') as f:
        f.write(','.join(columns) + '\n')
        for row in rows:
            values = []
            for col in columns:
                v = row.get(col, '')
                if v is None:
                    values.append('')
                elif isinstance(v, float):
                    values.append(f'{v:.6f}')
                else:
                    values.append(str(v))
            f.write(','.join(values) + '\n')

    print(f"  Saved CSV: {output_path} ({len(rows)} rows)")


# =============================================================================
# 3. Classifier comparison
# =============================================================================

def analyze_classifiers(results):
    """Compare classifiers across all (dataset, descriptor) pairs."""
    classifiers = ACTIVE_CLASSIFIERS
    clf_scores = {c: [] for c in classifiers}
    clf_scores_by_type = {c: defaultdict(list) for c in classifiers}

    for (dataset, desc), data in results.items():
        obj_type = data['object_type']
        for clf_name in classifiers:
            if clf_name in data['classifiers']:
                acc = data['classifiers'][clf_name]['balanced_accuracy_mean']
                clf_scores[clf_name].append(acc)
                clf_scores_by_type[clf_name][obj_type].append(acc)

    # Overall ranking
    clf_ranking = {}
    for clf_name in classifiers:
        scores = clf_scores[clf_name]
        if scores:
            clf_ranking[clf_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
                'n_evaluations': len(scores),
            }

    # By object type
    clf_by_type = {}
    for clf_name in classifiers:
        clf_by_type[clf_name] = {}
        for obj_type, scores in clf_scores_by_type[clf_name].items():
            if scores:
                clf_by_type[clf_name][obj_type] = {
                    'mean': float(np.mean(scores)),
                    'n': len(scores),
                }

    # Count how often each classifier is best
    best_counts = {c: 0 for c in classifiers}
    for (dataset, desc), data in results.items():
        best_clf = None
        best_acc = -1
        for clf_name in classifiers:
            if clf_name in data['classifiers']:
                acc = data['classifiers'][clf_name]['balanced_accuracy_mean']
                if acc > best_acc:
                    best_acc = acc
                    best_clf = clf_name
        if best_clf:
            best_counts[best_clf] += 1

    return {
        'overall': clf_ranking,
        'by_object_type': clf_by_type,
        'best_counts': best_counts,
    }


# =============================================================================
# 4. Descriptor rankings
# =============================================================================

def analyze_descriptors(results):
    """Rank descriptors per object type (best-classifier accuracy)."""
    # For each (dataset, descriptor), take the best classifier accuracy
    desc_scores = defaultdict(list)  # descriptor -> [best_acc, ...]
    desc_by_type = defaultdict(lambda: defaultdict(list))  # desc -> obj_type -> [best_acc, ...]
    desc_by_dataset = defaultdict(dict)  # desc -> dataset -> best_acc

    for (dataset, desc), data in results.items():
        obj_type = data['object_type']
        best_acc = max(
            clf['balanced_accuracy_mean']
            for clf in data['classifiers'].values()
        )
        desc_scores[desc].append(best_acc)
        desc_by_type[desc][obj_type].append(best_acc)
        desc_by_dataset[desc][dataset] = best_acc

    # Overall ranking
    desc_ranking = {}
    for desc in ALL_DESCRIPTORS:
        scores = desc_scores.get(desc, [])
        if scores:
            desc_ranking[desc] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
                'n_evaluations': len(scores),
            }

    # By object type ranking
    desc_by_type_ranking = {}
    for obj_type in OBJECT_TYPE_DATASETS:
        type_rankings = {}
        for desc in ALL_DESCRIPTORS:
            scores = desc_by_type.get(desc, {}).get(obj_type, [])
            if scores:
                type_rankings[desc] = {
                    'mean': float(np.mean(scores)),
                    'n': len(scores),
                }
        # Sort by mean descending
        desc_by_type_ranking[obj_type] = dict(
            sorted(type_rankings.items(), key=lambda x: x[1]['mean'], reverse=True)
        )

    return {
        'overall': desc_ranking,
        'by_object_type': desc_by_type_ranking,
        'by_dataset': {d: dict(v) for d, v in desc_by_dataset.items()},
    }


# =============================================================================
# 5. Best per dataset
# =============================================================================

def analyze_best_per_dataset(results):
    """Find the best (descriptor, classifier) for each dataset."""
    best_per_dataset = {}

    for dataset in ALL_DATASETS:
        best_acc = -1
        best_desc = None
        best_clf = None

        for desc in ALL_DESCRIPTORS:
            if (dataset, desc) not in results:
                continue
            data = results[(dataset, desc)]
            for clf_name, clf_data in data['classifiers'].items():
                acc = clf_data['balanced_accuracy_mean']
                if acc > best_acc:
                    best_acc = acc
                    best_desc = desc
                    best_clf = clf_name

        if best_desc:
            best_per_dataset[dataset] = {
                'best_descriptor': best_desc,
                'best_classifier': best_clf,
                'balanced_accuracy': float(best_acc),
                'object_type': DATASETS[dataset]['object_type'],
                'n_results': sum(1 for d in ALL_DESCRIPTORS if (dataset, d) in results),
            }

    return best_per_dataset


# =============================================================================
# 6. Object type validation
# =============================================================================

def analyze_object_types(results):
    """Check within-type consistency: do same-type datasets behave similarly?"""
    type_analysis = {}

    for obj_type, datasets in OBJECT_TYPE_DATASETS.items():
        # For each descriptor, get the rank across datasets of this type
        desc_means = {}
        for desc in ALL_DESCRIPTORS:
            scores = []
            for dataset in datasets:
                if (dataset, desc) in results:
                    data = results[(dataset, desc)]
                    best_acc = max(
                        c['balanced_accuracy_mean']
                        for c in data['classifiers'].values()
                    )
                    scores.append(best_acc)
            if scores:
                desc_means[desc] = float(np.mean(scores))

        # Rank descriptors within this object type
        if desc_means:
            ranked = sorted(desc_means.items(), key=lambda x: x[1], reverse=True)
            type_analysis[obj_type] = {
                'n_datasets': len(datasets),
                'datasets': datasets,
                'descriptor_ranking': [
                    {'descriptor': d, 'mean_accuracy': acc}
                    for d, acc in ranked
                ],
                'top_3': [d for d, _ in ranked[:3]],
            }

    return type_analysis


# =============================================================================
# 7. Summary markdown report
# =============================================================================

def generate_markdown(results, clf_analysis, desc_analysis,
                      best_per_dataset, type_analysis, missing, errors):
    """Generate a markdown summary report."""
    lines = []
    total = N_DATASETS * N_DESCRIPTORS
    n_done = len(results)

    lines.append("# Benchmark4 Results Summary")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n## Completion Status")
    lines.append(f"- **Results**: {n_done} / {total} ({100*n_done/total:.1f}%)")
    lines.append(f"- **Missing**: {len(missing)}")
    lines.append(f"- **Errors**: {len(errors)}")
    lines.append(f"- **Datasets**: {N_DATASETS}")
    lines.append(f"- **Descriptors**: {N_DESCRIPTORS}")
    lines.append(f"- **Classifiers**: {N_CLASSIFIERS} ({', '.join(ACTIVE_CLASSIFIERS)})")

    # Classifier comparison
    lines.append(f"\n## Classifier Comparison")
    lines.append(f"\n### Overall (mean balanced accuracy across all evaluations)")
    lines.append(f"| Classifier | Mean | Std | Median | N | #Best |")
    lines.append(f"|------------|------|-----|--------|---|-------|")
    clf_overall = clf_analysis['overall']
    best_counts = clf_analysis['best_counts']
    for clf_name in sorted(clf_overall.keys(),
                          key=lambda c: clf_overall[c]['mean'], reverse=True):
        stats = clf_overall[clf_name]
        lines.append(
            f"| {clf_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{stats['median']:.4f} | {stats['n_evaluations']} | "
            f"{best_counts.get(clf_name, 0)} |"
        )

    # Classifier by object type
    lines.append(f"\n### By Object Type (mean balanced accuracy)")
    obj_types = sorted(OBJECT_TYPE_DATASETS.keys())
    lines.append("| Classifier | " + " | ".join(obj_types) + " |")
    lines.append("|------------|" + "|".join(["---"] * len(obj_types)) + "|")
    for clf_name in ACTIVE_CLASSIFIERS:
        by_type = clf_analysis['by_object_type'].get(clf_name, {})
        vals = []
        for ot in obj_types:
            if ot in by_type:
                vals.append(f"{by_type[ot]['mean']:.4f}")
            else:
                vals.append("-")
        lines.append(f"| {clf_name} | " + " | ".join(vals) + " |")

    # Descriptor rankings
    lines.append(f"\n## Descriptor Rankings")
    lines.append(f"\n### Overall (best-classifier accuracy, averaged across datasets)")
    lines.append(f"| Rank | Descriptor | Mean | Std | N |")
    lines.append(f"|------|-----------|------|-----|---|")
    desc_overall = desc_analysis['overall']
    for rank, (desc, stats) in enumerate(
        sorted(desc_overall.items(), key=lambda x: x[1]['mean'], reverse=True), 1
    ):
        lines.append(
            f"| {rank} | {desc} | {stats['mean']:.4f} | "
            f"{stats['std']:.4f} | {stats['n_evaluations']} |"
        )

    # Descriptor rankings by object type
    for obj_type in obj_types:
        if obj_type in desc_analysis['by_object_type']:
            lines.append(f"\n### {obj_type}")
            type_ranks = desc_analysis['by_object_type'][obj_type]
            lines.append(f"| Rank | Descriptor | Mean | N |")
            lines.append(f"|------|-----------|------|---|")
            for rank, (desc, stats) in enumerate(type_ranks.items(), 1):
                lines.append(
                    f"| {rank} | {desc} | {stats['mean']:.4f} | {stats['n']} |"
                )

    # Best per dataset
    lines.append(f"\n## Best Results per Dataset")
    lines.append(f"| Dataset | Object Type | Best Descriptor | Best Classifier | Bal. Acc. | N |")
    lines.append(f"|---------|-------------|-----------------|-----------------|-----------|---|")
    for dataset in ALL_DATASETS:
        if dataset in best_per_dataset:
            b = best_per_dataset[dataset]
            lines.append(
                f"| {dataset} | {b['object_type']} | {b['best_descriptor']} | "
                f"{b['best_classifier']} | {b['balanced_accuracy']:.4f} | "
                f"{b['n_results']} |"
            )
        else:
            lines.append(f"| {dataset} | {DATASETS[dataset]['object_type']} | - | - | - | 0 |")

    # Object type validation
    lines.append(f"\n## Object Type Validation")
    for obj_type in obj_types:
        if obj_type in type_analysis:
            ta = type_analysis[obj_type]
            lines.append(f"\n### {obj_type} ({ta['n_datasets']} datasets: {', '.join(ta['datasets'])})")
            lines.append(f"Top 3 descriptors: {', '.join(ta['top_3'])}")

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark4 Analysis")
    parser.add_argument('--partial', action='store_true',
                       help='Analyze available results even if incomplete')
    parser.add_argument('--csv-only', action='store_true',
                       help='Only produce the master CSV')
    args = parser.parse_args()

    print("=" * 70)
    print("  BENCHMARK4 ANALYSIS")
    print("=" * 70)

    # Load results
    results, missing, errors = load_all_results(partial=args.partial or args.csv_only)

    if not results:
        print("\nNo results found. Run evaluation jobs first.")
        return

    # Build master CSV
    print("\n--- Building master CSV ---")
    rows = build_master_csv(results)
    SUMMARY_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_PATH / "benchmark4_all_results.csv"
    write_csv(rows, csv_path)

    if args.csv_only:
        return

    # Classifier analysis
    print("\n--- Classifier analysis ---")
    clf_analysis = analyze_classifiers(results)
    clf_path = SUMMARY_PATH / "classifier_rankings.json"
    with open(clf_path, 'w') as f:
        json.dump(clf_analysis, f, indent=2)
    print(f"  Saved: {clf_path}")

    # Print classifier summary
    print("\n  Classifier ranking (overall mean balanced accuracy):")
    for clf, stats in sorted(
        clf_analysis['overall'].items(),
        key=lambda x: x[1]['mean'], reverse=True
    ):
        print(f"    {clf:15s}  {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(n={stats['n_evaluations']})")
    print(f"\n  Best classifier counts: {clf_analysis['best_counts']}")

    # Descriptor analysis
    print("\n--- Descriptor analysis ---")
    desc_analysis = analyze_descriptors(results)
    desc_path = SUMMARY_PATH / "descriptor_rankings_by_type.json"
    with open(desc_path, 'w') as f:
        json.dump(desc_analysis, f, indent=2)
    print(f"  Saved: {desc_path}")

    # Print descriptor summary
    print("\n  Descriptor ranking (overall, best-classifier accuracy):")
    for desc, stats in sorted(
        desc_analysis['overall'].items(),
        key=lambda x: x[1]['mean'], reverse=True
    ):
        print(f"    {desc:35s}  {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(n={stats['n_evaluations']})")

    # Best per dataset
    print("\n--- Best per dataset ---")
    best_per_dataset = analyze_best_per_dataset(results)
    for dataset, b in best_per_dataset.items():
        print(f"  {dataset:25s}  {b['best_descriptor']:30s}  "
              f"{b['best_classifier']:12s}  {b['balanced_accuracy']:.4f}")

    # Object type validation
    print("\n--- Object type validation ---")
    type_analysis = analyze_object_types(results)
    for obj_type, ta in type_analysis.items():
        print(f"\n  {obj_type} ({ta['n_datasets']} datasets):")
        print(f"    Top 3: {', '.join(ta['top_3'])}")

    # Generate markdown report
    print("\n--- Generating markdown report ---")
    md_content = generate_markdown(
        results, clf_analysis, desc_analysis,
        best_per_dataset, type_analysis, missing, errors)
    md_path = SUMMARY_PATH / "benchmark4_summary.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"  Saved: {md_path}")

    # Save all analysis as single JSON
    all_analysis = {
        'generated': datetime.now().isoformat(),
        'n_results': len(results),
        'n_missing': len(missing),
        'n_errors': len(errors),
        'classifier_analysis': clf_analysis,
        'descriptor_analysis': {
            'overall': desc_analysis['overall'],
            'by_object_type': desc_analysis['by_object_type'],
        },
        'best_per_dataset': best_per_dataset,
        'object_type_analysis': type_analysis,
    }
    analysis_path = SUMMARY_PATH / "benchmark4_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(all_analysis, f, indent=2)
    print(f"  Saved: {analysis_path}")

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Results: {len(results)} / {N_DATASETS * N_DESCRIPTORS}")
    print(f"  Output: {SUMMARY_PATH}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
