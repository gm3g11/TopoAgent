#!/usr/bin/env python
"""
Descriptor Correlation Analysis

Analyze correlations between descriptors to identify:
1. Redundant descriptors (high correlation)
2. Complementary descriptors (low correlation)
3. Information groups (A: Spatial, B: Temporal, C: Aggregate)

Usage:
    # Analyze correlations from benchmark results
    python scripts/compute_correlations.py --results-dir results/benchmark1 --dataset dermamnist

    # Analyze all datasets
    python scripts/compute_correlations.py --results-dir results/benchmark1 --all-datasets

    # Generate visualization
    python scripts/compute_correlations.py --results-dir results/benchmark1 --visualize
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Descriptor Groups (from plan)
# =============================================================================

# Descriptors grouped by information type
DESCRIPTOR_GROUPS = {
    'A_Spatial': [
        'persistence_image',
        'tropical_coordinates',
        'atol',
        'persistence_codebook'
    ],
    'B_Temporal': [
        'betti_curves',
        'euler_characteristic_curve',
        'persistence_landscapes',
        'persistence_silhouette'
    ],
    'C_Aggregate': [
        'persistence_statistics',
        'persistence_entropy',
        'carlsson_coordinates'
    ],
    'Other': [
        'minkowski_functionals',
        'lbp_texture'
    ]
}

# Expected high correlations (to verify)
EXPECTED_HIGH_CORRELATIONS = [
    ('persistence_landscapes', 'persistence_silhouette', 0.95),  # Silhouette = weighted mean of Landscapes
    ('persistence_statistics', 'carlsson_coordinates', 0.90),   # Carlsson is subset of Statistics
    ('persistence_image', 'persistence_landscapes', 0.80),      # Both discretize same diagram
]


# =============================================================================
# Correlation Loading and Analysis
# =============================================================================

def load_correlations(results_dir: Path, dataset_name: str) -> Tuple[np.ndarray, List[str]]:
    """Load correlation matrix from benchmark results.

    Args:
        results_dir: Directory with benchmark results
        dataset_name: Name of dataset

    Returns:
        corr_matrix: (M, M) correlation matrix
        descriptor_names: List of descriptor names
    """
    corr_path = results_dir / f"{dataset_name}_correlations.npz"

    if not corr_path.exists():
        raise FileNotFoundError(f"Correlation file not found: {corr_path}")

    data = np.load(corr_path, allow_pickle=True)
    corr_matrix = data['corr_matrix']
    descriptor_names = list(data['descriptor_names'])

    return corr_matrix, descriptor_names


def load_features(results_dir: Path, dataset_name: str) -> Dict[str, np.ndarray]:
    """Load extracted features from benchmark results.

    Args:
        results_dir: Directory with benchmark results
        dataset_name: Name of dataset

    Returns:
        features: Dictionary of descriptor_name -> (N, D) features
    """
    features_path = results_dir / f"{dataset_name}_features.npz"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            f"Re-run benchmark with --save-features flag"
        )

    data = np.load(features_path, allow_pickle=True)

    # Extract features (everything except 'labels')
    features = {}
    for key in data.files:
        if key != 'labels':
            features[key] = data[key]

    return features


def compute_detailed_correlations(
    features: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Compute detailed correlation analysis.

    Args:
        features: Dictionary of descriptor_name -> (N, D) features

    Returns:
        Dictionary with correlation analyses
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    desc_names = list(features.keys())
    n_desc = len(desc_names)

    # 1. PCA-reduced correlation (as in benchmark)
    target_dim = 50
    reduced_features = {}

    for name, X in features.items():
        X_scaled = StandardScaler().fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        actual_dim = min(target_dim, X.shape[1], X.shape[0] - 1)
        if actual_dim > 0:
            pca = PCA(n_components=actual_dim, random_state=42)
            reduced_features[name] = pca.fit_transform(X_scaled)
        else:
            reduced_features[name] = X_scaled[:, :1]

    # Pearson correlation matrix
    pearson_corr = np.zeros((n_desc, n_desc))
    for i, name_i in enumerate(desc_names):
        for j, name_j in enumerate(desc_names):
            if i == j:
                pearson_corr[i, j] = 1.0
            elif i < j:
                Xi = reduced_features[name_i]
                Xj = reduced_features[name_j]
                min_dim = min(Xi.shape[1], Xj.shape[1])
                corrs = []
                for k in range(min_dim):
                    c = np.corrcoef(Xi[:, k], Xj[:, k])[0, 1]
                    if np.isfinite(c):
                        corrs.append(abs(c))
                pearson_corr[i, j] = np.mean(corrs) if corrs else 0.0
                pearson_corr[j, i] = pearson_corr[i, j]

    # 2. Spearman correlation (rank-based)
    spearman_corr = np.zeros((n_desc, n_desc))
    for i, name_i in enumerate(desc_names):
        for j, name_j in enumerate(desc_names):
            if i == j:
                spearman_corr[i, j] = 1.0
            elif i < j:
                Xi = reduced_features[name_i]
                Xj = reduced_features[name_j]
                min_dim = min(Xi.shape[1], Xj.shape[1])
                corrs = []
                for k in range(min_dim):
                    rho, _ = spearmanr(Xi[:, k], Xj[:, k])
                    if np.isfinite(rho):
                        corrs.append(abs(rho))
                spearman_corr[i, j] = np.mean(corrs) if corrs else 0.0
                spearman_corr[j, i] = spearman_corr[i, j]

    # 3. Canonical Correlation Analysis (CCA)
    try:
        from sklearn.cross_decomposition import CCA

        cca_corr = np.zeros((n_desc, n_desc))
        for i, name_i in enumerate(desc_names):
            for j, name_j in enumerate(desc_names):
                if i == j:
                    cca_corr[i, j] = 1.0
                elif i < j:
                    Xi = reduced_features[name_i]
                    Xj = reduced_features[name_j]

                    n_comp = min(5, Xi.shape[1], Xj.shape[1])
                    cca = CCA(n_components=n_comp)
                    Xi_c, Xj_c = cca.fit_transform(Xi, Xj)

                    # Mean correlation of canonical variates
                    corrs = []
                    for k in range(n_comp):
                        c = np.corrcoef(Xi_c[:, k], Xj_c[:, k])[0, 1]
                        if np.isfinite(c):
                            corrs.append(abs(c))
                    cca_corr[i, j] = np.mean(corrs) if corrs else 0.0
                    cca_corr[j, i] = cca_corr[i, j]

    except Exception as e:
        print(f"  Warning: CCA failed: {e}")
        cca_corr = pearson_corr.copy()

    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'cca': cca_corr,
        'descriptor_names': desc_names
    }


def analyze_group_correlations(
    corr_matrix: np.ndarray,
    descriptor_names: List[str]
) -> Dict[str, Dict]:
    """Analyze within-group and between-group correlations.

    Args:
        corr_matrix: (M, M) correlation matrix
        descriptor_names: List of descriptor names

    Returns:
        Dictionary with group correlation analysis
    """
    results = {
        'within_group': {},
        'between_group': {},
        'summary': {}
    }

    # Create name to index mapping
    name_to_idx = {name: i for i, name in enumerate(descriptor_names)}

    # Compute within-group correlations
    for group_name, group_descs in DESCRIPTOR_GROUPS.items():
        # Filter to descriptors that exist in our results
        valid_descs = [d for d in group_descs if d in name_to_idx]

        if len(valid_descs) < 2:
            results['within_group'][group_name] = {'mean': 0.0, 'std': 0.0, 'n_pairs': 0}
            continue

        correlations = []
        for i, d1 in enumerate(valid_descs):
            for d2 in valid_descs[i+1:]:
                idx1, idx2 = name_to_idx[d1], name_to_idx[d2]
                correlations.append(corr_matrix[idx1, idx2])

        results['within_group'][group_name] = {
            'mean': float(np.mean(correlations)),
            'std': float(np.std(correlations)),
            'n_pairs': len(correlations),
            'descriptors': valid_descs
        }

    # Compute between-group correlations
    group_names = list(DESCRIPTOR_GROUPS.keys())
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            descs1 = [d for d in DESCRIPTOR_GROUPS[g1] if d in name_to_idx]
            descs2 = [d for d in DESCRIPTOR_GROUPS[g2] if d in name_to_idx]

            if not descs1 or not descs2:
                continue

            correlations = []
            for d1 in descs1:
                for d2 in descs2:
                    idx1, idx2 = name_to_idx[d1], name_to_idx[d2]
                    correlations.append(corr_matrix[idx1, idx2])

            results['between_group'][f"{g1}_vs_{g2}"] = {
                'mean': float(np.mean(correlations)),
                'std': float(np.std(correlations)),
                'n_pairs': len(correlations)
            }

    # Summary statistics
    within_means = [v['mean'] for v in results['within_group'].values() if v['n_pairs'] > 0]
    between_means = [v['mean'] for v in results['between_group'].values() if v['n_pairs'] > 0]

    results['summary'] = {
        'avg_within_group': float(np.mean(within_means)) if within_means else 0.0,
        'avg_between_group': float(np.mean(between_means)) if between_means else 0.0,
        'group_separation': float(np.mean(within_means) - np.mean(between_means)) if within_means and between_means else 0.0
    }

    return results


def verify_expected_correlations(
    corr_matrix: np.ndarray,
    descriptor_names: List[str]
) -> List[Dict]:
    """Verify expected high correlations from plan.

    Args:
        corr_matrix: (M, M) correlation matrix
        descriptor_names: List of descriptor names

    Returns:
        List of verification results
    """
    name_to_idx = {name: i for i, name in enumerate(descriptor_names)}
    results = []

    for desc1, desc2, expected in EXPECTED_HIGH_CORRELATIONS:
        if desc1 not in name_to_idx or desc2 not in name_to_idx:
            results.append({
                'pair': (desc1, desc2),
                'expected': expected,
                'actual': None,
                'verified': False,
                'reason': 'Descriptor not found in results'
            })
            continue

        idx1, idx2 = name_to_idx[desc1], name_to_idx[desc2]
        actual = corr_matrix[idx1, idx2]

        results.append({
            'pair': (desc1, desc2),
            'expected': expected,
            'actual': float(actual),
            'verified': actual >= expected * 0.8,  # Within 80% of expected
            'diff': float(actual - expected)
        })

    return results


def find_redundant_pairs(
    corr_matrix: np.ndarray,
    descriptor_names: List[str],
    threshold: float = 0.85
) -> List[Tuple[str, str, float]]:
    """Find highly correlated (potentially redundant) descriptor pairs.

    Args:
        corr_matrix: (M, M) correlation matrix
        descriptor_names: List of descriptor names
        threshold: Correlation threshold for redundancy

    Returns:
        List of (desc1, desc2, correlation) tuples
    """
    redundant = []
    n = len(descriptor_names)

    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] >= threshold:
                redundant.append((
                    descriptor_names[i],
                    descriptor_names[j],
                    float(corr_matrix[i, j])
                ))

    return sorted(redundant, key=lambda x: x[2], reverse=True)


def find_complementary_pairs(
    corr_matrix: np.ndarray,
    descriptor_names: List[str],
    threshold: float = 0.3
) -> List[Tuple[str, str, float]]:
    """Find low-correlated (potentially complementary) descriptor pairs.

    Args:
        corr_matrix: (M, M) correlation matrix
        descriptor_names: List of descriptor names
        threshold: Correlation threshold for complementarity

    Returns:
        List of (desc1, desc2, correlation) tuples
    """
    complementary = []
    n = len(descriptor_names)

    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] <= threshold:
                complementary.append((
                    descriptor_names[i],
                    descriptor_names[j],
                    float(corr_matrix[i, j])
                ))

    return sorted(complementary, key=lambda x: x[2])


# =============================================================================
# Visualization
# =============================================================================

def create_correlation_heatmap(
    corr_matrix: np.ndarray,
    descriptor_names: List[str],
    output_path: Path,
    title: str = "Descriptor Correlation Matrix"
):
    """Create correlation heatmap visualization.

    Args:
        corr_matrix: (M, M) correlation matrix
        descriptor_names: List of descriptor names
        output_path: Output file path
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(14, 12))

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0.5,
            square=True,
            xticklabels=descriptor_names,
            yticklabels=descriptor_names,
            cbar_kws={'label': 'Correlation'}
        )

        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved heatmap to {output_path}")

    except ImportError:
        print("  Warning: matplotlib/seaborn not installed, skipping visualization")


def create_group_correlation_plot(
    group_results: Dict,
    output_path: Path,
    title: str = "Group Correlation Analysis"
):
    """Create bar plot of within-group vs between-group correlations.

    Args:
        group_results: Results from analyze_group_correlations
        output_path: Output file path
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Within-group correlations
        ax1 = axes[0]
        groups = list(group_results['within_group'].keys())
        means = [group_results['within_group'][g]['mean'] for g in groups]
        stds = [group_results['within_group'][g]['std'] for g in groups]

        bars = ax1.bar(range(len(groups)), means, yerr=stds, capsize=5,
                       color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
        ax1.set_xticks(range(len(groups)))
        ax1.set_xticklabels(groups, rotation=45, ha='right')
        ax1.set_ylabel('Mean Correlation')
        ax1.set_title('Within-Group Correlations')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Between-group correlations
        ax2 = axes[1]
        between_pairs = list(group_results['between_group'].keys())
        between_means = [group_results['between_group'][p]['mean'] for p in between_pairs]

        ax2.barh(range(len(between_pairs)), between_means, color='#95a5a6')
        ax2.set_yticks(range(len(between_pairs)))
        ax2.set_yticklabels([p.replace('_vs_', ' vs\n') for p in between_pairs])
        ax2.set_xlabel('Mean Correlation')
        ax2.set_title('Between-Group Correlations')
        ax2.set_xlim(0, 1)
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved group plot to {output_path}")

    except ImportError:
        print("  Warning: matplotlib not installed, skipping visualization")


# =============================================================================
# Reporting
# =============================================================================

def print_correlation_report(
    corr_matrix: np.ndarray,
    descriptor_names: List[str],
    dataset_name: str
):
    """Print detailed correlation analysis report."""
    print(f"\n{'='*80}")
    print(f"Correlation Analysis Report: {dataset_name}")
    print(f"{'='*80}\n")

    # 1. Overall statistics
    upper_tri = corr_matrix[np.triu_indices(len(descriptor_names), k=1)]
    print("Overall Correlation Statistics:")
    print(f"  Mean: {np.mean(upper_tri):.3f}")
    print(f"  Std:  {np.std(upper_tri):.3f}")
    print(f"  Min:  {np.min(upper_tri):.3f}")
    print(f"  Max:  {np.max(upper_tri):.3f}")

    # 2. Verify expected correlations
    print("\nExpected High Correlations (from plan):")
    verifications = verify_expected_correlations(corr_matrix, descriptor_names)
    for v in verifications:
        status = "VERIFIED" if v['verified'] else "NOT VERIFIED"
        if v['actual'] is not None:
            print(f"  {v['pair'][0]} <-> {v['pair'][1]}: "
                  f"expected={v['expected']:.2f}, actual={v['actual']:.2f} [{status}]")
        else:
            print(f"  {v['pair'][0]} <-> {v['pair'][1]}: {v['reason']}")

    # 3. Group analysis
    print("\nGroup Correlation Analysis:")
    group_results = analyze_group_correlations(corr_matrix, descriptor_names)

    print("\n  Within-Group Correlations (should be HIGH):")
    for group, stats in group_results['within_group'].items():
        if stats['n_pairs'] > 0:
            print(f"    {group}: {stats['mean']:.3f} (+/- {stats['std']:.3f}), n={stats['n_pairs']}")

    print("\n  Between-Group Correlations (should be LOWER):")
    for pair, stats in sorted(group_results['between_group'].items(),
                              key=lambda x: x[1]['mean'], reverse=True)[:5]:
        print(f"    {pair}: {stats['mean']:.3f} (+/- {stats['std']:.3f})")

    print(f"\n  Group Separation Score: {group_results['summary']['group_separation']:.3f}")
    print(f"    (Higher = better group structure)")

    # 4. Redundant pairs
    print("\nPotentially Redundant Pairs (correlation > 0.85):")
    redundant = find_redundant_pairs(corr_matrix, descriptor_names, threshold=0.85)
    if redundant:
        for d1, d2, corr in redundant[:5]:
            print(f"  {d1} <-> {d2}: {corr:.3f}")
    else:
        print("  None found (all pairs have correlation < 0.85)")

    # 5. Complementary pairs (good for ensembles)
    print("\nComplementary Pairs (correlation < 0.3, good for ensembles):")
    complementary = find_complementary_pairs(corr_matrix, descriptor_names, threshold=0.3)
    if complementary:
        for d1, d2, corr in complementary[:5]:
            print(f"  {d1} <-> {d2}: {corr:.3f}")
    else:
        print("  None found (all pairs have correlation > 0.3)")

    print(f"\n{'='*80}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Descriptor Correlation Analysis"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/benchmark1",
        help="Directory with benchmark results"
    )
    parser.add_argument(
        "--dataset", type=str, default="dermamnist",
        help="Dataset name"
    )
    parser.add_argument(
        "--all-datasets", action="store_true",
        help="Analyze all datasets"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for plots (default: results_dir/correlations)"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Compute detailed correlations (requires saved features)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / "correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['dermamnist', 'bloodmnist', 'pathmnist', 'retinamnist', 'organamnist'] \
        if args.all_datasets else [args.dataset]

    all_reports = {}

    for dataset_name in datasets:
        try:
            print(f"\nAnalyzing {dataset_name}...")

            # Load correlation matrix
            corr_matrix, descriptor_names = load_correlations(results_dir, dataset_name)

            # Print report
            print_correlation_report(corr_matrix, descriptor_names, dataset_name)

            # Store for summary
            group_results = analyze_group_correlations(corr_matrix, descriptor_names)
            all_reports[dataset_name] = {
                'group_analysis': group_results,
                'redundant_pairs': find_redundant_pairs(corr_matrix, descriptor_names),
                'complementary_pairs': find_complementary_pairs(corr_matrix, descriptor_names),
                'expected_verification': verify_expected_correlations(corr_matrix, descriptor_names)
            }

            # Visualization
            if args.visualize:
                create_correlation_heatmap(
                    corr_matrix, descriptor_names,
                    output_dir / f"{dataset_name}_correlation_heatmap.png",
                    f"Descriptor Correlations - {dataset_name}"
                )
                create_group_correlation_plot(
                    group_results,
                    output_dir / f"{dataset_name}_group_correlations.png",
                    f"Group Analysis - {dataset_name}"
                )

            # Detailed analysis if requested
            if args.detailed:
                try:
                    features = load_features(results_dir, dataset_name)
                    detailed = compute_detailed_correlations(features)

                    # Save detailed correlations
                    detailed_path = output_dir / f"{dataset_name}_detailed_correlations.npz"
                    np.savez(
                        detailed_path,
                        pearson=detailed['pearson'],
                        spearman=detailed['spearman'],
                        cca=detailed['cca'],
                        descriptor_names=detailed['descriptor_names']
                    )
                    print(f"  Saved detailed correlations to {detailed_path}")

                except FileNotFoundError as e:
                    print(f"  Warning: {e}")

        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue

    # Save summary report
    if len(all_reports) > 0:
        summary_path = output_dir / "correlation_summary.json"

        def convert_to_serializable(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            return obj

        # Convert numpy arrays to lists for JSON serialization
        serializable_reports = {}
        for dataset, report in all_reports.items():
            serializable_reports[dataset] = {
                'group_analysis': convert_to_serializable(report['group_analysis']),
                'redundant_pairs': [(d1, d2, float(c)) for d1, d2, c in report['redundant_pairs']],
                'complementary_pairs': [(d1, d2, float(c)) for d1, d2, c in report['complementary_pairs']],
                'expected_verification': convert_to_serializable(report['expected_verification'])
            }

        with open(summary_path, 'w') as f:
            json.dump(serializable_reports, f, indent=2)
        print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
