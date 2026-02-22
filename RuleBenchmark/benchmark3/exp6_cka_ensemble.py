#!/usr/bin/env python
"""
Benchmark3 Experiment 6: CKA + Ensemble Analysis

Measures descriptor complementarity via CKA and finds optimal ensembles
across 15 descriptors x 13 datasets.

Key questions:
1. Does benchmark2's best ensemble (Stats+ATOL+LBP) still win on 13 datasets?
2. Are new descriptors (template_functions, ECT, edge_histogram) complementary?
3. What's the optimal ensemble per dataset and overall?

Usage:
    # Quick test (200 samples, 1 dataset)
    python benchmarks/benchmark3/exp6_cka_ensemble.py --dataset BloodMNIST --n-samples 200

    # Single dataset (full)
    python benchmarks/benchmark3/exp6_cka_ensemble.py --dataset BloodMNIST

    # All 13 datasets
    python benchmarks/benchmark3/exp6_cka_ensemble.py --all-datasets

    # Test mode (200 samples)
    python benchmarks/benchmark3/exp6_cka_ensemble.py --all-datasets --test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from RuleBenchmark.benchmark3.config import (
    ALL_DATASETS, ALL_DESCRIPTORS, MANY_CLASS_DATASETS,
    DESCRIPTORS, RESULTS_PATH,
    get_classifier, get_output_dir,
)

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42
N_FOLDS = 5
PRIMARY_CLASSIFIER = 'TabPFN'

# Feature directory (pre-computed from exp1)
EXP1_DIR = RESULTS_PATH / 'exp1'

# 10 ensemble configurations
ENSEMBLE_CONFIGS = [
    # --- Benchmark2 best (validate) ---
    ['persistence_statistics', 'ATOL', 'lbp_texture'],
    ['persistence_statistics', 'persistence_image', 'ATOL'],

    # --- New descriptor combos ---
    ['persistence_statistics', 'ATOL', 'template_functions'],
    ['persistence_statistics', 'ATOL', 'edge_histogram'],
    ['persistence_codebook', 'ATOL', 'template_functions'],

    # --- High-dim combos ---
    ['persistence_image', 'euler_characteristic_transform'],
    ['persistence_image', 'ATOL', 'template_functions'],

    # --- Efficient combos ---
    ['persistence_statistics', 'template_functions'],
    ['ATOL', 'lbp_texture', 'edge_histogram'],

    # --- Kitchen sink (top-5 from exp1) ---
    ['persistence_codebook', 'ATOL', 'template_functions',
     'persistence_statistics', 'lbp_texture'],
]


# =============================================================================
# CKA COMPUTATION
# =============================================================================

def compute_cka(X1: np.ndarray, X2: np.ndarray) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X1: (n_samples, d1) - already standardized
        X2: (n_samples, d2) - already standardized

    Returns:
        CKA value in [0, 1]
    """
    assert X1.shape[0] == X2.shape[0], "Same number of samples required"

    def centering(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def hsic(K, L):
        return np.sum(centering(K) * centering(L)) / ((K.shape[0] - 1) ** 2)

    K = X1 @ X1.T
    L = X2 @ X2.T

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    if hsic_kk * hsic_ll == 0:
        return 0.0

    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))


def compute_cka_matrix(features_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute 15x15 CKA matrix between all descriptor pairs.

    Applies StandardScaler normalization before CKA.

    Args:
        features_dict: {descriptor_name: (n_samples, dim) array}

    Returns:
        (n_descriptors, n_descriptors) symmetric CKA matrix
    """
    names = list(features_dict.keys())
    n = len(names)
    cka_matrix = np.eye(n)

    # Pre-standardize all features
    scaled = {}
    for name in names:
        X = features_dict[name].copy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Skip constant features (std=0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scaled[name] = X

    for i in range(n):
        for j in range(i + 1, n):
            cka = compute_cka(scaled[names[i]], scaled[names[j]])
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka

    return cka_matrix


# =============================================================================
# CLASSIFICATION
# =============================================================================

def evaluate_features(
    features: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    seed: int = RANDOM_SEED,
    n_folds: int = N_FOLDS,
) -> Dict[str, float]:
    """Evaluate features with TabPFN using stratified K-fold CV.

    Returns dict with accuracy_mean, accuracy_std, balanced_accuracy_mean,
    balanced_accuracy_std, f1_macro_mean, f1_macro_std, cohen_kappa_mean.
    """
    n_classes = len(np.unique(labels))
    clf = get_classifier(PRIMARY_CLASSIFIER, seed=seed,
                         dataset_name=dataset_name, n_classes=n_classes)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_metrics = []
    for train_idx, val_idx in skf.split(features, labels):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            clf_copy = clone(clf)
            clf_copy.fit(X_train, y_train)
            y_pred = clf_copy.predict(X_val)

            fold_metrics.append({
                'accuracy': float(accuracy_score(y_val, y_pred)),
                'balanced_accuracy': float(balanced_accuracy_score(y_val, y_pred)),
                'f1_macro': float(f1_score(y_val, y_pred, average='macro', zero_division=0)),
                'cohen_kappa': float(cohen_kappa_score(y_val, y_pred)),
            })
        except Exception as e:
            print(f"    Fold failed: {e}")
            fold_metrics.append({
                'accuracy': 0.0, 'balanced_accuracy': 0.0,
                'f1_macro': 0.0, 'cohen_kappa': 0.0,
            })

    # Aggregate
    result = {}
    for metric in ['accuracy', 'balanced_accuracy', 'f1_macro', 'cohen_kappa']:
        values = [fm[metric] for fm in fold_metrics]
        result[f'{metric}_mean'] = float(np.mean(values))
        result[f'{metric}_std'] = float(np.std(values))

    return result


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def load_features(dataset_name: str, n_samples: Optional[int] = None) -> tuple:
    """Load pre-extracted features from exp1 NPZ file.

    Returns:
        features_dict: {descriptor_name: (n_samples, dim) array}
        labels: (n_samples,) array
        class_names: list of class name strings
    """
    npz_path = EXP1_DIR / f"{dataset_name}_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Features not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    labels = data['labels']
    class_names = list(data['class_names']) if 'class_names' in data else []

    features_dict = {}
    for desc_name in ALL_DESCRIPTORS:
        key = f'feat_{desc_name}'
        if key in data:
            feat = data[key].astype(np.float32)
            features_dict[desc_name] = feat

    # Subsample if requested
    if n_samples is not None and n_samples < len(labels):
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(labels), n_samples, replace=False)
        labels = labels[idx]
        features_dict = {k: v[idx] for k, v in features_dict.items()}

    return features_dict, labels, class_names


def run_cka_ensemble(
    dataset_name: str,
    n_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run CKA and ensemble analysis for one dataset."""
    if output_dir is None:
        output_dir = get_output_dir('exp6')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Exp 6: CKA + Ensemble - {dataset_name}")
    print(f"{'='*70}")

    t_start = time.time()

    # --- 1. Load features ---
    print(f"\n[1/5] Loading features from {EXP1_DIR / f'{dataset_name}_features.npz'}")
    features_dict, labels, class_names = load_features(dataset_name, n_samples)
    n_samples_actual = len(labels)
    n_classes = len(np.unique(labels))
    desc_available = list(features_dict.keys())

    print(f"  Loaded {n_samples_actual} samples, {n_classes} classes")
    print(f"  Descriptors available: {len(desc_available)}/{len(ALL_DESCRIPTORS)}")
    for d in desc_available:
        print(f"    {d}: {features_dict[d].shape}")

    # --- 2. Compute CKA matrix ---
    print(f"\n[2/5] Computing CKA matrix ({len(desc_available)}x{len(desc_available)})...")
    t_cka = time.time()
    cka_matrix = compute_cka_matrix(features_dict)
    cka_time = time.time() - t_cka
    print(f"  CKA computed in {cka_time:.1f}s")

    # Find complementary pairs (low CKA)
    complementary_pairs = []
    for i in range(len(desc_available)):
        for j in range(i + 1, len(desc_available)):
            cka_val = cka_matrix[i, j]
            complementary_pairs.append({
                'pair': [desc_available[i], desc_available[j]],
                'cka': float(cka_val),
            })
    complementary_pairs.sort(key=lambda x: x['cka'])

    print(f"\n  Most complementary pairs (low CKA):")
    for p in complementary_pairs[:5]:
        print(f"    {p['pair'][0]:30s} + {p['pair'][1]:30s}: CKA={p['cka']:.4f}")
    print(f"  Least complementary (high CKA):")
    for p in complementary_pairs[-3:]:
        print(f"    {p['pair'][0]:30s} + {p['pair'][1]:30s}: CKA={p['cka']:.4f}")

    # --- 3. Individual baselines ---
    print(f"\n[3/5] Evaluating individual descriptor baselines (TabPFN, {N_FOLDS}-fold CV)...")
    baselines = {}
    for desc_name in desc_available:
        print(f"  {desc_name} ({features_dict[desc_name].shape[1]}D)...", end=' ', flush=True)
        t0 = time.time()
        result = evaluate_features(
            features_dict[desc_name], labels, dataset_name
        )
        elapsed = time.time() - t0
        baselines[desc_name] = result
        print(f"bal_acc={result['balanced_accuracy_mean']:.4f} "
              f"(+/-{result['balanced_accuracy_std']:.4f}) [{elapsed:.1f}s]")

    # Rank baselines
    baseline_ranking = sorted(
        baselines.items(),
        key=lambda x: x[1]['balanced_accuracy_mean'],
        reverse=True
    )
    print(f"\n  Top-5 individual descriptors:")
    for rank, (name, res) in enumerate(baseline_ranking[:5], 1):
        print(f"    #{rank} {name}: bal_acc={res['balanced_accuracy_mean']:.4f}")

    # --- 4. Ensemble evaluation ---
    print(f"\n[4/5] Evaluating {len(ENSEMBLE_CONFIGS)} ensemble configurations...")
    ensemble_results = {}

    for config in ENSEMBLE_CONFIGS:
        config_name = '+'.join(config)

        # Check all descriptors in config are available
        missing = [d for d in config if d not in features_dict]
        if missing:
            print(f"  SKIP {config_name} (missing: {missing})")
            ensemble_results[config_name] = {
                'error': f'Missing descriptors: {missing}',
                'descriptors': config,
            }
            continue

        # Concatenate features
        concat_features = np.hstack([features_dict[d] for d in config])
        total_dim = concat_features.shape[1]

        print(f"  {config_name} ({total_dim}D)...", end=' ', flush=True)
        t0 = time.time()

        try:
            result = evaluate_features(
                concat_features, labels, dataset_name
            )

            # Compute synergy vs best individual in this ensemble
            best_single_acc = max(
                baselines[d]['balanced_accuracy_mean'] for d in config
            )
            synergy = result['balanced_accuracy_mean'] - best_single_acc

            ensemble_results[config_name] = {
                'descriptors': config,
                'dimension': total_dim,
                **result,
                'best_single_in_ensemble': float(best_single_acc),
                'synergy': float(synergy),
            }

            elapsed = time.time() - t0
            print(f"bal_acc={result['balanced_accuracy_mean']:.4f} "
                  f"synergy={synergy:+.4f} [{elapsed:.1f}s]")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED: {e} [{elapsed:.1f}s]")
            ensemble_results[config_name] = {
                'descriptors': config,
                'dimension': total_dim,
                'error': str(e),
            }

    # --- 5. Summary ---
    print(f"\n[5/5] Generating summary...")

    # Best ensemble
    valid_ensembles = {
        k: v for k, v in ensemble_results.items()
        if 'balanced_accuracy_mean' in v
    }
    if valid_ensembles:
        best_ensemble_name = max(
            valid_ensembles, key=lambda k: valid_ensembles[k]['balanced_accuracy_mean']
        )
        best_ensemble_result = valid_ensembles[best_ensemble_name]
    else:
        best_ensemble_name = None
        best_ensemble_result = {}

    # Best synergy
    synergy_values = {
        k: v.get('synergy', -999) for k, v in ensemble_results.items()
        if 'synergy' in v
    }
    best_synergy_name = max(synergy_values, key=synergy_values.get) if synergy_values else None

    total_time = time.time() - t_start

    # Build output
    output = {
        'metadata': {
            'dataset': dataset_name,
            'n_samples': n_samples_actual,
            'n_classes': n_classes,
            'class_names': class_names,
            'n_descriptors': len(desc_available),
            'descriptor_names': desc_available,
            'classifier': PRIMARY_CLASSIFIER,
            'n_folds': N_FOLDS,
            'seed': RANDOM_SEED,
            'cka_time': cka_time,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
        },
        'cka_matrix': cka_matrix.tolist(),
        'complementary_pairs': complementary_pairs[:15],
        'baselines': baselines,
        'baseline_ranking': [
            {'descriptor': name, 'balanced_accuracy_mean': res['balanced_accuracy_mean']}
            for name, res in baseline_ranking
        ],
        'ensemble_results': ensemble_results,
        'best_ensemble': {
            'name': best_ensemble_name,
            'result': best_ensemble_result,
        },
        'best_synergy': {
            'name': best_synergy_name,
            'synergy': synergy_values.get(best_synergy_name, 0) if best_synergy_name else 0,
        },
    }

    # Save outputs
    # CKA matrix as numpy
    cka_path = output_dir / f"{dataset_name}_cka_matrix.npy"
    np.save(cka_path, cka_matrix)
    print(f"  Saved CKA matrix: {cka_path}")

    # Full results as JSON
    results_path = output_dir / f"{dataset_name}_ensemble.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"  Saved results: {results_path}")

    # Print summary
    print(f"\n{'─'*70}")
    print(f"  Dataset: {dataset_name} ({n_samples_actual} samples, {n_classes} classes)")
    print(f"  Best individual: {baseline_ranking[0][0]} "
          f"(bal_acc={baseline_ranking[0][1]['balanced_accuracy_mean']:.4f})")
    if best_ensemble_name:
        print(f"  Best ensemble:   {best_ensemble_name} "
              f"(bal_acc={best_ensemble_result['balanced_accuracy_mean']:.4f})")
    if best_synergy_name:
        print(f"  Best synergy:    {best_synergy_name} "
              f"(synergy={synergy_values[best_synergy_name]:+.4f})")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'─'*70}\n")

    return output


def generate_summary(all_results: Dict[str, Dict], output_dir: Path):
    """Generate cross-dataset summary after all datasets are processed."""
    print(f"\n{'='*70}")
    print(f"Generating cross-dataset summary ({len(all_results)} datasets)")
    print(f"{'='*70}")

    # Average CKA matrix across datasets
    cka_matrices = []
    for dataset_name, result in all_results.items():
        if 'cka_matrix' in result:
            cka_matrices.append(np.array(result['cka_matrix']))

    if cka_matrices:
        avg_cka_matrix = np.mean(cka_matrices, axis=0)
        np.save(output_dir / 'avg_cka_matrix.npy', avg_cka_matrix)
    else:
        avg_cka_matrix = None

    # Descriptor names (from first result)
    first_result = next(iter(all_results.values()))
    desc_names = first_result['metadata']['descriptor_names']

    # Per-descriptor average balanced accuracy across datasets
    desc_avg_acc = {}
    for desc_name in desc_names:
        accs = []
        for dataset_name, result in all_results.items():
            if desc_name in result.get('baselines', {}):
                accs.append(result['baselines'][desc_name]['balanced_accuracy_mean'])
        if accs:
            desc_avg_acc[desc_name] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'n_datasets': len(accs),
            }

    desc_ranking = sorted(desc_avg_acc.items(), key=lambda x: x[1]['mean'], reverse=True)

    # Per-ensemble average across datasets
    ensemble_avg = {}
    for config in ENSEMBLE_CONFIGS:
        config_name = '+'.join(config)
        accs = []
        synergies = []
        for dataset_name, result in all_results.items():
            ens = result.get('ensemble_results', {}).get(config_name, {})
            if 'balanced_accuracy_mean' in ens:
                accs.append(ens['balanced_accuracy_mean'])
            if 'synergy' in ens:
                synergies.append(ens['synergy'])

        if accs:
            ensemble_avg[config_name] = {
                'descriptors': config,
                'dimension': sum(DESCRIPTORS[d]['dim'] for d in config if d in DESCRIPTORS),
                'balanced_accuracy_mean': float(np.mean(accs)),
                'balanced_accuracy_std': float(np.std(accs)),
                'synergy_mean': float(np.mean(synergies)) if synergies else 0.0,
                'synergy_std': float(np.std(synergies)) if synergies else 0.0,
                'n_datasets': len(accs),
                'wins': 0,  # filled below
            }

    # Count wins per ensemble (best on each dataset)
    for dataset_name, result in all_results.items():
        ens_results = result.get('ensemble_results', {})
        best_acc = -1
        best_name = None
        for config_name, ens in ens_results.items():
            if 'balanced_accuracy_mean' in ens and ens['balanced_accuracy_mean'] > best_acc:
                best_acc = ens['balanced_accuracy_mean']
                best_name = config_name
        if best_name and best_name in ensemble_avg:
            ensemble_avg[best_name]['wins'] += 1

    ensemble_ranking = sorted(
        ensemble_avg.items(),
        key=lambda x: x[1]['balanced_accuracy_mean'],
        reverse=True
    )

    # Average complementary pairs across datasets
    avg_complementary = {}
    for dataset_name, result in all_results.items():
        for pair_info in result.get('complementary_pairs', []):
            pair_key = tuple(sorted(pair_info['pair']))
            if pair_key not in avg_complementary:
                avg_complementary[pair_key] = []
            avg_complementary[pair_key].append(pair_info['cka'])

    avg_complementary_sorted = sorted(
        [
            {'pair': list(k), 'cka_mean': float(np.mean(v)), 'cka_std': float(np.std(v))}
            for k, v in avg_complementary.items()
        ],
        key=lambda x: x['cka_mean']
    )

    # Build summary
    summary = {
        'metadata': {
            'n_datasets': len(all_results),
            'datasets': list(all_results.keys()),
            'n_descriptors': len(desc_names),
            'descriptor_names': desc_names,
            'n_ensembles': len(ENSEMBLE_CONFIGS),
            'classifier': PRIMARY_CLASSIFIER,
            'timestamp': datetime.now().isoformat(),
        },
        'avg_cka_matrix': avg_cka_matrix.tolist() if avg_cka_matrix is not None else None,
        'descriptor_ranking': [
            {'descriptor': name, **stats} for name, stats in desc_ranking
        ],
        'ensemble_ranking': [
            {'ensemble': name, **stats} for name, stats in ensemble_ranking
        ],
        'most_complementary_pairs': avg_complementary_sorted[:15],
        'least_complementary_pairs': avg_complementary_sorted[-5:],
        'per_dataset_best_ensemble': {
            dataset_name: result.get('best_ensemble', {}).get('name', 'N/A')
            for dataset_name, result in all_results.items()
        },
        'per_dataset_best_individual': {
            dataset_name: result.get('baseline_ranking', [{}])[0].get('descriptor', 'N/A')
            for dataset_name, result in all_results.items()
        },
    }

    # Save summary JSON
    summary_path = output_dir / 'benchmark3_exp6_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"  Saved summary: {summary_path}")

    # Save flat CSV
    csv_path = output_dir / 'benchmark3_exp6_results.csv'
    _save_csv(all_results, csv_path)
    print(f"  Saved CSV: {csv_path}")

    # Print highlights
    print(f"\n{'─'*70}")
    print(f"  DESCRIPTOR RANKING (avg balanced_accuracy across {len(all_results)} datasets):")
    for rank, (name, stats) in enumerate(desc_ranking[:5], 1):
        print(f"    #{rank} {name}: {stats['mean']:.4f} (+/-{stats['std']:.4f})")

    print(f"\n  ENSEMBLE RANKING:")
    for rank, (name, stats) in enumerate(ensemble_ranking[:5], 1):
        print(f"    #{rank} {name}")
        print(f"        bal_acc={stats['balanced_accuracy_mean']:.4f} "
              f"synergy={stats['synergy_mean']:+.4f} "
              f"dim={stats['dimension']}D wins={stats['wins']}")

    print(f"\n  MOST COMPLEMENTARY PAIRS (avg CKA):")
    for p in avg_complementary_sorted[:5]:
        print(f"    {p['pair'][0]:30s} + {p['pair'][1]:30s}: "
              f"CKA={p['cka_mean']:.4f} (+/-{p['cka_std']:.4f})")
    print(f"{'─'*70}\n")

    return summary


def _save_csv(all_results: Dict[str, Dict], csv_path: Path):
    """Save flat CSV with one row per (dataset, ensemble_config)."""
    rows = []

    for dataset_name, result in all_results.items():
        # Individual baselines
        for desc_name, baseline in result.get('baselines', {}).items():
            rows.append({
                'dataset': dataset_name,
                'method': desc_name,
                'type': 'individual',
                'dimension': DESCRIPTORS.get(desc_name, {}).get('dim', 0),
                'balanced_accuracy_mean': baseline.get('balanced_accuracy_mean', 0),
                'balanced_accuracy_std': baseline.get('balanced_accuracy_std', 0),
                'accuracy_mean': baseline.get('accuracy_mean', 0),
                'accuracy_std': baseline.get('accuracy_std', 0),
                'f1_macro_mean': baseline.get('f1_macro_mean', 0),
                'f1_macro_std': baseline.get('f1_macro_std', 0),
                'synergy': 0.0,
            })

        # Ensembles
        for config_name, ens in result.get('ensemble_results', {}).items():
            if 'error' in ens and 'balanced_accuracy_mean' not in ens:
                continue
            rows.append({
                'dataset': dataset_name,
                'method': config_name,
                'type': 'ensemble',
                'dimension': ens.get('dimension', 0),
                'balanced_accuracy_mean': ens.get('balanced_accuracy_mean', 0),
                'balanced_accuracy_std': ens.get('balanced_accuracy_std', 0),
                'accuracy_mean': ens.get('accuracy_mean', 0),
                'accuracy_std': ens.get('accuracy_std', 0),
                'f1_macro_mean': ens.get('f1_macro_mean', 0),
                'f1_macro_std': ens.get('f1_macro_std', 0),
                'synergy': ens.get('synergy', 0.0),
            })

    # Write CSV
    if rows:
        headers = list(rows[0].keys())
        with open(csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
            for row in rows:
                f.write(','.join(str(row[h]) for h in headers) + '\n')


def _json_default(obj):
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark3 Exp 6: CKA + Ensemble Analysis"
    )
    parser.add_argument('--dataset', type=str, default='BloodMNIST',
                        help='Dataset name (default: BloodMNIST)')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run on all 13 datasets')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Subsample to N samples (default: use all)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (200 samples)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results/benchmark3/exp6)')

    args = parser.parse_args()

    if args.test:
        args.n_samples = 200

    output_dir = Path(args.output) if args.output else get_output_dir('exp6')
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
    print(f"Datasets to process: {datasets}")
    print(f"Output directory: {output_dir}")
    if args.n_samples:
        print(f"Subsampling to {args.n_samples} samples")

    all_results = {}
    for dataset_name in datasets:
        try:
            result = run_cka_ensemble(dataset_name, args.n_samples, output_dir)
            all_results[dataset_name] = result
        except Exception as e:
            import traceback
            print(f"\nERROR: {dataset_name} failed: {e}")
            traceback.print_exc()

    # Generate cross-dataset summary if multiple datasets
    if len(all_results) > 1:
        generate_summary(all_results, output_dir)

    print(f"\nDone. Processed {len(all_results)}/{len(datasets)} datasets.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
