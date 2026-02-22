"""Evaluation Metrics for TopoBenchmark.

Primary: MBA (Mean Balanced Accuracy), DSA (Descriptor Selection Accuracy)
Secondary: Regret, Per-Object-Type breakdown
Statistical: Wilcoxon signed-rank, Bootstrap CI, McNemar's test
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compute_mba(
    selections: Dict[str, str],
    ground_truth,
) -> Tuple[float, Dict[str, float]]:
    """Compute Mean Balanced Accuracy for descriptor selections.

    Args:
        selections: {dataset: descriptor} mapping
        ground_truth: GroundTruth object

    Returns:
        (mba, per_dataset_accuracies)
    """
    return ground_truth.compute_mba_for_selections(selections)


def compute_dsa(
    selections: Dict[str, str],
    ground_truth,
    top_n: int = 1,
) -> Tuple[float, Dict[str, bool]]:
    """Compute Descriptor Selection Accuracy.

    Args:
        selections: {dataset: descriptor} mapping
        ground_truth: GroundTruth object
        top_n: Consider correct if within top-n descriptors (1 = exact match)

    Returns:
        (dsa_score, per_dataset_correct)
    """
    correct = {}
    for dataset, descriptor in selections.items():
        top_descriptors = ground_truth.get_top_n_descriptors(dataset, n=top_n)
        correct[dataset] = descriptor in top_descriptors

    if not correct:
        return 0.0, {}

    dsa = np.mean(list(correct.values()))
    return float(dsa), correct


def compute_regret(
    selections: Dict[str, str],
    ground_truth,
) -> Tuple[float, Dict[str, float]]:
    """Compute regret: oracle_acc - agent_acc per dataset.

    Args:
        selections: {dataset: descriptor}
        ground_truth: GroundTruth object

    Returns:
        (mean_regret, per_dataset_regret)
    """
    regrets = {}
    for dataset, descriptor in selections.items():
        oracle_acc = ground_truth.get_oracle_accuracy(dataset)
        agent_acc = ground_truth.get_accuracy(dataset, descriptor)
        if oracle_acc is not None and agent_acc is not None:
            regrets[dataset] = oracle_acc - agent_acc

    if not regrets:
        return 0.0, {}

    mean_regret = float(np.mean(list(regrets.values())))
    return mean_regret, regrets


def compute_per_object_type_mba(
    selections: Dict[str, str],
    ground_truth,
    dataset_object_types: Dict[str, str],
) -> Dict[str, float]:
    """Compute MBA broken down by object type.

    Args:
        selections: {dataset: descriptor}
        ground_truth: GroundTruth object
        dataset_object_types: {dataset: object_type}

    Returns:
        {object_type: mba}
    """
    type_accs = {}
    for dataset, descriptor in selections.items():
        acc = ground_truth.get_accuracy(dataset, descriptor)
        obj_type = dataset_object_types.get(dataset, "unknown")
        if acc is not None:
            type_accs.setdefault(obj_type, []).append(acc)

    return {
        obj_type: float(np.mean(accs))
        for obj_type, accs in type_accs.items()
    }


def compute_selection_distribution(
    selections: Dict[str, str],
) -> Dict[str, int]:
    """Compute histogram of descriptor selections.

    Args:
        selections: {dataset: descriptor}

    Returns:
        {descriptor: count}
    """
    dist = {}
    for descriptor in selections.values():
        dist[descriptor] = dist.get(descriptor, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def wilcoxon_test(
    agent_accs: Dict[str, float],
    baseline_accs: Dict[str, float],
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired comparison.

    Args:
        agent_accs: {dataset: accuracy} for agent
        baseline_accs: {dataset: accuracy} for baseline

    Returns:
        (statistic, p_value)
    """
    # Get common datasets
    common = sorted(set(agent_accs.keys()) & set(baseline_accs.keys()))
    if len(common) < 5:
        return float("nan"), float("nan")

    a = np.array([agent_accs[d] for d in common])
    b = np.array([baseline_accs[d] for d in common])

    # If all differences are zero, return nan
    diffs = a - b
    if np.all(diffs == 0):
        return 0.0, 1.0

    try:
        stat, p = stats.wilcoxon(a, b, alternative="greater")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval.

    Args:
        values: List of values to bootstrap
        n_boot: Number of bootstrap resamples
        ci: Confidence level

    Returns:
        (mean, lower, upper)
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    mean = float(np.mean(values))

    return mean, lower, upper


def mcnemar_test(
    agent_correct: Dict[str, bool],
    baseline_correct: Dict[str, bool],
) -> Tuple[float, float]:
    """McNemar's test for binary DSA comparisons.

    Args:
        agent_correct: {dataset: True/False} for agent
        baseline_correct: {dataset: True/False} for baseline

    Returns:
        (statistic, p_value)
    """
    common = sorted(set(agent_correct.keys()) & set(baseline_correct.keys()))
    if not common:
        return float("nan"), float("nan")

    # Build contingency table
    # b: agent correct, baseline wrong
    # c: agent wrong, baseline correct
    b = sum(1 for d in common if agent_correct[d] and not baseline_correct[d])
    c = sum(1 for d in common if not agent_correct[d] and baseline_correct[d])

    if b + c == 0:
        return 0.0, 1.0

    # McNemar with continuity correction
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(stat, df=1)

    return float(stat), float(p)


def evaluate_selections(
    selections: Dict[str, str],
    ground_truth,
    dataset_object_types: Dict[str, str],
    label: str = "Agent",
) -> Dict[str, Any]:
    """Run all metrics on a set of selections.

    Args:
        selections: {dataset: descriptor}
        ground_truth: GroundTruth object
        dataset_object_types: {dataset: object_type}
        label: Name for this method

    Returns:
        Dict with all computed metrics
    """
    mba, per_ds_acc = compute_mba(selections, ground_truth)
    dsa, per_ds_correct = compute_dsa(selections, ground_truth, top_n=1)
    dsa3, per_ds_correct3 = compute_dsa(selections, ground_truth, top_n=3)
    regret, per_ds_regret = compute_regret(selections, ground_truth)
    ot_mba = compute_per_object_type_mba(selections, ground_truth, dataset_object_types)
    dist = compute_selection_distribution(selections)

    accs = list(per_ds_acc.values())
    mba_mean, mba_lower, mba_upper = bootstrap_ci(accs)

    return {
        "label": label,
        "mba": mba,
        "mba_ci_lower": mba_lower,
        "mba_ci_upper": mba_upper,
        "dsa": dsa,
        "dsa_top3": dsa3,
        "regret": regret,
        "per_object_type_mba": ot_mba,
        "selection_distribution": dist,
        "per_dataset_accuracy": per_ds_acc,
        "per_dataset_correct": per_ds_correct,
        "per_dataset_regret": per_ds_regret,
        "n_datasets": len(selections),
    }
