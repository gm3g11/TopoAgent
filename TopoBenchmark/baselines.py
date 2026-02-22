"""Baseline Implementations for TopoBenchmark.

8 baselines:
1. Random: uniform random descriptor
2. Fixed-ATOL: always ATOL
3. Fixed-persistence_statistics: always persistence_statistics
4. Fixed-persistence_image: always persistence_image
5. Fixed-best-per-type: best descriptor per object_type (from exp4 TOP_PERFORMERS)
6. Exp4-rules: full rule-based selection via get_top_descriptors()
7. TopoAgent-no-skills: agent without skill knowledge (placeholder — requires LLM)
8. TopoAgent-with-skills: full agent (placeholder — requires LLM)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ground_truth import GroundTruth, load_ground_truth, ALL_DESCRIPTORS
from .config import DATASET_DESCRIPTIONS
from .metrics import evaluate_selections


# Object type for each dataset (from benchmark4 config)
DATASET_OBJECT_TYPES = {
    ds: desc["object_type"]
    for ds, desc in DATASET_DESCRIPTIONS.items()
}


def _get_available_descriptors(
    dataset: str, ground_truth: GroundTruth
) -> List[str]:
    """Get descriptors that have results for this dataset."""
    return [
        desc for desc in ALL_DESCRIPTORS
        if ground_truth.get_accuracy(dataset, desc) is not None
    ]


def random_baseline(
    ground_truth: GroundTruth,
    n_seeds: int = 10,
) -> Dict[str, any]:
    """Random descriptor selection baseline.

    Uniformly selects from descriptors with available results per dataset.

    Returns:
        Dict with mean/std MBA across seeds, plus individual seed results.
    """
    seed_results = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        selections = {}
        for dataset in ground_truth.datasets_covered:
            available = _get_available_descriptors(dataset, ground_truth)
            if available:
                selections[dataset] = rng.choice(available)
        seed_results.append(selections)

    # Compute MBA for each seed
    mbas = []
    for selections in seed_results:
        mba, _ = ground_truth.compute_mba_for_selections(selections)
        mbas.append(mba)

    # Use mean selections for detailed metrics
    # Pick the median seed for representative analysis
    median_idx = int(np.argsort(mbas)[len(mbas) // 2])

    result = evaluate_selections(
        seed_results[median_idx],
        ground_truth,
        DATASET_OBJECT_TYPES,
        label="Random",
    )
    result["mba_seeds"] = mbas
    result["mba_mean"] = float(np.mean(mbas))
    result["mba_std"] = float(np.std(mbas))
    result["n_seeds"] = n_seeds

    return result


def fixed_descriptor_baseline(
    ground_truth: GroundTruth,
    descriptor: str,
    label: Optional[str] = None,
) -> Dict[str, any]:
    """Fixed single descriptor baseline.

    For datasets where the descriptor has no results, uses the next-best
    available descriptor (ranked by overall MBA).

    Returns:
        Dict with all metrics.
    """
    if label is None:
        label = f"Fixed-{descriptor}"

    selections = {}
    for dataset in ground_truth.datasets_covered:
        acc = ground_truth.get_accuracy(dataset, descriptor)
        if acc is not None:
            selections[dataset] = descriptor
        else:
            # Fallback: pick the descriptor with highest overall MBA among available ones
            available = _get_available_descriptors(dataset, ground_truth)
            if available:
                best_fallback = max(
                    available,
                    key=lambda d: ground_truth.get_accuracy(dataset, d) or 0.0,
                )
                selections[dataset] = best_fallback

    return evaluate_selections(
        selections, ground_truth, DATASET_OBJECT_TYPES, label=label,
    )


def fixed_best_per_type_baseline(
    ground_truth: GroundTruth,
) -> Dict[str, any]:
    """Best descriptor per object_type from exp4 TOP_PERFORMERS.

    Uses the #1 ranked descriptor for each object type.
    Falls back to available descriptors if the top performer has no results.
    """
    try:
        from topoagent.skills.rules_data import TOP_PERFORMERS
    except ImportError:
        # Hardcode fallback if import fails
        TOP_PERFORMERS = {
            "discrete_cells": [{"descriptor": "template_functions"}],
            "glands_lumens": [{"descriptor": "lbp_texture"}],
            "vessel_trees": [{"descriptor": "persistence_statistics"}],
            "surface_lesions": [{"descriptor": "persistence_statistics"}],
            "organ_shape": [{"descriptor": "minkowski_functionals"}],
        }

    # Map object_type -> best descriptor
    type_to_desc = {}
    for obj_type, performers in TOP_PERFORMERS.items():
        if performers:
            type_to_desc[obj_type] = performers[0]["descriptor"]

    selections = {}
    for dataset in ground_truth.datasets_covered:
        obj_type = DATASET_OBJECT_TYPES.get(dataset)
        if obj_type and obj_type in type_to_desc:
            descriptor = type_to_desc[obj_type]
            acc = ground_truth.get_accuracy(dataset, descriptor)
            if acc is not None:
                selections[dataset] = descriptor
            else:
                # Fallback: try other top performers for this type
                for entry in TOP_PERFORMERS.get(obj_type, []):
                    d = entry["descriptor"]
                    if ground_truth.get_accuracy(dataset, d) is not None:
                        selections[dataset] = d
                        break
                else:
                    # Ultimate fallback
                    available = _get_available_descriptors(dataset, ground_truth)
                    if available:
                        selections[dataset] = available[0]

    return evaluate_selections(
        selections, ground_truth, DATASET_OBJECT_TYPES,
        label="Fixed-Best-Per-Type",
    )


def exp4_rules_baseline(
    ground_truth: GroundTruth,
) -> Dict[str, any]:
    """Full exp4 rule-based selection.

    Uses get_top_descriptors(object_type)[0] for each dataset.
    Falls back through the ranking if top descriptor has no results.
    """
    try:
        from topoagent.skills.rules_data import get_top_descriptors
    except ImportError:
        # If import fails, fall back to fixed_best_per_type
        return fixed_best_per_type_baseline(ground_truth)

    selections = {}
    for dataset in ground_truth.datasets_covered:
        obj_type = DATASET_OBJECT_TYPES.get(dataset)
        if not obj_type:
            continue

        top = get_top_descriptors(obj_type, n=5)
        for entry in top:
            descriptor = entry["descriptor"]
            if ground_truth.get_accuracy(dataset, descriptor) is not None:
                selections[dataset] = descriptor
                break
        else:
            # Fallback: use any available descriptor
            available = _get_available_descriptors(dataset, ground_truth)
            if available:
                selections[dataset] = available[0]

    return evaluate_selections(
        selections, ground_truth, DATASET_OBJECT_TYPES,
        label="Exp4-Rules",
    )


def run_all_baselines(
    ground_truth: Optional[GroundTruth] = None,
) -> Dict[str, Dict]:
    """Run all non-agent baselines.

    Returns:
        Dict of {baseline_name: metrics_dict}
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    results = {}

    # 1. Random
    results["random"] = random_baseline(ground_truth, n_seeds=10)

    # 2. Fixed-ATOL
    results["fixed_ATOL"] = fixed_descriptor_baseline(
        ground_truth, "ATOL", label="Fixed-ATOL",
    )

    # 3. Fixed-persistence_statistics
    results["fixed_persistence_statistics"] = fixed_descriptor_baseline(
        ground_truth, "persistence_statistics",
        label="Fixed-persistence_statistics",
    )

    # 4. Fixed-persistence_image
    results["fixed_persistence_image"] = fixed_descriptor_baseline(
        ground_truth, "persistence_image",
        label="Fixed-persistence_image",
    )

    # 5. Fixed-best-per-type
    results["fixed_best_per_type"] = fixed_best_per_type_baseline(ground_truth)

    # 5. Exp4-rules
    results["exp4_rules"] = exp4_rules_baseline(ground_truth)

    return results


def print_baseline_table(results: Dict[str, Dict]) -> str:
    """Print a formatted comparison table."""
    lines = []
    header = f"{'Baseline':<30s} {'MBA':>8s} {'DSA':>8s} {'Top3-DSA':>10s} {'Regret':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, r in results.items():
        mba = r.get("mba", r.get("mba_mean", 0.0))
        dsa = r.get("dsa", 0.0)
        dsa3 = r.get("dsa_top3", 0.0)
        regret = r.get("regret", 0.0)
        label = r.get("label", name)
        lines.append(
            f"{label:<30s} {mba:>8.4f} {dsa:>7.1%} {dsa3:>9.1%} {regret:>8.4f}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    gt = load_ground_truth()
    print(f"Ground truth: {gt.n_results} results, {gt.n_datasets} datasets")
    print(f"Oracle MBA: {gt.mba:.4f}")
    print()

    results = run_all_baselines(gt)
    print(print_baseline_table(results))

    # Print oracle for comparison
    print(f"\n{'Oracle':<30s} {gt.mba:>8.4f} {'100.0%':>8s} {'100.0%':>10s} {'0.0000':>8s}")
