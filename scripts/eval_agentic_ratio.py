#!/usr/bin/env python
"""Evaluate the agentic ratio of TopoAgent v9 pipeline.

Runs demo_topoagent_e2e on multiple datasets (1 sample each, --no-eval)
and parses the JSON case study output to measure:
  - Stance distribution (AGREE / FOLLOW_HYPOTHESIS / FOLLOW_BENCHMARK)
  - Object type reconciliation accuracy
  - Descriptor diversity (unique descriptors chosen)
  - Fit score of chosen descriptor vs benchmark top pick
  - Whether chosen descriptor differs from simple lookup table

Usage:
    python scripts/eval_agentic_ratio.py [--datasets D1 D2 ...] [--model MODEL]
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

# All 26 datasets (or a subset)
ALL_DATASETS = [
    "BloodMNIST", "TissueMNIST", "PathMNIST", "OCTMNIST", "OrganAMNIST",
    "RetinaMNIST", "PneumoniaMNIST", "BreastMNIST", "DermaMNIST",
    "OrganCMNIST", "OrganSMNIST",
    "ISIC2019", "Kvasir", "BrainTumorMRI", "MURA", "BreakHis",
    "NCT_CRC_HE", "MalariaCell", "IDRiD", "PCam", "LC25000",
    "SIPaKMeD", "AML_Cytomorphology", "APTOS2019", "GasHisSDB", "Chaoyang",
]

# Representative subset (1 per object type + extras)
REPRESENTATIVE = [
    "BloodMNIST",    # discrete_cells
    "PathMNIST",     # glands_lumens
    "RetinaMNIST",   # vessel_trees
    "DermaMNIST",    # surface_lesions
    "OrganAMNIST",   # organ_shape
    # Additional diversity
    "PCam",          # discrete_cells (different from BloodMNIST)
    "ISIC2019",      # surface_lesions (different from DermaMNIST)
    "Kvasir",        # glands_lumens (different from PathMNIST)
    "MURA",          # organ_shape (different from OrganAMNIST)
    "APTOS2019",     # vessel_trees (different from RetinaMNIST)
]

DATASET_TO_OBJECT_TYPE = {
    "BloodMNIST": "discrete_cells", "TissueMNIST": "discrete_cells",
    "PathMNIST": "glands_lumens", "OCTMNIST": "organ_shape",
    "OrganAMNIST": "organ_shape", "RetinaMNIST": "vessel_trees",
    "PneumoniaMNIST": "organ_shape", "BreastMNIST": "organ_shape",
    "DermaMNIST": "surface_lesions", "OrganCMNIST": "organ_shape",
    "OrganSMNIST": "organ_shape",
    "ISIC2019": "surface_lesions", "Kvasir": "glands_lumens",
    "BrainTumorMRI": "organ_shape", "MURA": "organ_shape",
    "BreakHis": "glands_lumens", "NCT_CRC_HE": "glands_lumens",
    "MalariaCell": "discrete_cells", "IDRiD": "vessel_trees",
    "PCam": "discrete_cells", "LC25000": "glands_lumens",
    "SIPaKMeD": "discrete_cells", "AML_Cytomorphology": "discrete_cells",
    "APTOS2019": "vessel_trees", "GasHisSDB": "surface_lesions",
    "Chaoyang": "glands_lumens",
}

# Benchmark top pick per object type (from SUPPORTED_TOP_PERFORMERS[0])
BENCHMARK_TOP = {
    "discrete_cells": "minkowski_functionals",
    "glands_lumens": "persistence_statistics",
    "vessel_trees": "lbp_texture",
    "surface_lesions": "persistence_statistics",
    "organ_shape": "minkowski_functionals",
}


def run_one_dataset(dataset: str, model: str = "gpt-4o") -> dict:
    """Run demo_topoagent_e2e on one dataset and parse the case study JSON."""
    case_study_path = f"results/case_studies/{dataset}_explicit_case_study.json"

    cmd = [
        "conda", "run", "-n", "medrax", "python",
        "scripts/demo_topoagent_e2e.py",
        "--dataset", dataset,
        "--v9", "--model", model, "--no-eval",
    ]

    print(f"  Running {dataset}...", end="", flush=True)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).parent.parent),
        )
        if result.returncode != 0:
            print(f" FAILED (exit {result.returncode})")
            # Check stderr for clues
            err_lines = result.stderr.strip().split("\n")[-3:]
            for line in err_lines:
                print(f"    {line}")
            return {"dataset": dataset, "error": "subprocess failed"}
    except subprocess.TimeoutExpired:
        print(" TIMEOUT")
        return {"dataset": dataset, "error": "timeout"}

    # Parse case study JSON
    if not os.path.exists(case_study_path):
        print(f" NO JSON")
        return {"dataset": dataset, "error": "no json output"}

    with open(case_study_path) as f:
        cs = json.load(f)

    gt_ot = DATASET_TO_OBJECT_TYPE.get(dataset, "?")
    decisions = cs.get("decisions", {})

    # INTERPRET output
    v9_interpret = decisions.get("v9_interpret", {})
    interpret_ot = v9_interpret.get("object_type_guess", decisions.get("object_type", "?"))

    # ANALYZE (hypothesis) output
    v9_hypothesis = decisions.get("v9_hypothesis", {})
    reconciled_ot = v9_hypothesis.get("object_type_reconciled", interpret_ot)
    hypothesis_desc = v9_hypothesis.get("descriptor_hypothesis", "?")
    confidence = v9_hypothesis.get("confidence", "?")
    fit_cited = v9_hypothesis.get("fit_score_cited", "?")

    # ACT decision
    v9_act = decisions.get("v9_act_decision", {})
    final_desc = v9_act.get("final_descriptor", decisions.get("descriptor", "?"))
    act_reasoning = v9_act.get("reasoning", "")
    # Fall back to old schema if present
    if not act_reasoning:
        reconciliation = v9_act.get("reconciliation", {})
        act_reasoning = reconciliation.get("stance_reasoning", "")
    benchmark_top = BENCHMARK_TOP.get(gt_ot, "?")

    # Infer behavior post-hoc (for analysis — these are observations, not LLM labels)
    if final_desc == hypothesis_desc:
        behavior = "confirmed_hypothesis"
    elif final_desc == benchmark_top:
        behavior = "switched_to_benchmark"
    else:
        behavior = "chose_alternative"

    # Is the final descriptor different from the lookup table?
    differs_from_lookup = final_desc != benchmark_top

    print(f" OK | {behavior} | desc={final_desc} (hyp={hypothesis_desc}) | "
          f"ot_guess={interpret_ot}->reconciled={reconciled_ot} (GT={gt_ot})")

    return {
        "dataset": dataset,
        "gt_object_type": gt_ot,
        "interpret_object_type": interpret_ot,
        "reconciled_object_type": reconciled_ot,
        "ot_interpret_correct": interpret_ot == gt_ot,
        "ot_reconciled_correct": reconciled_ot == gt_ot,
        "descriptor_hypothesis": hypothesis_desc,
        "final_descriptor": final_desc,
        "benchmark_top": benchmark_top,
        "differs_from_lookup": differs_from_lookup,
        "behavior": behavior,
        "reasoning": act_reasoning,
        "confidence": confidence,
    }


def compute_agentic_ratio(results: list) -> dict:
    """Compute metrics from collected results.

    Behaviors are inferred post-hoc, not LLM labels:
    - confirmed_hypothesis: final == hypothesis (agent kept its own choice)
    - switched_to_benchmark: final == benchmark top (agent deferred to evidence)
    - chose_alternative: final != hypothesis AND != benchmark (agent found a third option)
    """
    valid = [r for r in results if "error" not in r]
    n = len(valid)
    if n == 0:
        return {"error": "no valid results"}

    behaviors = Counter(r["behavior"] for r in valid)
    n_confirmed = behaviors.get("confirmed_hypothesis", 0)
    n_switched = behaviors.get("switched_to_benchmark", 0)
    n_alternative = behaviors.get("chose_alternative", 0)

    # Descriptor diversity
    unique_descs = set(r["final_descriptor"] for r in valid)

    # Object type reconciliation
    ot_interpret_correct = sum(1 for r in valid if r["ot_interpret_correct"])
    ot_reconciled_correct = sum(1 for r in valid if r["ot_reconciled_correct"])
    ot_corrections = sum(1 for r in valid
                        if r["interpret_object_type"] != r["reconciled_object_type"])

    return {
        "n_datasets": n,
        "behavior_distribution": dict(behaviors),
        "n_confirmed_hypothesis": n_confirmed,
        "n_switched_to_benchmark": n_switched,
        "n_chose_alternative": n_alternative,
        "n_differs_from_lookup": sum(1 for r in valid if r["differs_from_lookup"]),
        "unique_descriptors": sorted(unique_descs),
        "descriptor_diversity": len(unique_descs),
        "ot_interpret_correct": f"{ot_interpret_correct}/{n}",
        "ot_reconciled_correct": f"{ot_reconciled_correct}/{n}",
        "ot_corrections_made": ot_corrections,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TopoAgent v9 agentic ratio")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to evaluate (default: representative subset)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 26 datasets")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--output", default="results/case_studies/agentic_ratio_eval.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if args.all:
        datasets = ALL_DATASETS
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = REPRESENTATIVE

    print(f"Evaluating agentic ratio on {len(datasets)} datasets with {args.model}")
    print("=" * 70)

    results = []
    for ds in datasets:
        r = run_one_dataset(ds, args.model)
        results.append(r)

    print()
    print("=" * 70)
    print("AGENTIC RATIO EVALUATION")
    print("=" * 70)

    metrics = compute_agentic_ratio(results)

    print(f"\nDatasets evaluated: {metrics['n_datasets']}")
    print(f"\nAgent behavior (observed post-hoc):")
    for behavior, count in sorted(metrics["behavior_distribution"].items()):
        pct = count / metrics["n_datasets"] * 100
        print(f"  {behavior}: {count} ({pct:.0f}%)")

    print(f"\nDiffers from lookup table: "
          f"{metrics['n_differs_from_lookup']}/{metrics['n_datasets']}")

    print(f"\nDescriptor diversity: {metrics['descriptor_diversity']} unique descriptors")
    print(f"  {', '.join(metrics['unique_descriptors'])}")

    print(f"\nObject type accuracy:")
    print(f"  INTERPRET (vision): {metrics['ot_interpret_correct']}")
    print(f"  After reconciliation: {metrics['ot_reconciled_correct']}")
    print(f"  Corrections made: {metrics['ot_corrections_made']}")

    print(f"\nPer-dataset details:")
    print(f"{'Dataset':<20} {'Behavior':<25} {'Hypothesis':<25} {'Final':<25} {'Benchmark':<25} {'OT guess->recon (GT)'}")
    print("-" * 140)
    for r in results:
        if "error" in r:
            print(f"{r['dataset']:<20} ERROR: {r['error']}")
            continue
        print(f"{r['dataset']:<20} {r['behavior']:<25} {r['descriptor_hypothesis']:<25} "
              f"{r['final_descriptor']:<25} {r['benchmark_top']:<25} "
              f"{r['interpret_object_type']}->{r['reconciled_object_type']} "
              f"({r['gt_object_type']})")

    # Save full results
    output = {
        "metrics": metrics,
        "per_dataset": results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
