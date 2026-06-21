#!/usr/bin/env python3
"""Protocol 2 — Classifier Evaluation (Live Feature Extraction).

Validates that the agent's descriptor choices produce competitive classification
accuracy with real feature extraction and 6-classifier evaluation.

Methods:
    oracle     — Best descriptor per dataset
    topoagent  — Agent's selection (requires OPENAI_API_KEY)
    sba        — persistence_statistics
    rule_based — TOP_PERFORMERS[object_type][0]

Deduplication: If multiple methods pick the same descriptor for a dataset,
features are only extracted once.

Usage:
    python scripts/protocol2_classifier_eval.py
    python scripts/protocol2_classifier_eval.py --n-eval 200 --cv-folds 3
    python scripts/protocol2_classifier_eval.py --dataset BloodMNIST
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DATASETS = [
    {"name": "BloodMNIST", "object_type": "discrete_cells", "color_mode": "per_channel"},
    {"name": "PathMNIST", "object_type": "glands_lumens", "color_mode": "per_channel"},
    {"name": "RetinaMNIST", "object_type": "vessel_trees", "color_mode": "per_channel"},
    {"name": "DermaMNIST", "object_type": "surface_lesions", "color_mode": "per_channel"},
    {"name": "OrganAMNIST", "object_type": "organ_shape", "color_mode": "grayscale"},
]

CLASSIFIERS = ["TabPFN", "XGBoost", "CatBoost", "RandomForest", "TabM", "RealMLP"]

METHODS = ["oracle", "topoagent", "sba", "rule_based"]


# =============================================================================
# Descriptor Selection per Method
# =============================================================================

def get_oracle_descriptor(dataset_name: str) -> str:
    """Best descriptor per dataset from accuracy_lookup.json."""
    assets_dir = PROJECT_ROOT / "results" / "topobenchmark" / "assets"
    lookup_path = assets_dir / "accuracy_lookup.json"
    with open(lookup_path) as f:
        lookup = json.load(f)

    ds_accs = lookup.get(dataset_name, {})
    if not ds_accs:
        return "persistence_statistics"
    return max(ds_accs, key=ds_accs.get)


def get_topoagent_descriptor(dataset_name: str, model) -> str:
    """Run the agent's descriptor selection (1 LLM call in lookup mode)."""
    from topoagent.benchmark_runner import TopoBenchmarkRunner
    from topoagent.memory.skill_memory import SkillMemory

    runner = TopoBenchmarkRunner(
        model=model,
        skill_memory=SkillMemory(),
        lookup_mode=True,
    )
    result = runner.evaluate_dataset(
        dataset_name=dataset_name,
        method="topoagent",
    )
    return result.get("descriptor_selected", "persistence_statistics")


def get_sba_descriptor() -> str:
    """Single Best Algorithm: persistence_statistics."""
    return "persistence_statistics"


def get_rule_based_descriptor(object_type: str) -> str:
    """TOP_PERFORMERS[object_type][0], no LLM."""
    from topoagent.skills.rules_data import SUPPORTED_TOP_PERFORMERS
    performers = SUPPORTED_TOP_PERFORMERS.get(object_type, [])
    if performers:
        return performers[0]["descriptor"]
    return "persistence_statistics"


# =============================================================================
# Feature Extraction + CV Evaluation
# =============================================================================

def extract_and_evaluate(
    dataset_name: str,
    descriptor: str,
    object_type: str,
    color_mode: str,
    n_eval: int,
    cv_folds: int,
    classifiers: List[str],
    seed: int = 42,
) -> Dict[str, Any]:
    """Extract features for one (dataset, descriptor) pair and evaluate with multiple classifiers.

    Returns:
        Dict with per_classifier results and best_accuracy.
    """
    from topoagent.benchmark_runner import TopoBenchmarkRunner
    runner = TopoBenchmarkRunner(model=None, lookup_mode=False)

    accuracy, details = runner._batch_extract_and_evaluate(
        dataset_name=dataset_name,
        descriptor=descriptor,
        object_type=object_type,
        color_mode=color_mode,
        n_eval=n_eval,
        cv_folds=cv_folds,
        seed=seed,
        classifiers=classifiers,
    )

    return {
        "descriptor": descriptor,
        "best_accuracy": accuracy,
        "best_classifier": details.get("best_classifier", "?"),
        "feature_dim": details.get("feature_dim", 0),
        "n_samples": details.get("n_samples", 0),
        "per_classifier": details.get("per_classifier", {}),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Protocol 2: Classifier Evaluation")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of evaluation images per dataset (default: 200)")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Number of CV folds (default: 3)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to test (default: all 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset:
        datasets = [d for d in DATASETS if d["name"] == args.dataset]
        if not datasets:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {[d['name'] for d in DATASETS]}")
            sys.exit(1)
    else:
        datasets = DATASETS

    # Create LLM model for topoagent method
    model = None
    if "topoagent" in METHODS:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
        else:
            print("WARNING: OPENAI_API_KEY not set — skipping 'topoagent' method")

    print(f"\n{'='*80}")
    print(f"CLASSIFIER EVALUATION (Live Feature Extraction)")
    print(f"{'='*80}")
    print(f"  Datasets:    {[d['name'] for d in datasets]}")
    print(f"  Methods:     {METHODS}")
    print(f"  Classifiers: {CLASSIFIERS}")
    print(f"  n_eval:      {args.n_eval}")
    print(f"  cv_folds:    {args.cv_folds}")

    # Step 1: Collect descriptor selections per method per dataset
    print(f"\n{'─'*80}")
    print("STEP 1: Collecting descriptor selections")
    print(f"{'─'*80}")

    # method -> dataset -> descriptor
    selections = {}
    for method in METHODS:
        selections[method] = {}
        for ds in datasets:
            ds_name = ds["name"]
            object_type = ds["object_type"]

            if method == "oracle":
                desc = get_oracle_descriptor(ds_name)
            elif method == "topoagent":
                if model is None:
                    desc = "persistence_statistics"  # fallback
                else:
                    desc = get_topoagent_descriptor(ds_name, model)
            elif method == "sba":
                desc = get_sba_descriptor()
            elif method == "rule_based":
                desc = get_rule_based_descriptor(object_type)
            else:
                desc = "persistence_statistics"

            selections[method][ds_name] = desc
            print(f"  {method:>20} / {ds_name:<14} -> {desc}")

    # Step 2: Deduplicate (dataset, descriptor) pairs
    print(f"\n{'─'*80}")
    print("STEP 2: Deduplicating extraction pairs")
    print(f"{'─'*80}")

    unique_pairs = set()
    for method in METHODS:
        for ds in datasets:
            ds_name = ds["name"]
            desc = selections[method][ds_name]
            unique_pairs.add((ds_name, desc))

    print(f"  Total method-dataset combos: {len(METHODS) * len(datasets)}")
    print(f"  Unique (dataset, descriptor) pairs: {len(unique_pairs)}")

    # Step 3: Extract features and evaluate for each unique pair
    print(f"\n{'─'*80}")
    print("STEP 3: Feature extraction + CV evaluation")
    print(f"{'─'*80}")

    # (dataset, descriptor) -> evaluation result
    eval_cache = {}
    ds_config_map = {d["name"]: d for d in datasets}

    for i, (ds_name, desc) in enumerate(sorted(unique_pairs)):
        ds = ds_config_map[ds_name]
        print(f"  [{i+1}/{len(unique_pairs)}] {ds_name} / {desc}...", end=" ", flush=True)

        start = time.time()
        try:
            result = extract_and_evaluate(
                dataset_name=ds_name,
                descriptor=desc,
                object_type=ds["object_type"],
                color_mode=ds["color_mode"],
                n_eval=args.n_eval,
                cv_folds=args.cv_folds,
                classifiers=CLASSIFIERS,
                seed=args.seed,
            )
            elapsed = time.time() - start
            result["elapsed"] = elapsed
            eval_cache[(ds_name, desc)] = result
            best = result["best_accuracy"]
            best_clf = result["best_classifier"]
            dim = result["feature_dim"]
            print(f"best={best:.1%} ({best_clf}), dim={dim}, {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"FAILED ({elapsed:.1f}s): {e}")
            eval_cache[(ds_name, desc)] = {
                "descriptor": desc,
                "best_accuracy": 0.0,
                "error": str(e),
                "elapsed": elapsed,
            }

    # Step 4: Build results matrix
    print(f"\n{'='*80}")
    print("RESULTS: Best Accuracy per Method per Dataset")
    print(f"{'='*80}")

    # Header
    ds_names = [d["name"] for d in datasets]
    header = f"{'Method':<22}"
    for ds in ds_names:
        header += f" | {ds:>14}"
    header += f" | {'Mean':>7}"
    print(header)
    print("-" * len(header))

    method_means = {}
    for method in METHODS:
        line = f"{method:<22}"
        accs = []
        for ds in ds_names:
            desc = selections[method][ds]
            result = eval_cache.get((ds, desc), {})
            acc = result.get("best_accuracy", 0.0)
            accs.append(acc)
            line += f" | {acc:>13.1%}"
        mean = np.mean(accs) if accs else 0.0
        method_means[method] = mean
        line += f" | {mean:>6.1%}"
        print(line)

    # Oracle regret
    print(f"\n{'─'*80}")
    print("REGRET vs ORACLE (Best Accuracy)")
    print(f"{'─'*80}")
    oracle_accs = {}
    for ds in ds_names:
        desc = selections["oracle"][ds]
        result = eval_cache.get((ds, desc), {})
        oracle_accs[ds] = result.get("best_accuracy", 0.0)

    for method in METHODS:
        if method == "oracle":
            continue
        regrets = []
        for ds in ds_names:
            desc = selections[method][ds]
            result = eval_cache.get((ds, desc), {})
            acc = result.get("best_accuracy", 0.0)
            regret = oracle_accs[ds] - acc
            regrets.append(regret)
        mean_regret = np.mean(regrets)
        print(f"  {method:<22} mean_regret={mean_regret:+.3f}")

    # Per-classifier detail for topoagent
    if "topoagent" in METHODS:
        print(f"\n{'─'*80}")
        print("PER-CLASSIFIER ACCURACY (topoagent method)")
        print(f"{'─'*80}")
        header2 = f"{'Dataset':<14} {'Descriptor':<22}"
        for clf in CLASSIFIERS:
            header2 += f" | {clf:>10}"
        print(header2)
        print("-" * len(header2))
        for ds in ds_names:
            desc = selections["topoagent"][ds]
            result = eval_cache.get((ds, desc), {})
            per_clf = result.get("per_classifier", {})
            line = f"{ds:<14} {desc:<22}"
            for clf in CLASSIFIERS:
                acc = per_clf.get(clf, {}).get("accuracy", 0.0)
                line += f" | {acc:>9.1%}"
            print(line)

    # Save results
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"classifier_eval_{timestamp}.json"

    # Make serializable
    serializable_cache = {}
    for (ds, desc), result in eval_cache.items():
        key = f"{ds}|{desc}"
        serializable_cache[key] = result

    output = {
        "config": {
            "datasets": ds_names,
            "methods": METHODS,
            "classifiers": CLASSIFIERS,
            "n_eval": args.n_eval,
            "cv_folds": args.cv_folds,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "selections": selections,
        "eval_cache": serializable_cache,
        "method_means": method_means,
        "oracle_accs": oracle_accs,
    }

    serializable = json.loads(json.dumps(output, default=str))
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
