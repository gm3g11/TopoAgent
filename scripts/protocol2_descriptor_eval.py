#!/usr/bin/env python3
"""Protocol 2 — Descriptor Selection Evaluation (Lookup Mode).

Evaluates whether the agent's descriptor selections are good by comparing
against baselines using pre-computed accuracy from Benchmark4.

Methods:
    oracle         — Best descriptor per dataset (upper bound)
    topoagent      — Full skills + benchmark rankings (our method)
    topoagent_no_skills — LLM + sample stats, no expert rules
    sba            — Single Best Algorithm: persistence_statistics
    rule_based     — TOP_PERFORMERS[object_type][0], no LLM
    random         — Random descriptor (lower bound, avg over 10 seeds)

Usage:
    python scripts/protocol2_descriptor_eval.py
    python scripts/protocol2_descriptor_eval.py --methods oracle,topoagent,sba,random
    python scripts/protocol2_descriptor_eval.py --dataset BloodMNIST,PathMNIST
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DATASETS = ["BloodMNIST", "PathMNIST", "RetinaMNIST", "DermaMNIST", "OrganAMNIST"]

DEFAULT_METHODS = [
    "oracle",
    "topoagent",
    "topoagent_no_skills",
    "sba",
    "rule_based",
    "random",
]

NO_LLM_METHODS = {"oracle", "sba", "rule_based", "random"}


def main():
    parser = argparse.ArgumentParser(description="Protocol 2: Descriptor Selection Evaluation")
    parser.add_argument("--dataset", type=str, default=",".join(DATASETS),
                        help="Comma-separated dataset names")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                        help="Comma-separated method names")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.dataset.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    # Check if any methods need LLM
    needs_llm = any(m not in NO_LLM_METHODS for m in methods)

    model = None
    if needs_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set (required for LLM methods)")
            sys.exit(1)
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

    from topoagent.benchmark_runner import TopoBenchmarkRunner
    from topoagent.memory.skill_memory import SkillMemory

    skill_memory = SkillMemory()
    runner = TopoBenchmarkRunner(
        model=model,
        skill_memory=skill_memory,
        lookup_mode=True,
    )

    print(f"\n{'='*80}")
    print(f"DESCRIPTOR SELECTION EVALUATION (Lookup Mode)")
    print(f"{'='*80}")
    print(f"  Datasets: {datasets}")
    print(f"  Methods:  {methods}")

    # Run evaluation
    start = time.time()
    all_output = runner.evaluate_all(
        datasets=datasets,
        methods=methods,
        seed=args.seed,
    )
    elapsed = time.time() - start

    # Print accuracy matrix table
    summary = all_output.get("summary", {})
    print(f"\n{'='*80}")
    print("ACCURACY MATRIX (Balanced Accuracy %)")
    print(f"{'='*80}")
    table = runner.format_matrix_table(summary, datasets)
    print(table)

    # Print aggregate metrics
    per_method = summary.get("per_method", {})
    if per_method:
        print(f"\n{'─'*80}")
        print("AGGREGATE METRICS")
        print(f"{'─'*80}")
        print(f"{'Method':<26} {'Mean Acc':>8} {'Regret':>8} {'Top-1':>7} {'Top-3':>7} {'LLM':>5}")
        print(f"{'-'*26} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*5}")
        for method, stats in per_method.items():
            mean_acc = stats.get("mean_accuracy", 0)
            mean_regret = stats.get("mean_regret", 0)
            top1 = stats.get("top1_rate", 0)
            top3 = stats.get("top3_rate", 0)
            llm_calls = stats.get("total_llm_calls", 0)
            print(f"{method:<26} {mean_acc:>7.1%} {mean_regret:>+7.3f} "
                  f"{top1:>6.1%} {top3:>6.1%} {llm_calls:>5}")

    # Print descriptor choices
    desc_matrix = summary.get("descriptor_matrix", {})
    if desc_matrix:
        print(f"\n{'─'*80}")
        print("DESCRIPTOR CHOICES")
        print(f"{'─'*80}")
        for method in methods:
            ds_descs = desc_matrix.get(method, {})
            choices = [ds_descs.get(ds, "?") for ds in datasets]
            print(f"  {method:<26} {choices}")

    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"descriptor_eval_{timestamp}.json"

    serializable = json.loads(json.dumps(all_output, default=str))
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
