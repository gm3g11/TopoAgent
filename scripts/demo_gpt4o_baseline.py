#!/usr/bin/env python3
"""GPT-4o Baseline — Ask the same question directly to a general LLM.

No skills, no tools, no memory, no reflection.
Just the raw LLM with the same user query.

Usage:
    python scripts/demo_gpt4o_baseline.py --dataset BloodMNIST
    python scripts/demo_gpt4o_baseline.py --all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

DATASET_INFO = {
    "BloodMNIST":  "discrete_cells",
    "PathMNIST":   "glands_lumens",
    "RetinaMNIST": "vessel_trees",
    "DermaMNIST":  "surface_lesions",
    "OrganAMNIST": "organ_shape",
}


def ask_gpt4o(dataset_name: str):
    """Ask GPT-4o the exact same question TopoAgent receives."""
    object_type = DATASET_INFO[dataset_name]

    query = (
        f"Calculate the optimal TDA descriptor for the provided medical image. "
        f"Dataset: {dataset_name}. "
        f"The image shows {object_type.replace('_', ' ')} structures."
    )

    print("=" * 80)
    print(f"  GPT-4o Baseline: {dataset_name}")
    print("=" * 80)

    print(f"\n[USER QUERY]")
    print(f'  "{query}"')

    model = ChatOpenAI(model="gpt-4o", temperature=0.7)

    t0 = time.time()
    response = model.invoke([HumanMessage(content=query)])
    elapsed = time.time() - t0

    print(f"\n[GPT-4o RESPONSE] ({elapsed:.1f}s, {len(response.content)} chars)")
    print("-" * 80)
    print(response.content)
    print("-" * 80)

    result = {
        "dataset": dataset_name,
        "query": query,
        "response": response.content,
        "time_seconds": round(elapsed, 2),
        "model": "gpt-4o",
        "skills": None,
        "tools": None,
        "memory": None,
        "reflection": None,
    }

    outdir = PROJECT_ROOT / "results" / "case_studies"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dataset_name}_gpt4o_baseline.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to: {path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="GPT-4o Baseline")
    parser.add_argument("--dataset", type=str, default="BloodMNIST",
                        choices=list(DATASET_INFO.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for ds in DATASET_INFO:
            ask_gpt4o(ds)
            print()
    else:
        ask_gpt4o(args.dataset)


if __name__ == "__main__":
    main()
