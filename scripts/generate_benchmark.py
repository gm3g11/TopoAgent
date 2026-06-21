#!/usr/bin/env python
"""Generate TopoQA benchmark dataset.

Usage:
    python scripts/generate_benchmark.py --output benchmark/topoqa_v1.json
    python scripts/generate_benchmark.py --output benchmark/topoqa_v1.json --n-samples 50
"""

import os
import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.topoqa import generate_topoqa_benchmark


def main():
    parser = argparse.ArgumentParser(description="Generate TopoQA benchmark")

    parser.add_argument("--output", "-o", type=str, default="benchmark/topoqa_v1.json",
                       help="Output JSON file path")
    parser.add_argument("--n-samples", "-n", type=int, default=200,
                       help="Samples per dataset")
    parser.add_argument("--datasets", "-d", nargs="+",
                       default=["dermamnist", "pathmnist", "retinamnist", "pneumoniamnist"],
                       help="MedMNIST datasets to use")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    print("=" * 50)
    print("TopoQA Benchmark Generator")
    print("=" * 50)
    print(f"Output: {args.output}")
    print(f"Samples per dataset: {args.n_samples}")
    print(f"Datasets: {args.datasets}")
    print(f"Seed: {args.seed}")
    print()

    result = generate_topoqa_benchmark(
        output_path=args.output,
        n_samples_per_dataset=args.n_samples,
        datasets=args.datasets,
        seed=args.seed
    )

    print()
    print("=" * 50)
    print("Generation Complete")
    print("=" * 50)
    print(f"Total questions: {result['total_questions']}")
    print(f"Datasets: {', '.join(result['datasets'])}")
    print(f"Categories: {', '.join(result['categories'])}")
    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()
