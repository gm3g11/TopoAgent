#!/usr/bin/env python
"""Benchmark descriptor extraction speed.

Usage:
    python scripts/benchmark_descriptor_speed.py                    # Default benchmark
    python scripts/benchmark_descriptor_speed.py --n-images 50      # More images
    python scripts/benchmark_descriptor_speed.py --image-size 128   # Smaller images
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Benchmark descriptor speed")
    parser.add_argument("--n-images", type=int, default=10, help="Number of images")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--n-repeats", type=int, default=3, help="Number of repeats")
    parser.add_argument("--optimized", action="store_true", help="Use optimized PH (cripser)")

    args = parser.parse_args()

    print("="*60)
    print("TopoAgent Descriptor Speed Benchmark")
    print("="*60)

    from descriptor_experiments.tests.test_speed import benchmark_all_descriptors

    results = benchmark_all_descriptors(
        n_images=args.n_images,
        image_size=args.image_size,
        n_repeats=args.n_repeats,
        verbose=True
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
