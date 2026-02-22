#!/usr/bin/env python
"""Quick test script for descriptor tools.

Usage:
    python scripts/test_descriptors.py              # Quick test with synthetic data
    python scripts/test_descriptors.py --real-image # Test with real MedMNIST image
    python scripts/test_descriptors.py --verbose    # Verbose output
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Test descriptor tools")
    parser.add_argument("--real-image", action="store_true", help="Test with real MedMNIST image")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer samples)")

    args = parser.parse_args()

    print("="*60)
    print("TopoAgent Descriptor Tests")
    print("="*60)

    # Run correctness tests
    from descriptor_experiments.tests.test_correctness import test_all_descriptors
    results = test_all_descriptors(verbose=True)

    n_passed = sum(results.values())
    n_total = len(results)

    if n_passed == n_total:
        print("\n✓ All descriptor tests passed!")
        return 0
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n✗ {len(failed)} tests failed: {failed}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
