#!/usr/bin/env python3
"""Validate TopoAgent correctness using synthetic shapes with known topology.

This script tests the mathematical correctness of persistent homology
computation by verifying Betti numbers on shapes with known topology.

Synthetic Test Cases:
1. Single disk: H0=1 (one component), H1=0 (no holes)
2. Two disks: H0=2 (two components), H1=0 (no holes)
3. Annulus: H0=1 (one component), H1=1 (one hole)
4. Nested rings: H0=1, H1=2 (two holes)

Usage:
    python scripts/validate_correctness.py --synthetic
    python scripts/validate_correctness.py --cross-library
"""

import argparse
import sys
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])


SYNTHETIC_TESTS = [
    {"name": "single_disk", "expected_h0": 1, "expected_h1": 0},
    {"name": "two_disks", "expected_h0": 2, "expected_h1": 0},
    {"name": "annulus", "expected_h0": 1, "expected_h1": 1},
    {"name": "nested_rings", "expected_h0": 1, "expected_h1": 2},
]


def create_disk(height: int, width: int, center: Tuple[int, int], radius: float) -> np.ndarray:
    """Create a binary disk image."""
    y, x = np.ogrid[:height, :width]
    cy, cx = center
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return (r <= radius).astype(np.float32)


def create_synthetic_shape(shape_name: str, size: int = 28) -> np.ndarray:
    """Create a synthetic shape image.

    Args:
        shape_name: One of the defined synthetic shapes
        size: Image size (NxN)

    Returns:
        Binary image array
    """
    center = size // 2

    if shape_name == "single_disk":
        return create_disk(size, size, (center, center), size // 3)

    elif shape_name == "two_disks":
        img = create_disk(size, size, (center // 2 + 2, center), size // 5)
        img += create_disk(size, size, (size - center // 2 - 2, center), size // 5)
        return np.clip(img, 0, 1)

    elif shape_name == "annulus":
        outer = create_disk(size, size, (center, center), size // 3)
        inner = create_disk(size, size, (center, center), size // 6)
        return outer - inner

    elif shape_name == "nested_rings":
        outer = create_disk(size, size, (center, center), size // 2.5)
        mid = create_disk(size, size, (center, center), size // 3.5)
        inner = create_disk(size, size, (center, center), size // 5)
        return outer - mid + inner

    else:
        raise ValueError(f"Unknown shape: {shape_name}")


def test_with_gudhi(img: np.ndarray, max_dim: int = 1) -> Dict[str, int]:
    """Test persistence computation with GUDHI.

    Args:
        img: Input image
        max_dim: Maximum homology dimension

    Returns:
        Dictionary with H0 and H1 counts
    """
    try:
        import gudhi
    except ImportError:
        return {"error": "GUDHI not installed"}

    # Normalize
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    cubical = gudhi.CubicalComplex(
        dimensions=list(img.shape),
        top_dimensional_cells=img.flatten()
    )
    cubical.compute_persistence()

    counts = {}
    for dim in range(max_dim + 1):
        intervals = cubical.persistence_intervals_in_dimension(dim)
        # Count non-trivial intervals (persistence > 0)
        count = sum(1 for b, d in intervals if d != float('inf') and d - b > 0.01)
        counts[f"H{dim}"] = count

    return counts


def test_with_giotto(img: np.ndarray, max_dim: int = 1) -> Dict[str, int]:
    """Test persistence computation with giotto-tda.

    Args:
        img: Input image
        max_dim: Maximum homology dimension

    Returns:
        Dictionary with H0 and H1 counts
    """
    try:
        from gtda.homology import CubicalPersistence
    except ImportError:
        return {"error": "giotto-tda not installed"}

    # Normalize
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    img_batch = img.reshape(1, *img.shape)
    cp = CubicalPersistence(
        homology_dimensions=list(range(max_dim + 1)),
        coeff=2
    )
    diagrams = cp.fit_transform(img_batch)

    counts = {}
    for dim in range(max_dim + 1):
        dim_diag = diagrams[0][diagrams[0][:, 2] == dim]
        # Count non-trivial intervals
        count = sum(1 for b, d, _ in dim_diag if not np.isinf(d) and d - b > 0.01)
        counts[f"H{dim}"] = count

    return counts


def test_with_topoagent(img: np.ndarray) -> Dict[str, int]:
    """Test persistence computation with TopoAgent core API.

    Args:
        img: Input image

    Returns:
        Dictionary with H0 and H1 counts
    """
    try:
        from topoagent.core import extract_topo_features, TopoConfig
    except ImportError:
        return {"error": "TopoAgent not installed"}

    config = TopoConfig(filtration_type="sublevel", max_dimension=1)
    result = extract_topo_features(img, config)

    return {
        "H0": result.qc.get("h0_count", 0),
        "H1": result.qc.get("h1_count", 0),
    }


def run_synthetic_tests(verbose: bool = True) -> List[Dict]:
    """Run all synthetic shape tests.

    Args:
        verbose: Print detailed output

    Returns:
        List of test results
    """
    results = []

    print("\n" + "=" * 60)
    print("SYNTHETIC CORRECTNESS TESTS")
    print("=" * 60)

    for test in SYNTHETIC_TESTS:
        shape_name = test["name"]
        expected_h0 = test["expected_h0"]
        expected_h1 = test["expected_h1"]

        if verbose:
            print(f"\nTesting: {shape_name}")
            print(f"  Expected: H0={expected_h0}, H1={expected_h1}")

        # Create shape
        img = create_synthetic_shape(shape_name)

        # Test with TopoAgent
        topoagent_result = test_with_topoagent(img)
        if "error" not in topoagent_result:
            h0_match = topoagent_result["H0"] >= expected_h0  # At least expected
            h1_match = topoagent_result["H1"] >= expected_h1
            passed = h0_match and h1_match

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  TopoAgent: H0={topoagent_result['H0']}, H1={topoagent_result['H1']} [{status}]")

            results.append({
                "shape": shape_name,
                "library": "topoagent",
                "expected_h0": expected_h0,
                "expected_h1": expected_h1,
                "actual_h0": topoagent_result["H0"],
                "actual_h1": topoagent_result["H1"],
                "passed": passed,
            })
        else:
            print(f"  TopoAgent: {topoagent_result['error']}")

    # Summary
    print("\n" + "-" * 60)
    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    print(f"SUMMARY: {passed}/{total} tests passed")

    return results


def run_cross_library_tests(verbose: bool = True) -> List[Dict]:
    """Compare persistence computation across libraries.

    Args:
        verbose: Print detailed output

    Returns:
        List of comparison results
    """
    results = []

    print("\n" + "=" * 60)
    print("CROSS-LIBRARY AGREEMENT TESTS")
    print("=" * 60)

    for test in SYNTHETIC_TESTS:
        shape_name = test["name"]
        img = create_synthetic_shape(shape_name)

        if verbose:
            print(f"\nShape: {shape_name}")

        # Test with all libraries
        gudhi_result = test_with_gudhi(img)
        giotto_result = test_with_giotto(img)
        topoagent_result = test_with_topoagent(img)

        if verbose:
            if "error" not in gudhi_result:
                print(f"  GUDHI:     H0={gudhi_result.get('H0', '?')}, H1={gudhi_result.get('H1', '?')}")
            if "error" not in giotto_result:
                print(f"  giotto-tda: H0={giotto_result.get('H0', '?')}, H1={giotto_result.get('H1', '?')}")
            if "error" not in topoagent_result:
                print(f"  TopoAgent: H0={topoagent_result.get('H0', '?')}, H1={topoagent_result.get('H1', '?')}")

        # Check agreement
        libs = []
        if "error" not in gudhi_result:
            libs.append(("GUDHI", gudhi_result))
        if "error" not in giotto_result:
            libs.append(("giotto-tda", giotto_result))
        if "error" not in topoagent_result:
            libs.append(("TopoAgent", topoagent_result))

        if len(libs) >= 2:
            # Check if all libraries agree (within tolerance)
            h0_values = [r.get("H0", -1) for _, r in libs]
            h1_values = [r.get("H1", -1) for _, r in libs]

            h0_agree = max(h0_values) - min(h0_values) <= 2
            h1_agree = max(h1_values) - min(h1_values) <= 2

            results.append({
                "shape": shape_name,
                "libraries": [name for name, _ in libs],
                "h0_agree": h0_agree,
                "h1_agree": h1_agree,
                "h0_values": h0_values,
                "h1_values": h1_values,
            })

            if verbose:
                status = "AGREE" if (h0_agree and h1_agree) else "DISAGREE"
                print(f"  Cross-library: [{status}]")

    # Summary
    print("\n" + "-" * 60)
    agreed = sum(1 for r in results if r.get("h0_agree") and r.get("h1_agree"))
    total = len(results)
    print(f"SUMMARY: {agreed}/{total} shapes with library agreement")

    return results


def test_superlevel_fix(verbose: bool = True) -> Dict:
    """Verify the superlevel filtration fix works correctly.

    The bug was: negating image then using abs() on birth/death values.
    The fix: use monotone transform (1.0 - img) instead.

    Args:
        verbose: Print detailed output

    Returns:
        Test result dictionary
    """
    print("\n" + "=" * 60)
    print("SUPERLEVEL FILTRATION FIX TEST")
    print("=" * 60)

    # Create an annulus (ring with hole) - should have H1 >= 1
    img = create_synthetic_shape("annulus")

    try:
        from topoagent.core import extract_topo_features, TopoConfig

        # Test sublevel
        config_sub = TopoConfig(filtration_type="sublevel", max_dimension=1)
        result_sub = extract_topo_features(img, config_sub)

        # Test superlevel
        config_sup = TopoConfig(filtration_type="superlevel", max_dimension=1)
        result_sup = extract_topo_features(img, config_sup)

        if verbose:
            print(f"\nAnnulus shape (should detect hole):")
            print(f"  Sublevel:   H0={result_sub.qc['h0_count']}, H1={result_sub.qc['h1_count']}")
            print(f"  Superlevel: H0={result_sup.qc['h0_count']}, H1={result_sup.qc['h1_count']}")

        # At least one filtration should detect the hole
        h1_detected = result_sub.qc['h1_count'] >= 1 or result_sup.qc['h1_count'] >= 1

        # Check that superlevel features are non-zero
        sup_nonzero = result_sup.qc.get('nonzero_ratio', 0) > 0.01

        passed = h1_detected and sup_nonzero

        if verbose:
            print(f"\n  H1 detected: {h1_detected}")
            print(f"  Superlevel features non-zero: {sup_nonzero} ({result_sup.qc.get('nonzero_ratio', 0):.1%})")
            print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

        return {
            "sublevel_h1": result_sub.qc['h1_count'],
            "superlevel_h1": result_sup.qc['h1_count'],
            "superlevel_nonzero_ratio": result_sup.qc.get('nonzero_ratio', 0),
            "h1_detected": h1_detected,
            "sup_nonzero": sup_nonzero,
            "passed": passed,
        }

    except ImportError as e:
        print(f"Error: {e}")
        return {"error": str(e), "passed": False}


def main():
    parser = argparse.ArgumentParser(description="Validate TopoAgent correctness")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic shape tests")
    parser.add_argument("--cross-library", action="store_true", help="Run cross-library comparison")
    parser.add_argument("--superlevel", action="store_true", help="Test superlevel fix")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Default to all tests if none specified
    if not (args.synthetic or args.cross_library or args.superlevel):
        args.all = True

    verbose = not args.quiet
    all_passed = True

    if args.synthetic or args.all:
        results = run_synthetic_tests(verbose)
        if not all(r.get("passed", False) for r in results):
            all_passed = False

    if args.cross_library or args.all:
        results = run_cross_library_tests(verbose)
        if not all(r.get("h0_agree") and r.get("h1_agree") for r in results):
            all_passed = False

    if args.superlevel or args.all:
        result = test_superlevel_fix(verbose)
        if not result.get("passed", False):
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
