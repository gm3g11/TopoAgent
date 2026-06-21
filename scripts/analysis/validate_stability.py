#!/usr/bin/env python3
"""Validate TopoAgent stability under perturbations.

This script tests the robustness of topology features to common
image perturbations: noise, blur, and rotation.

Key Metrics:
- Feature drift (cosine similarity between original and perturbed)
- Stability score (mean similarity across perturbation levels)

Usage:
    python scripts/validate_stability.py --noise
    python scripts/validate_stability.py --blur
    python scripts/validate_stability.py --rotation
    python scripts/validate_stability.py --all
"""

import argparse
import sys
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])


PERTURBATION_LEVELS = {
    "noise": [0.01, 0.05, 0.1, 0.2],
    "blur": [0.5, 1.0, 2.0, 3.0],
    "rotation": [5, 15, 30, 45],
}


def create_test_image(size: int = 28) -> np.ndarray:
    """Create a test image with known topology (annulus).

    Args:
        size: Image size

    Returns:
        Test image array
    """
    center = size // 2
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Create annulus
    outer_r = size // 3
    inner_r = size // 6
    img = ((r <= outer_r) & (r >= inner_r)).astype(np.float32)

    return img


def add_gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to image.

    Args:
        img: Input image
        std: Noise standard deviation

    Returns:
        Noisy image
    """
    noise = np.random.normal(0, std, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)


def add_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to image.

    Args:
        img: Input image
        sigma: Blur sigma

    Returns:
        Blurred image
    """
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=sigma)
    except ImportError:
        # Simple box blur fallback
        from scipy.ndimage import uniform_filter
        return uniform_filter(img, size=int(sigma * 2))


def add_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle.

    Args:
        img: Input image
        angle: Rotation angle in degrees

    Returns:
        Rotated image
    """
    try:
        from scipy.ndimage import rotate
        rotated = rotate(img, angle, reshape=False, mode='constant', cval=0)
        return np.clip(rotated, 0, 1)
    except ImportError:
        return img  # No rotation if scipy not available


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity (0 to 1)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract topology features from image.

    Args:
        img: Input image

    Returns:
        Feature vector
    """
    try:
        from topoagent.core import extract_topo_features, TopoConfig
        config = TopoConfig(filtration_type="sublevel", max_dimension=1)
        result = extract_topo_features(img, config)
        return result.vector
    except ImportError:
        raise ImportError("TopoAgent not installed")


def test_noise_stability(verbose: bool = True) -> Dict:
    """Test stability under Gaussian noise.

    Args:
        verbose: Print detailed output

    Returns:
        Stability results
    """
    print("\n" + "-" * 60)
    print("NOISE STABILITY TEST")
    print("-" * 60)

    img = create_test_image()
    original_features = extract_features(img)

    results = []
    for noise_std in PERTURBATION_LEVELS["noise"]:
        # Run multiple trials for noise
        similarities = []
        for _ in range(5):
            noisy = add_gaussian_noise(img, noise_std)
            noisy_features = extract_features(noisy)
            sim = cosine_similarity(original_features, noisy_features)
            similarities.append(sim)

        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        if verbose:
            print(f"  Noise std={noise_std:.2f}: similarity={mean_sim:.4f} (+/- {std_sim:.4f})")

        results.append({
            "noise_std": noise_std,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
        })

    # Compute overall stability score
    stability_score = np.mean([r["mean_similarity"] for r in results])
    print(f"\n  Overall stability score: {stability_score:.4f}")

    return {
        "perturbation_type": "noise",
        "results": results,
        "stability_score": stability_score,
        "passed": stability_score > 0.7,  # Threshold for "stable"
    }


def test_blur_stability(verbose: bool = True) -> Dict:
    """Test stability under Gaussian blur.

    Args:
        verbose: Print detailed output

    Returns:
        Stability results
    """
    print("\n" + "-" * 60)
    print("BLUR STABILITY TEST")
    print("-" * 60)

    img = create_test_image()
    original_features = extract_features(img)

    results = []
    for blur_sigma in PERTURBATION_LEVELS["blur"]:
        blurred = add_blur(img, blur_sigma)
        blurred_features = extract_features(blurred)
        sim = cosine_similarity(original_features, blurred_features)

        if verbose:
            print(f"  Blur sigma={blur_sigma:.1f}: similarity={sim:.4f}")

        results.append({
            "blur_sigma": blur_sigma,
            "similarity": sim,
        })

    # Compute overall stability score
    stability_score = np.mean([r["similarity"] for r in results])
    print(f"\n  Overall stability score: {stability_score:.4f}")

    return {
        "perturbation_type": "blur",
        "results": results,
        "stability_score": stability_score,
        "passed": stability_score > 0.6,
    }


def test_rotation_stability(verbose: bool = True) -> Dict:
    """Test stability under rotation.

    Args:
        verbose: Print detailed output

    Returns:
        Stability results
    """
    print("\n" + "-" * 60)
    print("ROTATION STABILITY TEST")
    print("-" * 60)

    img = create_test_image()
    original_features = extract_features(img)

    results = []
    for angle in PERTURBATION_LEVELS["rotation"]:
        rotated = add_rotation(img, angle)
        rotated_features = extract_features(rotated)
        sim = cosine_similarity(original_features, rotated_features)

        if verbose:
            print(f"  Rotation={angle:d} deg: similarity={sim:.4f}")

        results.append({
            "angle": angle,
            "similarity": sim,
        })

    # Compute overall stability score
    stability_score = np.mean([r["similarity"] for r in results])
    print(f"\n  Overall stability score: {stability_score:.4f}")

    return {
        "perturbation_type": "rotation",
        "results": results,
        "stability_score": stability_score,
        "passed": stability_score > 0.5,  # Rotation is harder
    }


def test_combined_perturbations(verbose: bool = True) -> Dict:
    """Test stability under combined perturbations.

    Args:
        verbose: Print detailed output

    Returns:
        Stability results
    """
    print("\n" + "-" * 60)
    print("COMBINED PERTURBATION TEST")
    print("-" * 60)

    img = create_test_image()
    original_features = extract_features(img)

    # Apply noise + blur + rotation
    perturbed = add_gaussian_noise(img, 0.05)
    perturbed = add_blur(perturbed, 1.0)
    perturbed = add_rotation(perturbed, 15)

    perturbed_features = extract_features(perturbed)
    sim = cosine_similarity(original_features, perturbed_features)

    if verbose:
        print(f"  Combined (noise=0.05, blur=1.0, rot=15): similarity={sim:.4f}")

    return {
        "perturbation_type": "combined",
        "similarity": sim,
        "passed": sim > 0.4,  # Combined is very challenging
    }


def run_all_stability_tests(verbose: bool = True) -> Dict:
    """Run all stability tests.

    Args:
        verbose: Print detailed output

    Returns:
        Combined results
    """
    print("\n" + "=" * 60)
    print("TOPOLOGY FEATURE STABILITY VALIDATION")
    print("=" * 60)

    results = {}

    try:
        results["noise"] = test_noise_stability(verbose)
        results["blur"] = test_blur_stability(verbose)
        results["rotation"] = test_rotation_stability(verbose)
        results["combined"] = test_combined_perturbations(verbose)
    except ImportError as e:
        print(f"\nError: {e}")
        return {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("STABILITY SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, result in results.items():
        if "stability_score" in result:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {name}: stability={result['stability_score']:.4f} [{status}]")
        elif "similarity" in result:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {name}: similarity={result['similarity']:.4f} [{status}]")
        if not result.get("passed", True):
            all_passed = False

    print("\n" + "-" * 60)
    if all_passed:
        print("ALL STABILITY TESTS PASSED")
    else:
        print("SOME STABILITY TESTS FAILED")

    results["all_passed"] = all_passed
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate TopoAgent stability")
    parser.add_argument("--noise", action="store_true", help="Test noise stability")
    parser.add_argument("--blur", action="store_true", help="Test blur stability")
    parser.add_argument("--rotation", action="store_true", help="Test rotation stability")
    parser.add_argument("--combined", action="store_true", help="Test combined perturbations")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Default to all tests if none specified
    if not (args.noise or args.blur or args.rotation or args.combined):
        args.all = True

    verbose = not args.quiet

    try:
        if args.all:
            results = run_all_stability_tests(verbose)
            return 0 if results.get("all_passed", False) else 1

        all_passed = True

        if args.noise:
            result = test_noise_stability(verbose)
            if not result.get("passed", False):
                all_passed = False

        if args.blur:
            result = test_blur_stability(verbose)
            if not result.get("passed", False):
                all_passed = False

        if args.rotation:
            result = test_rotation_stability(verbose)
            if not result.get("passed", False):
                all_passed = False

        if args.combined:
            result = test_combined_perturbations(verbose)
            if not result.get("passed", False):
                all_passed = False

        return 0 if all_passed else 1

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please ensure TopoAgent and dependencies are installed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
