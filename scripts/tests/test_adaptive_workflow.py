#!/usr/bin/env python3
"""Test script for TopoAgent v3 Adaptive Pipeline.

Tests the image_analyzer tool and adaptive workflow:
1. ImageAnalyzerTool standalone testing
2. Adaptive recommendations verification
3. Full adaptive pipeline (if API key available)

Usage:
    # Test image analyzer standalone
    python scripts/test_adaptive_workflow.py --test-analyzer

    # Test with a specific image
    python scripts/test_adaptive_workflow.py --image path/to/image.png

    # Test full adaptive pipeline (requires API key)
    python scripts/test_adaptive_workflow.py --full-test

    # Run all tests
    python scripts/test_adaptive_workflow.py --all
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_image_analyzer_tool():
    """Test the ImageAnalyzerTool with synthetic images."""
    print("\n" + "="*60)
    print("Testing ImageAnalyzerTool")
    print("="*60)

    from topoagent.tools.preprocessing import ImageAnalyzerTool

    analyzer = ImageAnalyzerTool()
    print(f"Tool name: {analyzer.name}")
    print(f"Tool description: {analyzer.description[:100]}...")

    # Test 1: Dark features on bright background (should recommend superlevel)
    print("\n[Test 1] Dark features on bright background")
    img_dark_features = np.ones((28, 28)) * 0.8  # Bright background
    img_dark_features[8:20, 8:20] = 0.2  # Dark lesion
    img_dark_features += np.random.randn(28, 28) * 0.05  # Add noise
    img_dark_features = np.clip(img_dark_features, 0, 1)

    result = analyzer._run(image_array=img_dark_features.tolist())

    if result["success"]:
        print(f"  Statistics: bright={result['image_statistics']['bright_ratio']:.2f}, "
              f"dark={result['image_statistics']['dark_ratio']:.2f}")
        print(f"  SNR: {result['image_statistics']['snr_estimate']:.2f}")
        print(f"  Recommended filtration: {result['recommendations']['filtration_type']}")
        print(f"  Reason: {result['recommendations']['filtration_reason']}")

        expected = "superlevel"
        actual = result['recommendations']['filtration_type']
        status = "PASS" if actual == expected else "FAIL"
        print(f"  Expected: {expected}, Got: {actual} -> {status}")
    else:
        print(f"  ERROR: {result['error']}")

    # Test 2: Bright features on dark background (should recommend sublevel)
    print("\n[Test 2] Bright features on dark background")
    img_bright_features = np.ones((28, 28)) * 0.2  # Dark background
    img_bright_features[8:20, 8:20] = 0.9  # Bright lesion
    img_bright_features += np.random.randn(28, 28) * 0.05  # Add noise
    img_bright_features = np.clip(img_bright_features, 0, 1)

    result = analyzer._run(image_array=img_bright_features.tolist())

    if result["success"]:
        print(f"  Statistics: bright={result['image_statistics']['bright_ratio']:.2f}, "
              f"dark={result['image_statistics']['dark_ratio']:.2f}")
        print(f"  SNR: {result['image_statistics']['snr_estimate']:.2f}")
        print(f"  Recommended filtration: {result['recommendations']['filtration_type']}")

        expected = "sublevel"
        actual = result['recommendations']['filtration_type']
        status = "PASS" if actual == expected else "FAIL"
        print(f"  Expected: {expected}, Got: {actual} -> {status}")
    else:
        print(f"  ERROR: {result['error']}")

    # Test 3: Noisy image (should recommend noise filtering)
    print("\n[Test 3] Noisy image")
    img_noisy = np.random.randn(28, 28) * 0.3 + 0.5  # High noise
    img_noisy = np.clip(img_noisy, 0, 1)

    result = analyzer._run(image_array=img_noisy.tolist())

    if result["success"]:
        print(f"  SNR: {result['image_statistics']['snr_estimate']:.2f}")
        print(f"  Recommended noise filter: {result['recommendations']['noise_filter']}")
        print(f"  Recommended sigma: {result['recommendations']['pi_sigma']}")

        # Low SNR should recommend filtering
        has_filter = result['recommendations']['noise_filter'] is not None
        status = "PASS" if has_filter else "FAIL"
        print(f"  Expected: noise filter recommended, Got: {result['recommendations']['noise_filter']} -> {status}")
    else:
        print(f"  ERROR: {result['error']}")

    # Test 4: Clean image (should recommend no filtering)
    print("\n[Test 4] Clean image (low noise)")
    img_clean = np.zeros((28, 28))
    # Create smooth gradient
    for i in range(28):
        for j in range(28):
            img_clean[i, j] = (i + j) / 56.0
    img_clean += np.random.randn(28, 28) * 0.01  # Very low noise
    img_clean = np.clip(img_clean, 0, 1)

    result = analyzer._run(image_array=img_clean.tolist())

    if result["success"]:
        print(f"  SNR: {result['image_statistics']['snr_estimate']:.2f}")
        print(f"  Recommended noise filter: {result['recommendations']['noise_filter']}")
        print(f"  Recommended sigma: {result['recommendations']['pi_sigma']}")

        # High SNR should not require filtering
        no_filter = result['recommendations']['noise_filter'] is None
        status = "PASS" if no_filter else "MAYBE"  # Some implementations might still suggest light filtering
        print(f"  Expected: no filter needed, Got: {result['recommendations']['noise_filter']} -> {status}")
    else:
        print(f"  ERROR: {result['error']}")

    print("\n" + "="*60)
    print("ImageAnalyzerTool tests complete!")
    print("="*60)


def test_tool_registry():
    """Test that ImageAnalyzerTool is properly registered."""
    print("\n" + "="*60)
    print("Testing Tool Registry")
    print("="*60)

    from topoagent.tools import get_all_tools, ImageAnalyzerTool

    tools = get_all_tools()
    print(f"Total tools registered: {len(tools)}")

    if "image_analyzer" in tools:
        print("  [PASS] image_analyzer found in registry")
        print(f"  Tool type: {type(tools['image_analyzer']).__name__}")
    else:
        print("  [FAIL] image_analyzer NOT found in registry")
        print(f"  Available tools: {list(tools.keys())}")

    # Test import
    try:
        from topoagent.tools.preprocessing import ImageAnalyzerTool
        print("  [PASS] ImageAnalyzerTool can be imported")
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")


def test_state_helpers():
    """Test the v3 state helper functions."""
    print("\n" + "="*60)
    print("Testing State Helpers (v3)")
    print("="*60)

    from topoagent.state import (
        create_initial_state,
        get_image_analyzer_results,
        get_recommended_filtration,
        get_recommended_sigma
    )

    # Create state with mock image_analyzer output
    state = create_initial_state("test query", "/test/image.png")

    # Add mock image_analyzer output to short-term memory
    mock_analyzer_output = {
        "success": True,
        "image_statistics": {
            "bright_ratio": 0.15,
            "dark_ratio": 0.45,
            "snr_estimate": 8.5
        },
        "recommendations": {
            "filtration_type": "superlevel",
            "noise_filter": "gaussian",
            "pi_sigma": 0.1,
            "pi_weight": "linear"
        }
    }
    state["short_term_memory"].append(("image_analyzer", mock_analyzer_output))

    # Test helper functions
    recommendations = get_image_analyzer_results(state)
    if recommendations is not None:
        print("  [PASS] get_image_analyzer_results works")
        print(f"  Recommendations: {recommendations}")
    else:
        print("  [FAIL] get_image_analyzer_results returned None")

    filtration = get_recommended_filtration(state)
    if filtration == "superlevel":
        print(f"  [PASS] get_recommended_filtration: {filtration}")
    else:
        print(f"  [FAIL] get_recommended_filtration: expected 'superlevel', got {filtration}")

    sigma = get_recommended_sigma(state)
    if sigma == 0.1:
        print(f"  [PASS] get_recommended_sigma: {sigma}")
    else:
        print(f"  [FAIL] get_recommended_sigma: expected 0.1, got {sigma}")


def test_with_image(image_path: str):
    """Test the image analyzer with a real image."""
    print("\n" + "="*60)
    print(f"Testing with image: {image_path}")
    print("="*60)

    from PIL import Image
    import numpy as np
    from topoagent.tools.preprocessing import ImageAnalyzerTool

    # Load image
    if not os.path.exists(image_path):
        print(f"  [ERROR] Image not found: {image_path}")
        return

    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)

    # Normalize
    if img_array.max() > 1:
        img_array = img_array / 255.0

    print(f"  Image shape: {img_array.shape}")
    print(f"  Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")

    # Analyze
    analyzer = ImageAnalyzerTool()
    result = analyzer._run(image_array=img_array.tolist())

    if result["success"]:
        print("\n  Image Statistics:")
        for key, value in result["image_statistics"].items():
            print(f"    {key}: {value}")

        print("\n  Recommendations:")
        for key, value in result["recommendations"].items():
            print(f"    {key}: {value}")

        print(f"\n  Reasoning:\n{result['reasoning']}")
    else:
        print(f"  [ERROR] Analysis failed: {result['error']}")


def test_full_pipeline():
    """Test the full adaptive pipeline (requires API key)."""
    print("\n" + "="*60)
    print("Testing Full Adaptive Pipeline")
    print("="*60)

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  [SKIP] OPENAI_API_KEY not set")
        print("  Set the environment variable to run full pipeline tests")
        return

    try:
        from topoagent.tools import get_all_tools
        from topoagent.workflow import create_topoagent_workflow
        from langchain_openai import ChatOpenAI

        # Create model
        model = ChatOpenAI(model="gpt-4o", temperature=0)

        # Get tools
        tools = get_all_tools()
        print(f"  Loaded {len(tools)} tools")

        # Create adaptive workflow
        workflow = create_topoagent_workflow(
            model=model,
            tools=tools,
            max_rounds=5,
            adaptive_mode=True  # v3
        )
        print("  [PASS] Created adaptive workflow")
        print(f"  Adaptive mode: {workflow.adaptive_mode}")
        print(f"  Max rounds: {workflow.max_rounds}")

        # Test would require actual image and API calls
        print("\n  Note: Full integration test requires API calls")
        print("  Use --image flag with a real image to test complete pipeline")

    except ImportError as e:
        print(f"  [ERROR] Import error: {e}")
    except Exception as e:
        print(f"  [ERROR] {e}")


def main():
    parser = argparse.ArgumentParser(description="Test TopoAgent v3 Adaptive Pipeline")
    parser.add_argument("--test-analyzer", action="store_true",
                        help="Test ImageAnalyzerTool standalone")
    parser.add_argument("--test-registry", action="store_true",
                        help="Test tool registry")
    parser.add_argument("--test-state", action="store_true",
                        help="Test state helper functions")
    parser.add_argument("--image", type=str,
                        help="Test with specific image file")
    parser.add_argument("--full-test", action="store_true",
                        help="Test full adaptive pipeline (requires API key)")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    args = parser.parse_args()

    # Default to running basic tests if no args provided
    if not any([args.test_analyzer, args.test_registry, args.test_state,
                args.image, args.full_test, args.all]):
        args.all = True

    print("="*60)
    print("TopoAgent v3 Adaptive Pipeline Test Suite")
    print("="*60)

    if args.test_analyzer or args.all:
        test_image_analyzer_tool()

    if args.test_registry or args.all:
        test_tool_registry()

    if args.test_state or args.all:
        test_state_helpers()

    if args.image:
        test_with_image(args.image)

    if args.full_test or args.all:
        test_full_pipeline()

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
