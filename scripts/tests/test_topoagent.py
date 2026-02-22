#!/usr/bin/env python3
"""Comprehensive test suite for TopoAgent.

This script validates all core components of TopoAgent:
1. Tool loading and initialization
2. Individual TDA tools (image_loader, compute_ph, persistence_image, etc.)
3. End-to-end TDA pipeline
4. PyTorch classifier
5. Direct classification mode
6. Accuracy on sample images

Usage:
    python scripts/test_topoagent.py
    python scripts/test_topoagent.py --verbose
    python scripts/test_topoagent.py --test-accuracy --n-samples 10
"""

import argparse
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
import traceback

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResult:
    """Store test results."""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration

    def __str__(self):
        status = "\u2713 PASS" if self.passed else "\u2717 FAIL"
        time_str = f" ({self.duration:.2f}s)" if self.duration > 0 else ""
        msg = f" - {self.message}" if self.message else ""
        return f"  {status} {self.name}{time_str}{msg}"


class TopoAgentTester:
    """Test suite for TopoAgent."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, msg: str):
        """Print if verbose mode enabled."""
        if self.verbose:
            print(f"    {msg}")

    def run_test(self, name: str, test_func) -> TestResult:
        """Run a single test and record result."""
        start = time.time()
        try:
            result, message = test_func()
            duration = time.time() - start
            test_result = TestResult(name, result, message, duration)
        except Exception as e:
            duration = time.time() - start
            test_result = TestResult(name, False, f"Exception: {str(e)}", duration)
            if self.verbose:
                traceback.print_exc()

        self.results.append(test_result)
        print(test_result)
        return test_result

    # ==================== Tool Loading Tests ====================

    def test_tool_imports(self) -> Tuple[bool, str]:
        """Test that all tool modules can be imported."""
        from topoagent.tools import get_all_tools
        tools = get_all_tools()
        self.log(f"Loaded {len(tools)} tools")
        if len(tools) >= 15:
            return True, f"{len(tools)} tools loaded"
        return False, f"Only {len(tools)} tools (expected >= 15)"

    def test_tool_initialization(self) -> Tuple[bool, str]:
        """Test that tools can be instantiated."""
        try:
            from topoagent.tools.preprocessing import ImageLoaderTool
            from topoagent.tools.homology import ComputePHTool, PersistenceImageTool
            from topoagent.tools.features import TopologicalFeaturesTool
            from topoagent.tools.classification import PyTorchClassifierTool

            tools = [
                ImageLoaderTool(),
                ComputePHTool(),
                PersistenceImageTool(),
                TopologicalFeaturesTool(),
                PyTorchClassifierTool()
            ]
            self.log(f"Instantiated {len(tools)} core tools")
            return True, f"{len(tools)} tools instantiated"
        except Exception as e:
            return False, str(e)

    # ==================== Individual Tool Tests ====================

    def test_image_loader(self) -> Tuple[bool, str]:
        """Test image_loader tool with a sample image."""
        from topoagent.tools.preprocessing import ImageLoaderTool

        # Create a simple test image
        test_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            from PIL import Image
            img = Image.fromarray(test_array)
            img.save(f.name)
            temp_path = f.name

        try:
            tool = ImageLoaderTool()
            result = tool._run(image_path=temp_path)

            if not result.get("success", False):
                return False, result.get("error", "Unknown error")

            img_array = result.get("image_array")
            if img_array is None:
                return False, "No image_array in result"

            self.log(f"Loaded image shape: {np.array(img_array).shape}")
            return True, f"Shape {np.array(img_array).shape}"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_compute_ph(self) -> Tuple[bool, str]:
        """Test compute_ph tool with a sample image."""
        from topoagent.tools.homology import ComputePHTool

        # Create test image (simple gradient for predictable topology)
        test_array = np.zeros((28, 28), dtype=np.float32)
        for i in range(28):
            for j in range(28):
                test_array[i, j] = (i + j) / 56.0

        tool = ComputePHTool()
        result = tool._run(image_array=test_array.tolist())

        if not result.get("success", False):
            return False, result.get("error", "Unknown error")

        persistence = result.get("persistence", {})
        h0 = persistence.get("H0", [])
        h1 = persistence.get("H1", [])

        self.log(f"H0 features: {len(h0)}, H1 features: {len(h1)}")

        if len(h0) == 0:
            return False, "No H0 features computed"

        return True, f"H0: {len(h0)}, H1: {len(h1)}"

    def test_persistence_image(self) -> Tuple[bool, str]:
        """Test persistence_image tool."""
        from topoagent.tools.homology import PersistenceImageTool

        # Create sample persistence data (format expected by the tool)
        persistence_data = {
            "H0": [
                {"birth": 0.0, "death": 0.5, "persistence": 0.5},
                {"birth": 0.1, "death": 0.8, "persistence": 0.7},
                {"birth": 0.2, "death": 1.0, "persistence": 0.8}
            ],
            "H1": [
                {"birth": 0.3, "death": 0.6, "persistence": 0.3},
                {"birth": 0.4, "death": 0.9, "persistence": 0.5}
            ]
        }

        tool = PersistenceImageTool()
        result = tool._run(persistence_data=persistence_data)

        if not result.get("success", False):
            return False, result.get("error", "Unknown error")

        # Tool returns combined_vector (not feature_vector)
        feature_vector = result.get("combined_vector", [])
        self.log(f"Feature vector dim: {len(feature_vector)}")

        if len(feature_vector) != 800:
            return False, f"Expected 800D, got {len(feature_vector)}D"

        return True, f"{len(feature_vector)}D feature vector"

    def test_topological_features(self) -> Tuple[bool, str]:
        """Test topological_features tool."""
        from topoagent.tools.features import TopologicalFeaturesTool

        persistence_data = {
            "H0": [
                {"birth": 0.0, "death": 0.5, "persistence": 0.5},
                {"birth": 0.1, "death": 0.8, "persistence": 0.7}
            ],
            "H1": [
                {"birth": 0.3, "death": 0.6, "persistence": 0.3}
            ]
        }

        tool = TopologicalFeaturesTool()
        result = tool._run(persistence_data=persistence_data)

        if not result.get("success", False):
            return False, result.get("error", "Unknown error")

        num_features = result.get("num_features", 0)
        self.log(f"Extracted {num_features} statistical features")

        if num_features < 20:
            return False, f"Expected >= 20 features, got {num_features}"

        return True, f"{num_features} features extracted"

    # ==================== Pipeline Tests ====================

    def test_full_tda_pipeline(self) -> Tuple[bool, str]:
        """Test complete TDA pipeline: image -> PH -> PI -> features."""
        from topoagent.tools.preprocessing import ImageLoaderTool
        from topoagent.tools.homology import ComputePHTool, PersistenceImageTool

        # Create test image
        test_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            from PIL import Image
            img = Image.fromarray(test_array)
            img.save(f.name)
            temp_path = f.name

        try:
            # Step 1: Load image
            loader = ImageLoaderTool()
            img_result = loader._run(image_path=temp_path)
            if not img_result.get("success"):
                return False, f"Image loading failed: {img_result.get('error')}"

            # Step 2: Compute PH
            ph_tool = ComputePHTool()
            ph_result = ph_tool._run(image_array=img_result["image_array"])
            if not ph_result.get("success"):
                return False, f"PH computation failed: {ph_result.get('error')}"

            # Step 3: Generate PI
            pi_tool = PersistenceImageTool()
            persistence_data = ph_result.get("persistence", {})

            pi_result = pi_tool._run(persistence_data=persistence_data)
            if not pi_result.get("success"):
                return False, f"PI generation failed: {pi_result.get('error')}"

            feature_dim = len(pi_result.get("combined_vector", []))
            self.log(f"Pipeline output: {feature_dim}D feature vector")

            return True, f"Pipeline produced {feature_dim}D features"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    # ==================== Classifier Tests ====================

    def test_classifier_loading(self) -> Tuple[bool, str]:
        """Test that PyTorch classifier can be loaded."""
        from topoagent.tools.classification import PyTorchClassifierTool

        tool = PyTorchClassifierTool()

        # Check if model file exists
        model_path = Path("models/dermamnist_pi_mlp.pt")
        if not model_path.exists():
            # Try weighted version
            model_path = Path("models/dermamnist_pi_mlp_weighted.pt")

        if not model_path.exists():
            return False, "No trained model found in models/"

        self.log(f"Found model: {model_path}")
        return True, f"Model found: {model_path.name}"

    def test_classifier_inference(self) -> Tuple[bool, str]:
        """Test classifier inference with random features."""
        from topoagent.tools.classification import PyTorchClassifierTool

        # Check for model
        model_paths = [
            Path("models/dermamnist_pi_mlp.pt"),
            Path("models/dermamnist_pi_mlp_weighted.pt"),
            Path("models/dermamnist_pi_mlp_focal.pt")
        ]

        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = str(p)
                break

        if model_path is None:
            return False, "No trained model found"

        # Create random 800D feature vector
        feature_vector = np.random.randn(800).tolist()

        tool = PyTorchClassifierTool()
        result = tool._run(feature_vector=feature_vector, model_path=model_path)

        if not result.get("success", False):
            return False, result.get("error", "Unknown error")

        pred_class = result.get("predicted_class", "unknown")
        confidence = result.get("confidence", 0)

        self.log(f"Predicted: {pred_class} (confidence: {confidence:.1f}%)")

        return True, f"Predicted '{pred_class}' ({confidence:.1f}%)"

    # ==================== Direct Mode Tests ====================

    def test_direct_classification(self) -> Tuple[bool, str]:
        """Test direct classification mode (bypasses LLM)."""
        # Check for model first
        model_paths = [
            Path("models/dermamnist_pi_mlp.pt"),
            Path("models/dermamnist_pi_mlp_weighted.pt")
        ]

        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = str(p)
                break

        if model_path is None:
            return False, "No trained model found for direct mode"

        # Create test image
        test_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            from PIL import Image
            img = Image.fromarray(test_array)
            img.save(f.name)
            temp_path = f.name

        try:
            from topoagent.agent import TopoAgent

            # Create minimal agent (no LLM needed for direct mode)
            # We'll initialize tools manually
            from topoagent.tools.preprocessing import ImageLoaderTool
            from topoagent.tools.homology import ComputePHTool, PersistenceImageTool
            from topoagent.tools.classification import PyTorchClassifierTool

            tools = {
                "image_loader": ImageLoaderTool(),
                "compute_ph": ComputePHTool(),
                "persistence_image": PersistenceImageTool(),
                "pytorch_classifier": PyTorchClassifierTool()
            }

            # Create a mock model for the agent
            class MockModel:
                def bind_tools(self, tools):
                    return self

            agent = TopoAgent(model=MockModel(), tools=tools)

            # Run direct classification
            result = agent.classify_direct(temp_path, model_path=model_path)

            if not result.get("success", False):
                return False, result.get("error", "Unknown error")

            latency = result.get("latency_ms", 0)
            pred_class = result.get("predicted_class", "unknown")

            self.log(f"Direct mode: {pred_class} in {latency:.1f}ms")

            return True, f"{pred_class} in {latency:.1f}ms"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    # ==================== Accuracy Tests ====================

    def test_accuracy_on_samples(self, n_samples: int = 5) -> Tuple[bool, str]:
        """Test accuracy on real DermaMNIST samples."""
        try:
            from medmnist import DermaMNIST
        except ImportError:
            return False, "medmnist not installed"

        # Check for model
        model_paths = [
            Path("models/dermamnist_pi_mlp.pt"),
            Path("models/dermamnist_pi_mlp_weighted.pt")
        ]

        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = str(p)
                break

        if model_path is None:
            return False, "No trained model found"

        # Load test set
        dataset = DermaMNIST(split='test', download=True, size=28)

        # Initialize tools
        from topoagent.tools.preprocessing import ImageLoaderTool
        from topoagent.tools.homology import ComputePHTool, PersistenceImageTool
        from topoagent.tools.classification import PyTorchClassifierTool

        loader = ImageLoaderTool()
        ph_tool = ComputePHTool()
        pi_tool = PersistenceImageTool()
        classifier = PyTorchClassifierTool()

        correct = 0
        total = 0
        class_names = [
            "actinic keratosis", "basal cell carcinoma", "benign keratosis",
            "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"
        ]

        # Sample random indices
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

        for idx in indices:
            img, label = dataset[idx]
            true_label = int(label[0])

            # Convert PIL to numpy
            img_np = np.array(img)
            if len(img_np.shape) == 3:
                img_np = np.mean(img_np, axis=2)

            # Run pipeline
            ph_result = ph_tool._run(image_array=img_np.tolist())
            if not ph_result.get("success"):
                continue

            persistence_data = ph_result.get("persistence", {})

            pi_result = pi_tool._run(persistence_data=persistence_data)
            if not pi_result.get("success"):
                continue

            class_result = classifier._run(
                feature_vector=pi_result["combined_vector"],
                model_path=model_path
            )
            if not class_result.get("success"):
                continue

            pred_class_id = class_result.get("class_id", -1)
            total += 1

            if pred_class_id == true_label:
                correct += 1

            self.log(f"Sample {total}: true={class_names[true_label]}, pred={class_names[pred_class_id]}")

        if total == 0:
            return False, "No samples processed successfully"

        accuracy = 100.0 * correct / total
        self.log(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")

        # Consider test passed if accuracy > 20% (random = 14.3% for 7 classes)
        if accuracy > 20:
            return True, f"{accuracy:.1f}% accuracy ({correct}/{total})"
        else:
            return False, f"Low accuracy: {accuracy:.1f}%"

    # ==================== Test Runner ====================

    def run_all_tests(self, test_accuracy: bool = False, n_samples: int = 5):
        """Run all tests."""
        print("\n" + "=" * 60)
        print("TopoAgent Test Suite")
        print("=" * 60)

        # Tool Loading Tests
        print("\n[1/6] Tool Loading Tests")
        self.run_test("Import all tools", self.test_tool_imports)
        self.run_test("Initialize core tools", self.test_tool_initialization)

        # Individual Tool Tests
        print("\n[2/6] Individual Tool Tests")
        self.run_test("ImageLoaderTool", self.test_image_loader)
        self.run_test("ComputePHTool", self.test_compute_ph)
        self.run_test("PersistenceImageTool", self.test_persistence_image)
        self.run_test("TopologicalFeaturesTool", self.test_topological_features)

        # Pipeline Tests
        print("\n[3/6] Pipeline Tests")
        self.run_test("Full TDA Pipeline", self.test_full_tda_pipeline)

        # Classifier Tests
        print("\n[4/6] Classifier Tests")
        self.run_test("Classifier loading", self.test_classifier_loading)
        self.run_test("Classifier inference", self.test_classifier_inference)

        # Direct Mode Tests
        print("\n[5/6] Direct Classification Mode")
        self.run_test("Direct classification", self.test_direct_classification)

        # Accuracy Tests (optional)
        if test_accuracy:
            print(f"\n[6/6] Accuracy Tests (n={n_samples})")
            self.run_test(f"DermaMNIST accuracy ({n_samples} samples)",
                          lambda: self.test_accuracy_on_samples(n_samples))
        else:
            print("\n[6/6] Accuracy Tests (skipped, use --test-accuracy)")

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print("\n" + "=" * 60)
        print(f"Test Summary: {passed}/{total} passed, {failed} failed")
        print("=" * 60)

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")

        if failed == 0:
            print("\nAll tests passed! TopoAgent is working correctly.")
        else:
            print(f"\n{failed} test(s) failed. See details above.")

        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="TopoAgent Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--test-accuracy", action="store_true",
                        help="Run accuracy tests on real samples")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of samples for accuracy test")
    args = parser.parse_args()

    tester = TopoAgentTester(verbose=args.verbose)
    success = tester.run_all_tests(
        test_accuracy=args.test_accuracy,
        n_samples=args.n_samples
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
