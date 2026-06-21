#!/usr/bin/env python
"""Direct PI+MLP Classification Baseline (No Agent).

This baseline computes persistence images and applies the trained MLP classifier
directly without any agent orchestration. This isolates the TDA+classifier performance
from the LLM agent overhead.

Key Insight: If TopoAgent v2 and PI+MLP Direct have similar accuracy, it shows
the agent correctly orchestrates the pipeline. If PI+MLP Direct is better,
the agent adds unnecessary overhead.

Usage:
    # Single image
    python baselines/pi_mlp_direct.py --image path/to/image.png

    # Dataset evaluation
    python baselines/pi_mlp_direct.py --dataset dermamnist --n-samples 100
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PIMlpDirectClassifier:
    """Direct Persistence Image + MLP classification without agent."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        pi_resolution: Tuple[int, int] = (20, 20),
        pi_sigma: float = 0.1,
        pi_weight: str = "linear"
    ):
        """Initialize direct PI+MLP classifier.

        Args:
            model_path: Path to trained PyTorch model
            pi_resolution: Resolution for persistence images (H, W)
            pi_sigma: Sigma for Gaussian smoothing in PI
            pi_weight: Weight function for PI ("linear", "persistence")
        """
        self.model_path = model_path or self._find_model()
        self.pi_resolution = pi_resolution
        self.pi_sigma = pi_sigma
        self.pi_weight = pi_weight

        # Load trained classifier
        self.classifier = self._load_classifier()

        # PI computation cache
        self._check_gudhi()

    def _find_model(self) -> str:
        """Find trained model path."""
        candidates = [
            "models/dermamnist_pi_mlp.pt",
            "models/dermamnist_pi_mlp.pth",
            "models/pi_mlp_classifier.pt",
            Path(__file__).parent.parent / "models" / "dermamnist_pi_mlp.pt"
        ]

        for path in candidates:
            if Path(path).exists():
                return str(path)

        raise FileNotFoundError(
            "Trained model not found. Run: python scripts/train_classifier.py --output models/"
        )

    def _load_classifier(self):
        """Load trained PyTorch classifier."""
        try:
            import torch
            import torch.nn as nn

            # Define MLP architecture (must match training script)
            class PIMlp(nn.Module):
                """Architecture matching DermaMNIST_MLP in train_classifier.py"""
                def __init__(self, input_dim=800, num_classes=7, dropout=0.3):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, num_classes)
                    )

                def forward(self, x):
                    return self.layers(x)

            # Load model - the training script saves just state_dict
            state_dict = torch.load(self.model_path, map_location='cpu')

            # Infer dimensions from state dict
            # First layer weight shape is (256, input_dim)
            # Last layer weight shape is (num_classes, 64)
            input_dim = state_dict['layers.0.weight'].shape[1]
            num_classes = state_dict['layers.9.weight'].shape[0]

            model = PIMlp(input_dim=input_dim, num_classes=num_classes)
            model.load_state_dict(state_dict)
            model.eval()

            self.n_classes = num_classes
            self.input_dim = input_dim

            return model

        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load classifier: {e}")

    def _check_gudhi(self):
        """Check if GUDHI is available."""
        try:
            import gudhi
            self.has_gudhi = True
        except ImportError:
            self.has_gudhi = False
            print("Warning: GUDHI not installed. Using fallback PI computation.")

    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image.

        Args:
            image_path: Path to image

        Returns:
            Preprocessed image array (normalized grayscale)
        """
        from PIL import Image

        img = Image.open(image_path)
        arr = np.array(img)

        # Convert to grayscale if color
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)

        # Normalize to [0, 1]
        if arr.max() > 1:
            arr = arr / 255.0

        return arr.astype(np.float64)

    def compute_persistence(self, image: np.ndarray) -> Dict[str, List]:
        """Compute persistent homology using cubical complex.

        Args:
            image: Grayscale image array

        Returns:
            Dictionary with H0 and H1 persistence pairs
        """
        if self.has_gudhi:
            return self._compute_persistence_gudhi(image)
        else:
            return self._compute_persistence_fallback(image)

    def _compute_persistence_gudhi(self, image: np.ndarray) -> Dict[str, List]:
        """Compute persistence using GUDHI."""
        import gudhi

        cc = gudhi.CubicalComplex(
            dimensions=image.shape,
            top_dimensional_cells=image.flatten()
        )
        cc.compute_persistence()

        result = {"H0": [], "H1": []}

        for dim, (birth, death) in cc.persistence():
            if death == float('inf'):
                death = float(np.max(image))

            entry = {"birth": float(birth), "death": float(death)}

            if dim == 0:
                result["H0"].append(entry)
            elif dim == 1:
                result["H1"].append(entry)

        return result

    def _compute_persistence_fallback(self, image: np.ndarray) -> Dict[str, List]:
        """Fallback persistence computation without GUDHI."""
        # Simple approximation using thresholding
        h0 = [{"birth": 0.0, "death": float(np.max(image))}]
        h1 = []

        # Estimate holes from binary image at multiple thresholds
        thresholds = np.linspace(0.2, 0.8, 5)
        for t in thresholds:
            binary = image > t
            # Count potential holes (simplified)
            holes = np.sum(~binary) / (image.shape[0] * image.shape[1])
            if holes > 0.1 and holes < 0.9:
                h1.append({"birth": float(t - 0.1), "death": float(t + 0.1)})

        return {"H0": h0, "H1": h1}

    def compute_persistence_image(
        self,
        persistence_data: Dict[str, List]
    ) -> np.ndarray:
        """Compute persistence images from persistence data.

        Args:
            persistence_data: Dictionary with H0, H1 persistence pairs

        Returns:
            Flattened feature vector (concatenated H0 and H1 PIs)
        """
        h0_pi = self._compute_single_pi(persistence_data.get("H0", []))
        h1_pi = self._compute_single_pi(persistence_data.get("H1", []))

        # Concatenate and flatten
        return np.concatenate([h0_pi.flatten(), h1_pi.flatten()])

    def _compute_single_pi(self, pairs: List[Dict]) -> np.ndarray:
        """Compute persistence image for one dimension.

        Args:
            pairs: List of {birth, death} dictionaries

        Returns:
            2D persistence image array
        """
        resolution = self.pi_resolution
        sigma = self.pi_sigma

        # Initialize image
        pi = np.zeros(resolution)

        if not pairs:
            return pi

        # Convert to birth-persistence coordinates
        points = []
        for pair in pairs:
            b = pair["birth"]
            d = pair["death"]
            persistence = d - b
            if persistence > 1e-6:  # Filter noise
                points.append((b, persistence))

        if not points:
            return pi

        # Determine grid range
        births = [p[0] for p in points]
        perss = [p[1] for p in points]

        birth_min, birth_max = min(births), max(births)
        pers_min, pers_max = min(perss), max(perss)

        # Add padding
        birth_range = max(birth_max - birth_min, 0.1)
        pers_range = max(pers_max - pers_min, 0.1)

        birth_min -= 0.1 * birth_range
        birth_max += 0.1 * birth_range
        pers_max += 0.1 * pers_range

        # Create grid
        birth_grid = np.linspace(birth_min, birth_max, resolution[0])
        pers_grid = np.linspace(0, pers_max, resolution[1])

        # Compute persistence image with Gaussian kernels
        for b, p in points:
            # Weight function
            if self.pi_weight == "linear":
                weight = p
            elif self.pi_weight == "persistence":
                weight = p ** 2
            else:
                weight = 1.0

            # Add Gaussian at (b, p)
            for i, bg in enumerate(birth_grid):
                for j, pg in enumerate(pers_grid):
                    dist_sq = ((bg - b) ** 2 + (pg - p) ** 2) / (2 * sigma ** 2)
                    pi[i, j] += weight * np.exp(-dist_sq)

        # Normalize
        if pi.max() > 0:
            pi = pi / pi.max()

        return pi

    def classify(self, image_path: str) -> Dict[str, Any]:
        """Classify a single image.

        Args:
            image_path: Path to image

        Returns:
            Classification result
        """
        import torch

        start_time = time.time()

        try:
            # Step 1: Load image
            image = self.load_image(image_path)

            # Step 2: Compute persistence
            persistence_data = self.compute_persistence(image)

            # Step 3: Compute persistence image
            features = self.compute_persistence_image(persistence_data)

            # Step 4: Classify
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                logits = self.classifier(x)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            return {
                "success": True,
                "prediction": int(pred_class),
                "confidence": float(confidence),
                "probabilities": probs[0].tolist(),
                "feature_dim": len(features),
                "persistence_stats": {
                    "n_h0": len(persistence_data.get("H0", [])),
                    "n_h1": len(persistence_data.get("H1", []))
                },
                "time": time.time() - start_time
            }

        except Exception as e:
            return {
                "success": False,
                "prediction": -1,
                "error": str(e),
                "time": time.time() - start_time
            }


def evaluate_pi_mlp_direct(
    dataset: str,
    n_samples: int = 100,
    seed: int = 42,
    model_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate direct PI+MLP classification on a dataset.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        seed: Random seed
        model_path: Path to trained model
        verbose: Print progress

    Returns:
        Evaluation results
    """
    start_time = time.time()

    if verbose:
        print(f"\n=== Direct PI+MLP Classification on {dataset} ===")
        print(f"Samples: {n_samples}")

    # Load dataset
    try:
        import medmnist
        from medmnist import INFO

        info = INFO[dataset]
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        test_dataset = DataClass(split='test', download=True)
        if verbose:
            print(f"Dataset loaded: {len(test_dataset)} samples, {n_classes} classes")

    except ImportError:
        print("Error: MedMNIST not installed")
        return {"error": "MedMNIST not installed"}

    # Create classifier
    try:
        classifier = PIMlpDirectClassifier(model_path=model_path)
        if verbose:
            print(f"Loaded classifier from: {classifier.model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    # Sample indices
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)

    # Create temp directory
    temp_dir = Path("temp_pi_mlp_eval")
    temp_dir.mkdir(exist_ok=True)

    # Evaluate
    predictions = []
    ground_truths = []
    confidences = []
    timings = []
    errors = []

    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        label = int(label.squeeze())

        # Save image temporarily
        img_path = temp_dir / f"sample_{idx}.png"
        if hasattr(img, 'save'):
            img.save(img_path)
        else:
            from PIL import Image
            Image.fromarray(np.array(img).squeeze()).save(img_path)

        # Classify
        result = classifier.classify(str(img_path))

        predictions.append(result.get("prediction", -1))
        ground_truths.append(label)
        confidences.append(result.get("confidence", 0.0))
        timings.append(result.get("time", 0))

        if not result.get("success"):
            errors.append({"sample_idx": int(idx), "error": result.get("error", "unknown")})

        # Progress
        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n_samples - i - 1)
            print(f"  Progress: {i + 1}/{n_samples} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    # Compute metrics
    from scripts.evaluate import compute_comprehensive_metrics
    metrics = compute_comprehensive_metrics(
        predictions, ground_truths, info['label'], n_classes
    )

    total_time = time.time() - start_time

    results = {
        "method": "pi_mlp_direct",
        "dataset": dataset,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "model_path": classifier.model_path,
        "seed": seed,
        "metrics": metrics,
        "timing": {
            "total_seconds": float(total_time),
            "avg_per_sample": float(np.mean(timings)) if timings else 0.0,
            "min_per_sample": float(np.min(timings)) if timings else 0.0,
            "max_per_sample": float(np.max(timings)) if timings else 0.0,
        },
        "confidence": {
            "avg": float(np.mean(confidences)) if confidences else 0.0,
            "min": float(np.min(confidences)) if confidences else 0.0,
            "max": float(np.max(confidences)) if confidences else 0.0,
        },
        "errors": errors,
        "n_errors": len(errors),
        "timestamp": datetime.now().isoformat()
    }

    # Cleanup
    for f in temp_dir.glob("*.png"):
        f.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    if verbose:
        print(f"\nResults:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1: {metrics['macro_f1']*100:.2f}%")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Avg time/sample: {np.mean(timings)*1000:.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Direct PI+MLP Classification Baseline")

    parser.add_argument("--image", "-i", type=str,
                       help="Single image path")
    parser.add_argument("--dataset", "-d", type=str, default="dermamnist",
                       choices=["dermamnist", "pathmnist", "bloodmnist",
                                "retinamnist", "pneumoniamnist"],
                       help="MedMNIST dataset")
    parser.add_argument("--n-samples", "-n", type=int, default=100,
                       help="Number of samples")
    parser.add_argument("--model", "-m", type=str,
                       help="Path to trained model")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", "-o", type=str,
                       help="Output JSON file")

    args = parser.parse_args()

    if args.image:
        # Single image classification
        classifier = PIMlpDirectClassifier(model_path=args.model)
        result = classifier.classify(args.image)
        print(f"Prediction: {result}")

    else:
        # Dataset evaluation
        results = evaluate_pi_mlp_direct(
            dataset=args.dataset,
            n_samples=args.n_samples,
            seed=args.seed,
            model_path=args.model
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
