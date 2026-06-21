#!/usr/bin/env python
"""Run fixed TDA pipeline baselines on datasets.

Usage:
    python baselines/run_baselines.py --dataset dermamnist --n-samples 100
    python baselines/run_baselines.py --image /path/to/image.png --pipeline pipeline_a
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from collections import Counter

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.fixed_pipelines import get_all_pipelines, FixedTDAPipeline


def run_single_image(
    image_path: str,
    pipeline_name: Optional[str] = None
) -> Dict[str, Any]:
    """Run pipeline(s) on a single image.

    Args:
        image_path: Path to image file
        pipeline_name: Specific pipeline to run (None = all)

    Returns:
        Results dictionary
    """
    pipelines = get_all_pipelines()

    if pipeline_name:
        if pipeline_name not in pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        pipelines = {pipeline_name: pipelines[pipeline_name]}

    results = {}
    for name, pipeline in pipelines.items():
        print(f"Running {name}...")
        result = pipeline.classify(image_path)
        results[name] = result
        print(f"  Prediction: {result.get('prediction')}")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
        print(f"  Time: {result.get('timing', {}).get('total', 0):.3f}s")

    return results


def run_dataset_evaluation(
    dataset: str,
    n_samples: int = 100,
    pipeline_name: Optional[str] = None,
    seed: int = 42,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Run pipeline(s) on a MedMNIST dataset.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples to evaluate
        pipeline_name: Specific pipeline (None = all)
        seed: Random seed for sampling
        output_file: Optional output JSON file

    Returns:
        Evaluation results
    """
    # Load dataset
    try:
        import medmnist
        from medmnist import INFO

        info = INFO[dataset]
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        test_dataset = DataClass(split='test', download=True)
        print(f"Dataset loaded: {len(test_dataset)} samples, {n_classes} classes")
        print(f"Classes: {info['label']}")

    except ImportError:
        print("Error: MedMNIST not installed. Install with: pip install medmnist")
        return {"error": "MedMNIST not installed"}

    # Get pipelines
    pipelines = get_all_pipelines()
    if pipeline_name:
        if pipeline_name not in pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        pipelines = {pipeline_name: pipelines[pipeline_name]}

    # Sample indices
    np.random.seed(seed)
    indices = np.random.choice(
        len(test_dataset),
        min(n_samples, len(test_dataset)),
        replace=False
    )

    # Create temp directory for images
    temp_dir = Path("temp_baseline_eval")
    temp_dir.mkdir(exist_ok=True)

    # Initialize results
    all_results = {
        name: {
            "predictions": [],
            "ground_truths": [],
            "confidences": [],
            "timings": [],
            "errors": 0
        }
        for name in pipelines.keys()
    }

    # Run evaluation
    print(f"\nEvaluating on {n_samples} samples...")

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

        # Run each pipeline
        for name, pipeline in pipelines.items():
            try:
                result = pipeline.classify(str(img_path))

                all_results[name]["predictions"].append(result.get("prediction", "error"))
                all_results[name]["ground_truths"].append(label)
                all_results[name]["confidences"].append(result.get("confidence", 0.0))
                all_results[name]["timings"].append(result.get("timing", {}).get("total", 0))

            except Exception as e:
                all_results[name]["predictions"].append("error")
                all_results[name]["ground_truths"].append(label)
                all_results[name]["confidences"].append(0.0)
                all_results[name]["timings"].append(0.0)
                all_results[name]["errors"] += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_samples}")

    # Compute metrics
    results = {
        "dataset": dataset,
        "n_samples": n_samples,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "pipelines": {}
    }

    for name, data in all_results.items():
        metrics = compute_metrics(
            data["predictions"],
            data["ground_truths"],
            info['label']
        )

        results["pipelines"][name] = {
            "metrics": metrics,
            "avg_confidence": float(np.mean(data["confidences"])),
            "avg_time": float(np.mean(data["timings"])),
            "total_time": float(np.sum(data["timings"])),
            "n_errors": data["errors"]
        }

    # Cleanup
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, data in results["pipelines"].items():
        m = data["metrics"]
        print(f"\n{name}:")
        print(f"  Accuracy: {m['accuracy']*100:.1f}%")
        print(f"  Unknown rate: {m['unknown_rate']*100:.1f}%")
        print(f"  Avg time: {data['avg_time']:.3f}s")
        print(f"  Errors: {data['n_errors']}")

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def compute_metrics(
    predictions: List[str],
    ground_truths: List[int],
    label_names: Dict[str, str]
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Predicted labels
        ground_truths: True labels
        label_names: Mapping of label indices to names

    Returns:
        Metrics dictionary
    """
    # Try to match predictions to label indices
    matched_preds = []
    for pred in predictions:
        pred_lower = str(pred).lower()
        matched = False

        for idx, name in label_names.items():
            if name.lower() in pred_lower or pred_lower in name.lower():
                matched_preds.append(int(idx))
                matched = True
                break

        if not matched:
            # Try to extract class number
            import re
            match = re.search(r'class_?(\d+)', pred_lower)
            if match:
                matched_preds.append(int(match.group(1)))
            else:
                matched_preds.append(-1)  # Unknown

    # Accuracy
    correct = sum(1 for p, g in zip(matched_preds, ground_truths) if p == g)
    accuracy = correct / len(ground_truths) if ground_truths else 0

    # Unknown rate
    unknown_rate = sum(1 for p in matched_preds if p == -1) / len(matched_preds) if matched_preds else 0

    # Per-class accuracy
    per_class = {}
    for idx in set(ground_truths):
        class_preds = [p for p, g in zip(matched_preds, ground_truths) if g == idx]
        if class_preds:
            per_class[str(idx)] = sum(1 for p in class_preds if p == idx) / len(class_preds)

    return {
        "accuracy": float(accuracy),
        "correct": int(correct),
        "total": len(ground_truths),
        "unknown_rate": float(unknown_rate),
        "per_class_accuracy": per_class
    }


def compare_pipelines(results: Dict[str, Any]) -> None:
    """Print comparison table of pipeline results.

    Args:
        results: Results from run_dataset_evaluation
    """
    if "pipelines" not in results:
        return

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    print(f"\n{'Pipeline':<20} {'Accuracy':<12} {'Unknown':<12} {'Avg Time':<12}")
    print("-" * 56)

    for name, data in results["pipelines"].items():
        m = data["metrics"]
        print(f"{name:<20} {m['accuracy']*100:>6.1f}%{'':<5} "
              f"{m['unknown_rate']*100:>6.1f}%{'':<5} "
              f"{data['avg_time']:>7.3f}s")

    # Find best
    best_name = max(
        results["pipelines"].keys(),
        key=lambda n: results["pipelines"][n]["metrics"]["accuracy"]
    )
    best_acc = results["pipelines"][best_name]["metrics"]["accuracy"]
    print(f"\nBest pipeline: {best_name} ({best_acc*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run fixed TDA pipeline baselines")

    parser.add_argument("--image", "-i", type=str,
                        help="Single image path to process")
    parser.add_argument("--dataset", "-d", type=str,
                        choices=["dermamnist", "pathmnist", "bloodmnist",
                                 "retinamnist", "pneumoniamnist"],
                        help="MedMNIST dataset to evaluate")
    parser.add_argument("--n-samples", "-n", type=int, default=100,
                        help="Number of samples for dataset evaluation")
    parser.add_argument("--pipeline", "-p", type=str,
                        choices=["pipeline_a", "pipeline_b", "pipeline_c"],
                        help="Specific pipeline to run (default: all)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", "-o", type=str,
                        help="Output JSON file")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available pipelines")

    args = parser.parse_args()

    if args.list:
        pipelines = get_all_pipelines()
        print("Available pipelines:")
        for name, p in pipelines.items():
            print(f"\n  {name}:")
            print(f"    Description: {p.description}")
            print(f"    Steps: {' -> '.join(p._get_pipeline_steps())}")
        return

    if args.image:
        # Single image mode
        results = run_single_image(args.image, args.pipeline)

    elif args.dataset:
        # Dataset evaluation mode
        results = run_dataset_evaluation(
            dataset=args.dataset,
            n_samples=args.n_samples,
            pipeline_name=args.pipeline,
            seed=args.seed,
            output_file=args.output
        )
        compare_pipelines(results)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
