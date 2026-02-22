#!/usr/bin/env python
"""Evaluate TopoAgent on MedMNIST datasets.

Enhanced evaluation script with comprehensive metrics including:
- Per-class accuracy, precision, recall, F1
- Confusion matrix generation
- Rate limit handling with exponential backoff
- Consistent JSON output format

Usage:
    # Using OpenAI (requires API key)
    python scripts/evaluate.py --dataset dermamnist --n-samples 100

    # Using Ollama (free local LLM)
    python scripts/evaluate.py --dataset dermamnist --n-samples 100 --ollama

    # Run ablation study
    python scripts/evaluate.py --dataset dermamnist --n-samples 100 --ablation --ollama

    # Compare with fixed baselines
    python scripts/evaluate.py --dataset dermamnist --n-samples 100 --compare-baselines --ollama
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
from collections import Counter, defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# Rate limit configuration
# With 30K TPM limit and ~5K tokens/request, need ~10s between requests
RATE_LIMIT_CONFIG = {
    "min_delay_ms": 12000,  # 12 second delay between requests (safe for 30K TPM)
    "max_retries": 5,  # Maximum retries on rate limit
    "backoff_base_s": 20,  # Exponential backoff base (20s, 40s, 80s)
}


# Ablation configurations following EndoAgent methodology
ABLATION_CONFIGS = {
    "full_v2": {
        "description": "Full TopoAgent v2 (reflection + memory + 4 rounds + PyTorch classifier)",
        "enable_reflection": True,
        "enable_memory": True,
        "max_rounds": 4,
        "use_trained_classifier": True
    },
    "no_reflection": {
        "description": "No reflection mechanism",
        "enable_reflection": False,
        "enable_memory": True,
        "max_rounds": 1,  # Without reflection, single round
        "use_trained_classifier": True
    },
    "no_memory": {
        "description": "No memory (clear between rounds)",
        "enable_reflection": True,
        "enable_memory": False,
        "max_rounds": 4,
        "use_trained_classifier": True
    },
    "no_classifier": {
        "description": "No trained classifier (LLM heuristics)",
        "enable_reflection": True,
        "enable_memory": True,
        "max_rounds": 3,  # No classifier round
        "use_trained_classifier": False
    },
    "single_round": {
        "description": "Single round only",
        "enable_reflection": True,
        "enable_memory": True,
        "max_rounds": 1,
        "use_trained_classifier": True
    },
}


def compute_comprehensive_metrics(
    predictions: List[Any],
    ground_truths: List[int],
    label_names: Dict[str, str],
    n_classes: int
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        predictions: Predicted labels (can be class indices or names)
        ground_truths: True labels (as class indices)
        label_names: Mapping of label indices to names
        n_classes: Number of classes

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - per_class_accuracy: Accuracy per class
            - precision: Per-class and macro precision
            - recall: Per-class and macro recall
            - f1: Per-class and macro F1
            - confusion_matrix: NxN confusion matrix
            - support: Number of samples per class
    """
    # Match predictions to label indices
    matched_preds = []
    for pred in predictions:
        pred_val = pred
        if isinstance(pred, str):
            pred_lower = pred.lower()
            matched = False

            # Try exact match first
            for idx, name in label_names.items():
                if name.lower() == pred_lower:
                    matched_preds.append(int(idx))
                    matched = True
                    break

            # Try partial match
            if not matched:
                for idx, name in label_names.items():
                    if name.lower() in pred_lower or pred_lower in name.lower():
                        matched_preds.append(int(idx))
                        matched = True
                        break

            # Try to extract class number
            if not matched:
                import re
                match = re.search(r'class[_\s]*(\d+)', pred_lower)
                if match:
                    class_idx = int(match.group(1))
                    if 0 <= class_idx < n_classes:
                        matched_preds.append(class_idx)
                        matched = True

            if not matched:
                matched_preds.append(-1)  # Unknown
        else:
            # Already numeric
            try:
                idx = int(pred)
                if 0 <= idx < n_classes:
                    matched_preds.append(idx)
                else:
                    matched_preds.append(-1)
            except (ValueError, TypeError):
                matched_preds.append(-1)

    # Build confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for pred, gt in zip(matched_preds, ground_truths):
        if pred >= 0 and gt >= 0:
            confusion_matrix[gt, pred] += 1

    # Compute per-class metrics
    per_class_metrics = {}
    total_correct = 0
    total_samples = 0

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    valid_classes = 0

    for cls_idx in range(n_classes):
        # True positives, false positives, false negatives
        tp = confusion_matrix[cls_idx, cls_idx]
        fp = confusion_matrix[:, cls_idx].sum() - tp
        fn = confusion_matrix[cls_idx, :].sum() - tp
        support = confusion_matrix[cls_idx, :].sum()

        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / support if support > 0 else 0.0

        cls_name = label_names.get(str(cls_idx), f"class_{cls_idx}")
        per_class_metrics[cls_name] = {
            "class_idx": cls_idx,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }

        total_correct += tp
        total_samples += support

        if support > 0:
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            valid_classes += 1

    # Compute overall metrics
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    unknown_count = sum(1 for p in matched_preds if p == -1)
    unknown_rate = unknown_count / len(matched_preds) if matched_preds else 0.0

    # Macro averages
    macro_precision = macro_precision / valid_classes if valid_classes > 0 else 0.0
    macro_recall = macro_recall / valid_classes if valid_classes > 0 else 0.0
    macro_f1 = macro_f1 / valid_classes if valid_classes > 0 else 0.0

    return {
        "accuracy": float(overall_accuracy),
        "correct": int(total_correct),
        "total": int(total_samples),
        "unknown_rate": float(unknown_rate),
        "unknown_count": int(unknown_count),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": [label_names.get(str(i), f"class_{i}") for i in range(n_classes)]
    }


def with_rate_limit(func):
    """Decorator for rate limit handling with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(RATE_LIMIT_CONFIG["max_retries"] + 1):
            try:
                # Add minimum delay between requests
                time.sleep(RATE_LIMIT_CONFIG["min_delay_ms"] / 1000)
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "limit" in error_str:
                    if attempt < RATE_LIMIT_CONFIG["max_retries"]:
                        wait_time = RATE_LIMIT_CONFIG["backoff_base_s"] * (2 ** attempt)
                        print(f"    Rate limit hit, waiting {wait_time}s before retry {attempt + 1}...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return None
    return wrapper


def evaluate_topoagent(
    dataset: str,
    n_samples: int = 100,
    model_name: str = "gpt-4o",
    max_rounds: int = 4,
    disable_reflection: bool = False,
    use_ollama: bool = False,
    ollama_model: str = "llama3.1:8b",
    seed: int = 42,
    temperature: float = 0.1,
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate TopoAgent on a MedMNIST dataset.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples to evaluate
        model_name: LLM model name (for OpenAI)
        max_rounds: Max reasoning rounds
        disable_reflection: Ablation - disable reflection
        use_ollama: Use Ollama local LLM instead of OpenAI
        ollama_model: Ollama model name (default: llama3.1:8b)
        seed: Random seed for reproducibility
        temperature: LLM temperature (default: 0.1 for consistency)
        verbose: Print progress updates

    Returns:
        Evaluation results with comprehensive metrics
    """
    start_time = time.time()

    if use_ollama:
        from topoagent import create_topoagent_ollama
        model_display = f"Ollama/{ollama_model}"
    else:
        from topoagent import create_topoagent
        model_display = model_name

    if verbose:
        print(f"\n=== Evaluating TopoAgent on {dataset} ===")
        print(f"Samples: {n_samples}, Model: {model_display}, Reflection: {not disable_reflection}")

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
        print("Warning: MedMNIST not installed. Using mock data.")
        return mock_evaluation(dataset, n_samples)

    # Create agent
    if use_ollama:
        agent = create_topoagent_ollama(
            model_name=ollama_model,
            max_rounds=1 if disable_reflection else max_rounds,
            temperature=temperature
        )
    else:
        agent = create_topoagent(
            model_name=model_name,
            max_rounds=1 if disable_reflection else max_rounds,
            temperature=temperature
        )

    # Sample indices
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)

    # Evaluate
    predictions = []
    ground_truths = []
    rounds_used = []
    tool_sequences = []
    confidences = []
    timings = []
    errors = []

    temp_dir = Path("temp_eval")
    temp_dir.mkdir(exist_ok=True)

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

        # Run agent with rate limiting
        sample_start = time.time()
        try:
            @with_rate_limit
            def run_classification():
                return agent.classify(
                    image_path=str(img_path),
                    query=f"Classify this {dataset.replace('mnist', '')} image"
                )

            result = run_classification()

            # Parse prediction
            pred_class = result.get('classification', 'unknown')
            confidence = result.get('confidence', 0.0)

            predictions.append(pred_class)
            ground_truths.append(label)
            rounds_used.append(result.get('rounds_used', 0))
            tool_sequences.append(result.get('tools_used', []))
            confidences.append(confidence)
            timings.append(time.time() - sample_start)

        except Exception as e:
            if verbose:
                print(f"  Error on sample {idx}: {e}")
            predictions.append('error')
            ground_truths.append(label)
            rounds_used.append(0)
            tool_sequences.append([])
            confidences.append(0.0)
            timings.append(time.time() - sample_start)
            errors.append({"sample_idx": int(idx), "error": str(e)})

        # Progress
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n_samples - i - 1)
            print(f"  Progress: {i + 1}/{n_samples} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(
        predictions, ground_truths, info['label'], n_classes
    )

    total_time = time.time() - start_time

    results = {
        "dataset": dataset,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "model": ollama_model if use_ollama else model_name,
        "backend": "ollama" if use_ollama else "openai",
        "max_rounds": max_rounds,
        "reflection_enabled": not disable_reflection,
        "temperature": temperature,
        "seed": seed,
        "metrics": metrics,
        "timing": {
            "total_seconds": float(total_time),
            "avg_per_sample": float(np.mean(timings)) if timings else 0.0,
            "min_per_sample": float(np.min(timings)) if timings else 0.0,
            "max_per_sample": float(np.max(timings)) if timings else 0.0,
        },
        "rounds": {
            "avg": float(np.mean(rounds_used)) if rounds_used else 0.0,
            "min": int(np.min(rounds_used)) if rounds_used else 0,
            "max": int(np.max(rounds_used)) if rounds_used else 0,
            "distribution": dict(Counter(rounds_used))
        },
        "tool_usage": dict(Counter([t for seq in tool_sequences for t in seq])),
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

    return results


def mock_evaluation(dataset: str, n_samples: int) -> Dict[str, Any]:
    """Mock evaluation when MedMNIST is not available.

    Args:
        dataset: Dataset name
        n_samples: Number of samples

    Returns:
        Mock results
    """
    return {
        "dataset": dataset,
        "n_samples": n_samples,
        "metrics": {
            "accuracy": 0.0,
            "correct": 0,
            "total": n_samples,
            "note": "MedMNIST not installed - mock results"
        },
        "timestamp": datetime.now().isoformat()
    }


def print_metrics_summary(results: Dict[str, Any]) -> None:
    """Print a summary of evaluation metrics.

    Args:
        results: Evaluation results dictionary
    """
    metrics = results.get("metrics", {})

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nDataset: {results.get('dataset', 'N/A')}")
    print(f"Samples: {results.get('n_samples', 'N/A')}")
    print(f"Model: {results.get('model', 'N/A')} ({results.get('backend', 'N/A')})")

    print(f"\n--- Overall Metrics ---")
    print(f"Accuracy:        {metrics.get('accuracy', 0)*100:.2f}%")
    print(f"Macro Precision: {metrics.get('macro_precision', 0)*100:.2f}%")
    print(f"Macro Recall:    {metrics.get('macro_recall', 0)*100:.2f}%")
    print(f"Macro F1:        {metrics.get('macro_f1', 0)*100:.2f}%")
    print(f"Unknown Rate:    {metrics.get('unknown_rate', 0)*100:.2f}%")

    print(f"\n--- Per-Class Performance ---")
    per_class = metrics.get("per_class", {})
    if per_class:
        print(f"{'Class':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
        print("-" * 60)
        for cls_name, cls_metrics in per_class.items():
            print(f"{cls_name[:20]:<20} "
                  f"{cls_metrics['accuracy']*100:>7.1f}% "
                  f"{cls_metrics['precision']*100:>7.1f}% "
                  f"{cls_metrics['recall']*100:>7.1f}% "
                  f"{cls_metrics['f1']*100:>7.1f}% "
                  f"{cls_metrics['support']:>8}")

    print(f"\n--- Timing ---")
    timing = results.get("timing", {})
    print(f"Total time:     {timing.get('total_seconds', 0):.1f}s")
    print(f"Avg per sample: {timing.get('avg_per_sample', 0):.2f}s")

    print(f"\n--- Rounds ---")
    rounds = results.get("rounds", {})
    print(f"Average rounds: {rounds.get('avg', 0):.2f}")

    if results.get("n_errors", 0) > 0:
        print(f"\n--- Errors ---")
        print(f"Total errors: {results.get('n_errors', 0)}")


def print_confusion_matrix(metrics: Dict[str, Any]) -> None:
    """Print confusion matrix in a readable format.

    Args:
        metrics: Metrics dictionary containing confusion_matrix
    """
    cm = np.array(metrics.get("confusion_matrix", []))
    class_names = metrics.get("class_names", [])

    if cm.size == 0:
        return

    print("\n--- Confusion Matrix ---")

    # Header
    header = "True\\Pred"
    for name in class_names:
        header += f" {name[:6]:>6}"
    print(header)
    print("-" * len(header))

    # Rows
    for i, name in enumerate(class_names):
        row = f"{name[:8]:<8}"
        for j in range(len(class_names)):
            row += f" {cm[i, j]:>6}"
        print(row)


def print_comparison_table(results1: Dict, results2: Dict, name1: str = "Full", name2: str = "Ablated"):
    """Print comparison table between two evaluation results.

    Args:
        results1: First evaluation results
        results2: Second evaluation results
        name1: Name for first configuration
        name2: Name for second configuration
    """
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    m1 = results1.get("metrics", {})
    m2 = results2.get("metrics", {})

    print(f"\n{'Metric':<25} {name1:<15} {name2:<15} {'Diff':<10}")
    print("-" * 65)

    metrics_to_compare = [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Unknown Rate", "unknown_rate"),
    ]

    for label, key in metrics_to_compare:
        v1 = m1.get(key, 0) * 100
        v2 = m2.get(key, 0) * 100
        diff = v1 - v2
        sign = "+" if diff > 0 else ""
        print(f"{label:<25} {v1:>6.1f}%{'':<8} {v2:>6.1f}%{'':<8} {sign}{diff:.1f}%")

    # Timing comparison
    t1 = results1.get("timing", {}).get("avg_per_sample", 0)
    t2 = results2.get("timing", {}).get("avg_per_sample", 0)
    print(f"{'Avg Time/Sample':<25} {t1:>6.2f}s{'':<8} {t2:>6.2f}s")

    # Rounds comparison
    r1 = results1.get("rounds", {}).get("avg", 0)
    r2 = results2.get("rounds", {}).get("avg", 0)
    print(f"{'Avg Rounds':<25} {r1:>6.1f}{'':<9} {r2:>6.1f}")


def evaluate_with_baselines(
    dataset: str,
    n_samples: int = 100,
    use_ollama: bool = False,
    ollama_model: str = "llama3.1:8b",
    model_name: str = "gpt-4o",
    seed: int = 42,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """Evaluate TopoAgent alongside fixed pipeline baselines.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        use_ollama: Use Ollama local LLM
        ollama_model: Ollama model name
        model_name: OpenAI model name
        seed: Random seed
        temperature: LLM temperature

    Returns:
        Combined results for TopoAgent and baselines
    """
    from baselines.fixed_pipelines import get_all_pipelines
    from baselines.run_baselines import run_dataset_evaluation

    print("\n=== Comparing TopoAgent with Fixed Baselines ===")

    # Run TopoAgent
    print("\n--- TopoAgent v2 ---")
    topoagent_results = evaluate_topoagent(
        dataset=dataset,
        n_samples=n_samples,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        model_name=model_name,
        seed=seed,
        temperature=temperature
    )

    # Run fixed baselines
    print("\n--- Fixed Baselines ---")
    baseline_results = run_dataset_evaluation(
        dataset=dataset,
        n_samples=n_samples,
        seed=seed
    )

    # Combine results
    combined = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "n_samples": n_samples,
        "seed": seed,
        "topoagent": topoagent_results,
        "baselines": baseline_results.get("pipelines", {})
    }

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON: TopoAgent v2 vs Fixed Baselines")
    print("=" * 80)

    print(f"\n{'Method':<25} {'Accuracy':>10} {'Macro F1':>10} {'Avg Time':>10}")
    print("-" * 55)

    # TopoAgent
    ta_acc = topoagent_results.get("metrics", {}).get("accuracy", 0)
    ta_f1 = topoagent_results.get("metrics", {}).get("macro_f1", 0)
    ta_time = topoagent_results.get("timing", {}).get("avg_per_sample", 0)
    print(f"{'TopoAgent v2':<25} {ta_acc*100:>9.1f}% {ta_f1*100:>9.1f}% {ta_time:>9.2f}s")

    # Baselines
    for name, data in combined["baselines"].items():
        acc = data.get("metrics", {}).get("accuracy", 0)
        f1 = data.get("metrics", {}).get("macro_f1", 0) if "macro_f1" in data.get("metrics", {}) else 0
        avg_time = data.get("avg_time", 0)
        print(f"{name:<25} {acc*100:>9.1f}% {f1*100:>9.1f}% {avg_time:>9.2f}s")

    # Highlight winner
    all_accs = {"TopoAgent v2": ta_acc}
    for name, data in combined["baselines"].items():
        all_accs[name] = data.get("metrics", {}).get("accuracy", 0)

    winner = max(all_accs.keys(), key=lambda k: all_accs[k])
    print(f"\nBest method: {winner} ({all_accs[winner]*100:.1f}%)")

    if winner == "TopoAgent v2" and len(all_accs) > 1:
        best_baseline = max(
            [k for k in all_accs.keys() if k != "TopoAgent v2"],
            key=lambda k: all_accs[k]
        )
        improvement = (ta_acc - all_accs[best_baseline]) * 100
        print(f"TopoAgent improvement over best baseline: {improvement:+.1f}%")

    return combined


def run_ablation_study(
    dataset: str,
    n_samples: int = 100,
    use_ollama: bool = False,
    ollama_model: str = "llama3.1:8b",
    model_name: str = "gpt-4o",
    seed: int = 42,
    temperature: float = 0.1,
    configs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run comprehensive ablation study.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        use_ollama: Use Ollama local LLM
        ollama_model: Ollama model name
        model_name: OpenAI model name
        seed: Random seed
        temperature: LLM temperature
        configs: List of config names to run (default: all)

    Returns:
        Ablation study results
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "n_samples": n_samples,
        "seed": seed,
        "configs": {}
    }

    for config_name in configs:
        if config_name not in ABLATION_CONFIGS:
            print(f"Warning: Unknown config '{config_name}', skipping")
            continue

        config = ABLATION_CONFIGS[config_name]
        print(f"\n--- Running: {config_name} ---")
        print(f"    {config['description']}")

        try:
            config_results = evaluate_topoagent(
                dataset=dataset,
                n_samples=n_samples,
                model_name=model_name,
                max_rounds=config.get("max_rounds", 4),
                disable_reflection=not config.get("enable_reflection", True),
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                seed=seed,
                temperature=temperature
            )
            results["configs"][config_name] = {
                "description": config["description"],
                "settings": config,
                "results": config_results
            }
        except Exception as e:
            print(f"    Error running {config_name}: {e}")
            results["configs"][config_name] = {
                "description": config["description"],
                "settings": config,
                "error": str(e)
            }

    # Print comparison table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Configuration':<20} {'Accuracy':>10} {'F1':>10} {'Rounds':>8} {'Time':>10}")
    print("-" * 60)

    baseline_acc = None
    for config_name, data in results["configs"].items():
        if "error" in data:
            print(f"{config_name:<20} {'ERROR':<10}")
            continue

        r = data.get("results", {})
        acc = r.get("metrics", {}).get("accuracy", 0)
        f1 = r.get("metrics", {}).get("macro_f1", 0)
        avg_rounds = r.get("rounds", {}).get("avg", 0)
        avg_time = r.get("timing", {}).get("avg_per_sample", 0)

        if config_name == "full_v2":
            baseline_acc = acc

        diff_str = ""
        if baseline_acc is not None and config_name != "full_v2":
            diff = (acc - baseline_acc) * 100
            diff_str = f" ({diff:+.1f}%)"

        print(f"{config_name:<20} {acc*100:>9.1f}%{diff_str:<8} {f1*100:>9.1f}% {avg_rounds:>7.1f} {avg_time:>9.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TopoAgent")

    parser.add_argument("--dataset", "-d", type=str, default="dermamnist",
                       choices=["dermamnist", "pathmnist", "bloodmnist",
                                "retinamnist", "pneumoniamnist"],
                       help="MedMNIST dataset")
    parser.add_argument("--n-samples", "-n", type=int, default=100,
                       help="Number of samples")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o",
                       help="OpenAI LLM model")
    parser.add_argument("--max-rounds", "-r", type=int, default=4,
                       help="Max rounds")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--temperature", "-t", type=float, default=0.1,
                       help="LLM temperature (default: 0.1 for consistency)")

    # Ollama support
    parser.add_argument("--ollama", action="store_true",
                       help="Use Ollama local LLM instead of OpenAI (free)")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b",
                       help="Ollama model name (default: llama3.1:8b)")

    # Evaluation modes
    parser.add_argument("--benchmark", "-b", type=str,
                       help="TopoQA benchmark JSON file")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    parser.add_argument("--ablation-configs", type=str, nargs="+",
                       help="Specific ablation configs to run")
    parser.add_argument("--compare-baselines", action="store_true",
                       help="Compare with fixed TDA pipeline baselines")

    # Output options
    parser.add_argument("--output", "-o", type=str,
                       help="Output JSON file")
    parser.add_argument("--show-confusion-matrix", action="store_true",
                       help="Print confusion matrix")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")

    args = parser.parse_args()

    if args.benchmark:
        # Evaluate on TopoQA benchmark
        print("Benchmark evaluation not yet implemented")
        return

    if args.compare_baselines:
        # Compare TopoAgent with fixed baselines
        results = evaluate_with_baselines(
            dataset=args.dataset,
            n_samples=args.n_samples,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            model_name=args.model,
            seed=args.seed,
            temperature=args.temperature
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.ablation:
        # Run ablation study
        results = run_ablation_study(
            dataset=args.dataset,
            n_samples=args.n_samples,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            model_name=args.model,
            seed=args.seed,
            temperature=args.temperature,
            configs=args.ablation_configs
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    else:
        # Single evaluation
        results = evaluate_topoagent(
            dataset=args.dataset,
            n_samples=args.n_samples,
            model_name=args.model,
            max_rounds=args.max_rounds,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            seed=args.seed,
            temperature=args.temperature,
            verbose=not args.quiet
        )

        if not args.quiet:
            print_metrics_summary(results)
            if args.show_confusion_matrix:
                print_confusion_matrix(results.get("metrics", {}))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
