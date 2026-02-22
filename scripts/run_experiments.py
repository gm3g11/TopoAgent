#!/usr/bin/env python
"""Master Experiment Runner for TopoAgent Evaluation.

This script orchestrates all experiments from the experimental plan:
- Experiment 1: Comprehensive Baseline Comparison
- Experiment 2: Ablation Study
- Experiment 3: Cross-Dataset Generalization
- Experiment 4: Error Analysis
- Experiment 5: Efficiency Analysis

Usage:
    # Run all experiments
    python scripts/run_experiments.py --all --dataset dermamnist --n-samples 100

    # Run specific experiment
    python scripts/run_experiments.py --exp1 --dataset dermamnist --n-samples 100

    # Run with Ollama (free local LLM)
    python scripts/run_experiments.py --all --ollama --dataset dermamnist --n-samples 100

    # Cross-dataset experiments
    python scripts/run_experiments.py --exp3 --n-samples 100 --datasets dermamnist pathmnist bloodmnist
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# Experiment configurations
EXPERIMENT_CONFIGS = {
    "exp1": {
        "name": "Comprehensive Baseline Comparison",
        "description": "Compare TopoAgent v2 against ALL relevant baselines",
        "methods": [
            "topoagent_v2",
            "gpt4v_direct",
            "pi_mlp_direct",
            "pipeline_a",
            "pipeline_b",
            "pipeline_c"
        ]
    },
    "exp2": {
        "name": "Ablation Study",
        "description": "Validate contribution of each v2 component",
        "configs": ["full_v2", "no_reflection", "no_memory", "no_classifier", "single_round"]
    },
    "exp3": {
        "name": "Cross-Dataset Generalization",
        "description": "Test generalization across medical imaging modalities",
        "datasets": ["dermamnist", "pathmnist", "bloodmnist", "retinamnist", "pneumoniamnist"]
    },
    "exp4": {
        "name": "Error Analysis",
        "description": "Understand failure modes and classify errors",
        "sample_errors": 50
    },
    "exp5": {
        "name": "Efficiency Analysis",
        "description": "Compare computational costs",
        "metrics": ["time_per_image", "llm_tokens", "tool_calls", "memory_usage"]
    }
}


def create_results_directory(base_dir: str = "results/experiments") -> Path:
    """Create results directory structure.

    Args:
        base_dir: Base directory for results

    Returns:
        Path to results directory
    """
    results_dir = Path(base_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (results_dir / "figures").mkdir(exist_ok=True)
    (results_dir / "tables").mkdir(exist_ok=True)
    (results_dir / "raw").mkdir(exist_ok=True)

    return results_dir


def run_experiment_1(
    dataset: str,
    n_samples: int,
    seed: int,
    use_ollama: bool,
    ollama_model: str,
    model_name: str,
    temperature: float,
    results_dir: Path,
    skip_gpt4v: bool = False,
    skip_pi_mlp: bool = False
) -> Dict[str, Any]:
    """Run Experiment 1: Comprehensive Baseline Comparison.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        seed: Random seed
        use_ollama: Use Ollama instead of OpenAI
        ollama_model: Ollama model name
        model_name: OpenAI model name
        temperature: LLM temperature
        results_dir: Directory to save results
        skip_gpt4v: Skip GPT-4V baseline (expensive)
        skip_pi_mlp: Skip PI+MLP direct baseline

    Returns:
        Experiment results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Comprehensive Baseline Comparison")
    print("=" * 80)
    print(f"Dataset: {dataset}, Samples: {n_samples}, Seed: {seed}")

    from scripts.evaluate import evaluate_topoagent, compute_comprehensive_metrics
    from baselines.run_baselines import run_dataset_evaluation

    results = {
        "experiment": "exp1_baseline_comparison",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": dataset,
            "n_samples": n_samples,
            "seed": seed,
            "backend": "ollama" if use_ollama else "openai",
            "model": ollama_model if use_ollama else model_name
        },
        "methods": {}
    }

    # 1. TopoAgent v2
    print("\n--- Running TopoAgent v2 ---")
    try:
        topoagent_results = evaluate_topoagent(
            dataset=dataset,
            n_samples=n_samples,
            model_name=model_name,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            seed=seed,
            temperature=temperature
        )
        results["methods"]["topoagent_v2"] = topoagent_results
        print(f"  Accuracy: {topoagent_results['metrics']['accuracy']*100:.2f}%")
    except Exception as e:
        print(f"  Error: {e}")
        results["methods"]["topoagent_v2"] = {"error": str(e)}

    # 2. GPT-4V Direct (if not skipped)
    if not skip_gpt4v and not use_ollama:
        print("\n--- Running GPT-4V Direct ---")
        try:
            from baselines.gpt4v_direct import evaluate_gpt4v_direct
            gpt4v_results = evaluate_gpt4v_direct(
                dataset=dataset,
                n_samples=n_samples,
                model_name=model_name,
                seed=seed,
                temperature=temperature
            )
            results["methods"]["gpt4v_direct"] = gpt4v_results
            print(f"  Accuracy: {gpt4v_results['metrics']['accuracy']*100:.2f}%")
        except Exception as e:
            print(f"  Error: {e}")
            results["methods"]["gpt4v_direct"] = {"error": str(e)}
    else:
        print("\n--- Skipping GPT-4V Direct (use --include-gpt4v to enable) ---")

    # 3. PI+MLP Direct (if not skipped)
    if not skip_pi_mlp:
        print("\n--- Running PI+MLP Direct ---")
        try:
            from baselines.pi_mlp_direct import evaluate_pi_mlp_direct
            pi_mlp_results = evaluate_pi_mlp_direct(
                dataset=dataset,
                n_samples=n_samples,
                seed=seed
            )
            results["methods"]["pi_mlp_direct"] = pi_mlp_results
            print(f"  Accuracy: {pi_mlp_results['metrics']['accuracy']*100:.2f}%")
        except Exception as e:
            print(f"  Error: {e}")
            results["methods"]["pi_mlp_direct"] = {"error": str(e)}
    else:
        print("\n--- Skipping PI+MLP Direct ---")

    # 4. Fixed TDA Pipelines
    print("\n--- Running Fixed TDA Pipelines ---")
    try:
        baseline_results = run_dataset_evaluation(
            dataset=dataset,
            n_samples=n_samples,
            seed=seed
        )
        for name, data in baseline_results.get("pipelines", {}).items():
            results["methods"][name] = data
            acc = data.get("metrics", {}).get("accuracy", 0)
            print(f"  {name}: {acc*100:.2f}%")
    except Exception as e:
        print(f"  Error: {e}")
        results["methods"]["fixed_pipelines"] = {"error": str(e)}

    # Print summary table
    print_experiment_summary(results)

    # Save results
    output_file = results_dir / "raw" / f"exp1_{dataset}_{n_samples}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def run_experiment_2(
    dataset: str,
    n_samples: int,
    seed: int,
    use_ollama: bool,
    ollama_model: str,
    model_name: str,
    temperature: float,
    results_dir: Path,
    configs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run Experiment 2: Ablation Study.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        seed: Random seed
        use_ollama: Use Ollama
        ollama_model: Ollama model name
        model_name: OpenAI model name
        temperature: LLM temperature
        results_dir: Directory to save results
        configs: Specific ablation configs to run

    Returns:
        Experiment results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Ablation Study")
    print("=" * 80)
    print(f"Dataset: {dataset}, Samples: {n_samples}, Seed: {seed}")

    from scripts.evaluate import run_ablation_study

    results = run_ablation_study(
        dataset=dataset,
        n_samples=n_samples,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        model_name=model_name,
        seed=seed,
        temperature=temperature,
        configs=configs
    )

    results["experiment"] = "exp2_ablation"

    # Save results
    output_file = results_dir / "raw" / f"exp2_ablation_{dataset}_{n_samples}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def run_experiment_3(
    datasets: List[str],
    n_samples: int,
    seed: int,
    use_ollama: bool,
    ollama_model: str,
    model_name: str,
    temperature: float,
    results_dir: Path
) -> Dict[str, Any]:
    """Run Experiment 3: Cross-Dataset Generalization.

    Args:
        datasets: List of MedMNIST dataset names
        n_samples: Number of samples per dataset
        seed: Random seed
        use_ollama: Use Ollama
        ollama_model: Ollama model name
        model_name: OpenAI model name
        temperature: LLM temperature
        results_dir: Directory to save results

    Returns:
        Experiment results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Cross-Dataset Generalization")
    print("=" * 80)
    print(f"Datasets: {datasets}, Samples per dataset: {n_samples}")

    from scripts.evaluate import evaluate_topoagent
    from baselines.run_baselines import run_dataset_evaluation

    results = {
        "experiment": "exp3_cross_dataset",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": datasets,
            "n_samples": n_samples,
            "seed": seed,
            "backend": "ollama" if use_ollama else "openai"
        },
        "datasets": {}
    }

    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---")
        dataset_results = {}

        # TopoAgent v2
        try:
            topoagent_results = evaluate_topoagent(
                dataset=dataset,
                n_samples=n_samples,
                model_name=model_name,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                seed=seed,
                temperature=temperature
            )
            dataset_results["topoagent_v2"] = {
                "accuracy": topoagent_results["metrics"]["accuracy"],
                "macro_f1": topoagent_results["metrics"]["macro_f1"],
                "avg_time": topoagent_results["timing"]["avg_per_sample"]
            }
            print(f"  TopoAgent v2: {topoagent_results['metrics']['accuracy']*100:.2f}%")
        except Exception as e:
            print(f"  TopoAgent error: {e}")
            dataset_results["topoagent_v2"] = {"error": str(e)}

        # Fixed pipelines
        try:
            baseline_results = run_dataset_evaluation(
                dataset=dataset,
                n_samples=n_samples,
                seed=seed
            )
            for name, data in baseline_results.get("pipelines", {}).items():
                acc = data.get("metrics", {}).get("accuracy", 0)
                dataset_results[name] = {
                    "accuracy": acc,
                    "avg_time": data.get("avg_time", 0)
                }
                print(f"  {name}: {acc*100:.2f}%")
        except Exception as e:
            print(f"  Baselines error: {e}")

        results["datasets"][dataset] = dataset_results

    # Print cross-dataset summary
    print_cross_dataset_summary(results)

    # Save results
    output_file = results_dir / "raw" / f"exp3_cross_dataset_{n_samples}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def run_experiment_4(
    dataset: str,
    n_samples: int,
    seed: int,
    use_ollama: bool,
    ollama_model: str,
    model_name: str,
    temperature: float,
    results_dir: Path,
    n_error_samples: int = 50
) -> Dict[str, Any]:
    """Run Experiment 4: Error Analysis.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples for evaluation
        seed: Random seed
        use_ollama: Use Ollama
        ollama_model: Ollama model name
        model_name: OpenAI model name
        temperature: LLM temperature
        results_dir: Directory to save results
        n_error_samples: Number of error samples to analyze

    Returns:
        Error analysis results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Error Analysis")
    print("=" * 80)
    print(f"Dataset: {dataset}, Samples: {n_samples}")

    from scripts.evaluate import evaluate_topoagent

    # Run evaluation and collect detailed results
    if use_ollama:
        from topoagent import create_topoagent_ollama
        agent = create_topoagent_ollama(
            model_name=ollama_model,
            temperature=temperature
        )
    else:
        from topoagent import create_topoagent
        agent = create_topoagent(
            model_name=model_name,
            temperature=temperature
        )

    # Load dataset
    try:
        import medmnist
        from medmnist import INFO

        info = INFO[dataset]
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        test_dataset = DataClass(split='test', download=True)
    except ImportError:
        return {"error": "MedMNIST not installed"}

    # Sample and evaluate
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)

    temp_dir = Path("temp_error_analysis")
    temp_dir.mkdir(exist_ok=True)

    # Collect detailed results
    detailed_results = []
    error_categories = {
        "tool_selection": [],
        "feature_extraction": [],
        "classification": [],
        "data_passing": [],
        "unknown": []
    }

    print("\nCollecting detailed results...")

    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        label = int(label.squeeze())

        img_path = temp_dir / f"sample_{idx}.png"
        if hasattr(img, 'save'):
            img.save(img_path)
        else:
            from PIL import Image
            Image.fromarray(np.array(img).squeeze()).save(img_path)

        try:
            result = agent.classify(
                image_path=str(img_path),
                query=f"Classify this {dataset.replace('mnist', '')} image"
            )

            is_correct = False  # Will determine based on prediction matching

            entry = {
                "sample_idx": int(idx),
                "ground_truth": label,
                "ground_truth_name": info['label'].get(str(label), f"class_{label}"),
                "prediction": result.get("classification", "unknown"),
                "confidence": result.get("confidence", 0),
                "tools_used": result.get("tools_used", []),
                "rounds_used": result.get("rounds_used", 0)
            }

            detailed_results.append(entry)

        except Exception as e:
            entry = {
                "sample_idx": int(idx),
                "ground_truth": label,
                "error": str(e),
                "error_type": "execution_error"
            }
            detailed_results.append(entry)
            error_categories["unknown"].append(entry)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{n_samples}")

        # Rate limiting
        time.sleep(0.3)

    # Analyze errors
    print("\nAnalyzing errors...")

    n_errors = 0
    for entry in detailed_results:
        if "error" in entry:
            n_errors += 1
            continue

        # Try to match prediction to class index
        pred_str = str(entry.get("prediction", "")).lower()
        pred_matched = False

        for idx, name in info['label'].items():
            if name.lower() in pred_str or pred_str in name.lower():
                entry["predicted_class"] = int(idx)
                entry["is_correct"] = int(idx) == entry["ground_truth"]
                pred_matched = True
                break

        if not pred_matched:
            entry["predicted_class"] = -1
            entry["is_correct"] = False

        if not entry.get("is_correct", False):
            n_errors += 1

            # Categorize error
            tools = entry.get("tools_used", [])
            if not tools:
                error_categories["tool_selection"].append(entry)
            elif "compute_ph" in tools and entry.get("rounds_used", 0) < 2:
                error_categories["feature_extraction"].append(entry)
            elif "pytorch_classifier" in tools:
                error_categories["classification"].append(entry)
            else:
                error_categories["unknown"].append(entry)

    # Summary
    results = {
        "experiment": "exp4_error_analysis",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": dataset,
            "n_samples": n_samples,
            "seed": seed
        },
        "summary": {
            "total_samples": n_samples,
            "total_errors": n_errors,
            "error_rate": n_errors / n_samples if n_samples > 0 else 0,
            "error_distribution": {
                category: len(errors)
                for category, errors in error_categories.items()
            }
        },
        "error_categories": {
            category: errors[:n_error_samples]
            for category, errors in error_categories.items()
        },
        "sample_results": detailed_results[:100]  # Store first 100 for analysis
    }

    # Print summary
    print(f"\n--- Error Analysis Summary ---")
    print(f"Total samples: {n_samples}")
    print(f"Total errors: {n_errors} ({n_errors/n_samples*100:.1f}%)")
    print(f"\nError distribution:")
    for category, errors in error_categories.items():
        if errors:
            print(f"  {category}: {len(errors)}")

    # Cleanup
    for f in temp_dir.glob("*.png"):
        f.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    # Save results
    output_file = results_dir / "raw" / f"exp4_error_analysis_{dataset}_{n_samples}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def run_experiment_5(
    dataset: str,
    n_samples: int,
    seed: int,
    use_ollama: bool,
    ollama_model: str,
    model_name: str,
    temperature: float,
    results_dir: Path
) -> Dict[str, Any]:
    """Run Experiment 5: Efficiency Analysis.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        seed: Random seed
        use_ollama: Use Ollama
        ollama_model: Ollama model name
        model_name: OpenAI model name
        temperature: LLM temperature
        results_dir: Directory to save results

    Returns:
        Efficiency analysis results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Efficiency Analysis")
    print("=" * 80)
    print(f"Dataset: {dataset}, Samples: {n_samples}")

    from scripts.evaluate import evaluate_topoagent
    from baselines.run_baselines import run_dataset_evaluation

    results = {
        "experiment": "exp5_efficiency",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": dataset,
            "n_samples": n_samples,
            "seed": seed
        },
        "methods": {}
    }

    # TopoAgent v2
    print("\n--- TopoAgent v2 Efficiency ---")
    topoagent_results = evaluate_topoagent(
        dataset=dataset,
        n_samples=n_samples,
        model_name=model_name,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        seed=seed,
        temperature=temperature
    )

    results["methods"]["topoagent_v2"] = {
        "timing": topoagent_results.get("timing", {}),
        "rounds": topoagent_results.get("rounds", {}),
        "tool_usage": topoagent_results.get("tool_usage", {}),
        "accuracy": topoagent_results["metrics"]["accuracy"]
    }

    avg_time = topoagent_results.get("timing", {}).get("avg_per_sample", 0)
    avg_rounds = topoagent_results.get("rounds", {}).get("avg", 0)
    tool_count = sum(topoagent_results.get("tool_usage", {}).values())
    print(f"  Avg time/sample: {avg_time:.2f}s")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"  Total tool calls: {tool_count}")

    # PI+MLP Direct (for comparison)
    print("\n--- PI+MLP Direct Efficiency ---")
    try:
        from baselines.pi_mlp_direct import evaluate_pi_mlp_direct
        pi_mlp_results = evaluate_pi_mlp_direct(
            dataset=dataset,
            n_samples=n_samples,
            seed=seed
        )
        results["methods"]["pi_mlp_direct"] = {
            "timing": pi_mlp_results.get("timing", {}),
            "accuracy": pi_mlp_results["metrics"]["accuracy"]
        }
        pi_time = pi_mlp_results.get("timing", {}).get("avg_per_sample", 0)
        print(f"  Avg time/sample: {pi_time*1000:.1f}ms")
    except Exception as e:
        print(f"  Error: {e}")

    # Fixed pipelines
    print("\n--- Fixed Pipeline Efficiency ---")
    baseline_results = run_dataset_evaluation(
        dataset=dataset,
        n_samples=n_samples,
        seed=seed
    )
    for name, data in baseline_results.get("pipelines", {}).items():
        results["methods"][name] = {
            "timing": {
                "avg_per_sample": data.get("avg_time", 0),
                "total_seconds": data.get("total_time", 0)
            },
            "accuracy": data.get("metrics", {}).get("accuracy", 0)
        }
        print(f"  {name}: {data.get('avg_time', 0)*1000:.1f}ms/sample")

    # Print efficiency comparison
    print("\n--- Efficiency Summary ---")
    print(f"{'Method':<20} {'Time/Sample':>15} {'Accuracy':>10}")
    print("-" * 47)
    for name, data in results["methods"].items():
        t = data.get("timing", {}).get("avg_per_sample", 0)
        acc = data.get("accuracy", 0)
        if t > 1:
            time_str = f"{t:.2f}s"
        else:
            time_str = f"{t*1000:.1f}ms"
        print(f"{name:<20} {time_str:>15} {acc*100:>9.1f}%")

    # Save results
    output_file = results_dir / "raw" / f"exp5_efficiency_{dataset}_{n_samples}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def print_experiment_summary(results: Dict[str, Any]) -> None:
    """Print summary table for experiment results.

    Args:
        results: Experiment results dictionary
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<25} {'Accuracy':>10} {'Macro F1':>10} {'Time':>10}")
    print("-" * 55)

    methods = results.get("methods", {})
    for name, data in methods.items():
        if "error" in data:
            print(f"{name:<25} {'ERROR':<10}")
            continue

        metrics = data.get("metrics", {})
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("macro_f1", 0)

        timing = data.get("timing", {})
        if isinstance(timing, dict):
            t = timing.get("avg_per_sample", data.get("avg_time", 0))
        else:
            t = data.get("avg_time", 0)

        if t > 1:
            time_str = f"{t:.2f}s"
        else:
            time_str = f"{t*1000:.0f}ms"

        print(f"{name:<25} {acc*100:>9.1f}% {f1*100:>9.1f}% {time_str:>10}")

    # Find best method
    best_method = None
    best_acc = 0
    for name, data in methods.items():
        if "error" not in data:
            acc = data.get("metrics", {}).get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_method = name

    if best_method:
        print(f"\nBest method: {best_method} ({best_acc*100:.1f}%)")


def print_cross_dataset_summary(results: Dict[str, Any]) -> None:
    """Print cross-dataset summary table.

    Args:
        results: Cross-dataset experiment results
    """
    print("\n" + "=" * 80)
    print("CROSS-DATASET SUMMARY")
    print("=" * 80)

    datasets = results.get("datasets", {})
    if not datasets:
        return

    # Get all methods
    all_methods = set()
    for ds_results in datasets.values():
        all_methods.update(ds_results.keys())

    # Header
    header = f"{'Dataset':<15}"
    for method in sorted(all_methods):
        header += f" {method[:12]:>12}"
    print(f"\n{header}")
    print("-" * len(header))

    # Rows
    for ds_name, ds_results in datasets.items():
        row = f"{ds_name:<15}"
        for method in sorted(all_methods):
            data = ds_results.get(method, {})
            if "error" in data:
                row += f" {'ERROR':>12}"
            else:
                acc = data.get("accuracy", 0)
                row += f" {acc*100:>11.1f}%"
        print(row)

    # Average row
    print("-" * len(header))
    avg_row = f"{'Average':<15}"
    for method in sorted(all_methods):
        accs = []
        for ds_results in datasets.values():
            data = ds_results.get(method, {})
            if "accuracy" in data:
                accs.append(data["accuracy"])
        if accs:
            avg_row += f" {np.mean(accs)*100:>11.1f}%"
        else:
            avg_row += f" {'N/A':>12}"
    print(avg_row)


def run_all_experiments(
    dataset: str,
    n_samples: int,
    seed: int,
    use_ollama: bool,
    ollama_model: str,
    model_name: str,
    temperature: float,
    results_dir: Path,
    datasets: Optional[List[str]] = None,
    skip_gpt4v: bool = True,
    skip_pi_mlp: bool = False
) -> Dict[str, Any]:
    """Run all experiments.

    Args:
        dataset: Primary MedMNIST dataset
        n_samples: Number of samples
        seed: Random seed
        use_ollama: Use Ollama
        ollama_model: Ollama model name
        model_name: OpenAI model name
        temperature: LLM temperature
        results_dir: Directory to save results
        datasets: Datasets for cross-dataset experiment
        skip_gpt4v: Skip GPT-4V baseline
        skip_pi_mlp: Skip PI+MLP baseline

    Returns:
        All experiment results
    """
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "primary_dataset": dataset,
            "n_samples": n_samples,
            "seed": seed,
            "backend": "ollama" if use_ollama else "openai"
        },
        "experiments": {}
    }

    # Experiment 1
    print("\n" + "#" * 80)
    print("# RUNNING EXPERIMENT 1")
    print("#" * 80)
    try:
        all_results["experiments"]["exp1"] = run_experiment_1(
            dataset=dataset,
            n_samples=n_samples,
            seed=seed,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            model_name=model_name,
            temperature=temperature,
            results_dir=results_dir,
            skip_gpt4v=skip_gpt4v,
            skip_pi_mlp=skip_pi_mlp
        )
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        all_results["experiments"]["exp1"] = {"error": str(e)}

    # Experiment 2
    print("\n" + "#" * 80)
    print("# RUNNING EXPERIMENT 2")
    print("#" * 80)
    try:
        all_results["experiments"]["exp2"] = run_experiment_2(
            dataset=dataset,
            n_samples=n_samples,
            seed=seed,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            model_name=model_name,
            temperature=temperature,
            results_dir=results_dir
        )
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        all_results["experiments"]["exp2"] = {"error": str(e)}

    # Experiment 3 (cross-dataset)
    if datasets and len(datasets) > 1:
        print("\n" + "#" * 80)
        print("# RUNNING EXPERIMENT 3")
        print("#" * 80)
        try:
            all_results["experiments"]["exp3"] = run_experiment_3(
                datasets=datasets,
                n_samples=n_samples,
                seed=seed,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                model_name=model_name,
                temperature=temperature,
                results_dir=results_dir
            )
        except Exception as e:
            print(f"Experiment 3 failed: {e}")
            all_results["experiments"]["exp3"] = {"error": str(e)}

    # Experiment 5 (efficiency - runs faster)
    print("\n" + "#" * 80)
    print("# RUNNING EXPERIMENT 5")
    print("#" * 80)
    try:
        all_results["experiments"]["exp5"] = run_experiment_5(
            dataset=dataset,
            n_samples=min(n_samples, 50),  # Smaller sample for efficiency
            seed=seed,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            model_name=model_name,
            temperature=temperature,
            results_dir=results_dir
        )
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
        all_results["experiments"]["exp5"] = {"error": str(e)}

    # Save combined results
    output_file = results_dir / f"all_experiments_{dataset}_{n_samples}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TopoAgent Master Experiment Runner")

    # Dataset options
    parser.add_argument("--dataset", "-d", type=str, default="dermamnist",
                       choices=["dermamnist", "pathmnist", "bloodmnist",
                                "retinamnist", "pneumoniamnist"],
                       help="Primary MedMNIST dataset")
    parser.add_argument("--datasets", type=str, nargs="+",
                       help="Datasets for cross-dataset experiments")
    parser.add_argument("--n-samples", "-n", type=int, default=100,
                       help="Number of samples")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed")

    # Model options
    parser.add_argument("--model", "-m", type=str, default="gpt-4o",
                       help="OpenAI model name")
    parser.add_argument("--temperature", "-t", type=float, default=0.1,
                       help="LLM temperature")
    parser.add_argument("--ollama", action="store_true",
                       help="Use Ollama instead of OpenAI")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b",
                       help="Ollama model name")

    # Experiment selection
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--exp1", action="store_true",
                       help="Run Experiment 1: Baseline Comparison")
    parser.add_argument("--exp2", action="store_true",
                       help="Run Experiment 2: Ablation Study")
    parser.add_argument("--exp3", action="store_true",
                       help="Run Experiment 3: Cross-Dataset")
    parser.add_argument("--exp4", action="store_true",
                       help="Run Experiment 4: Error Analysis")
    parser.add_argument("--exp5", action="store_true",
                       help="Run Experiment 5: Efficiency Analysis")

    # Options
    parser.add_argument("--include-gpt4v", action="store_true",
                       help="Include GPT-4V baseline (expensive)")
    parser.add_argument("--skip-pi-mlp", action="store_true",
                       help="Skip PI+MLP direct baseline")
    parser.add_argument("--ablation-configs", type=str, nargs="+",
                       help="Specific ablation configs for exp2")

    # Output
    parser.add_argument("--output-dir", "-o", type=str, default="results/experiments",
                       help="Output directory")

    args = parser.parse_args()

    # Create results directory
    results_dir = create_results_directory(args.output_dir)
    print(f"Results will be saved to: {results_dir}")

    # Determine which experiments to run
    run_any = args.all or args.exp1 or args.exp2 or args.exp3 or args.exp4 or args.exp5

    if not run_any:
        print("No experiment specified. Use --all or --expN flags.")
        parser.print_help()
        return

    if args.all:
        # Run all experiments
        run_all_experiments(
            dataset=args.dataset,
            n_samples=args.n_samples,
            seed=args.seed,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
            model_name=args.model,
            temperature=args.temperature,
            results_dir=results_dir,
            datasets=args.datasets or [args.dataset],
            skip_gpt4v=not args.include_gpt4v,
            skip_pi_mlp=args.skip_pi_mlp
        )
    else:
        # Run selected experiments
        if args.exp1:
            run_experiment_1(
                dataset=args.dataset,
                n_samples=args.n_samples,
                seed=args.seed,
                use_ollama=args.ollama,
                ollama_model=args.ollama_model,
                model_name=args.model,
                temperature=args.temperature,
                results_dir=results_dir,
                skip_gpt4v=not args.include_gpt4v,
                skip_pi_mlp=args.skip_pi_mlp
            )

        if args.exp2:
            run_experiment_2(
                dataset=args.dataset,
                n_samples=args.n_samples,
                seed=args.seed,
                use_ollama=args.ollama,
                ollama_model=args.ollama_model,
                model_name=args.model,
                temperature=args.temperature,
                results_dir=results_dir,
                configs=args.ablation_configs
            )

        if args.exp3:
            datasets = args.datasets or ["dermamnist", "pathmnist", "bloodmnist"]
            run_experiment_3(
                datasets=datasets,
                n_samples=args.n_samples,
                seed=args.seed,
                use_ollama=args.ollama,
                ollama_model=args.ollama_model,
                model_name=args.model,
                temperature=args.temperature,
                results_dir=results_dir
            )

        if args.exp4:
            run_experiment_4(
                dataset=args.dataset,
                n_samples=args.n_samples,
                seed=args.seed,
                use_ollama=args.ollama,
                ollama_model=args.ollama_model,
                model_name=args.model,
                temperature=args.temperature,
                results_dir=results_dir
            )

        if args.exp5:
            run_experiment_5(
                dataset=args.dataset,
                n_samples=args.n_samples,
                seed=args.seed,
                use_ollama=args.ollama,
                ollama_model=args.ollama_model,
                model_name=args.model,
                temperature=args.temperature,
                results_dir=results_dir
            )


if __name__ == "__main__":
    main()
