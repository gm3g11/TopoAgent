#!/usr/bin/env python3
"""Direct TopoAgent evaluation with timeout handling.

This script evaluates TopoAgent using a stream-based approach to avoid
workflow hangs. It processes the workflow step-by-step and extracts
the final classification from the pipeline output.

Usage:
    python scripts/evaluate_topoagent_direct.py --n-samples 20 --model gpt-4o-mini
"""

import os
import sys
import json
import time
import argparse
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def evaluate_single_sample_with_timeout(
    agent,
    image_path: str,
    query: str,
    timeout_seconds: int = 120
) -> Dict[str, Any]:
    """Evaluate a single sample with timeout protection.

    Args:
        agent: TopoAgent instance
        image_path: Path to image
        query: Classification query
        timeout_seconds: Max time per sample

    Returns:
        Classification result or error dict
    """
    start_time = time.time()

    try:
        with time_limit(timeout_seconds):
            result = agent.classify(image_path=image_path, query=query)
            return {
                "success": True,
                "classification": result.get("classification", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "tools_used": result.get("tools_used", []),
                "rounds_used": result.get("rounds_used", 0),
                "time": time.time() - start_time
            }
    except TimeoutException:
        return {
            "success": False,
            "classification": "timeout",
            "confidence": 0.0,
            "error": f"Timed out after {timeout_seconds}s",
            "time": time.time() - start_time
        }
    except Exception as e:
        return {
            "success": False,
            "classification": "error",
            "confidence": 0.0,
            "error": str(e),
            "time": time.time() - start_time
        }


def evaluate_topoagent_direct(
    dataset: str = "dermamnist",
    n_samples: int = 20,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_rounds: int = 4,
    timeout_per_sample: int = 90,
    seed: int = 42,
    disable_reflection: bool = True,  # Disable for speed
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate TopoAgent with direct timeout handling.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        model_name: OpenAI model name
        temperature: LLM temperature
        max_rounds: Max reasoning rounds
        timeout_per_sample: Timeout in seconds per sample
        seed: Random seed
        disable_reflection: Disable reflection for faster execution
        verbose: Print progress

    Returns:
        Evaluation results
    """
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"TopoAgent Direct Evaluation")
        print(f"{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"Samples: {n_samples}")
        print(f"Model: {model_name}")
        print(f"Reflection: {not disable_reflection}")
        print(f"Timeout: {timeout_per_sample}s per sample")

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

    # Create agent with reduced tools for faster execution
    from topoagent.agent import TopoAgent
    from langchain_openai import ChatOpenAI

    # Create model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        request_timeout=60,  # 60 second timeout per LLM call
    )

    # Create agent with only essential tools
    from topoagent.tools.preprocessing import ImageLoaderTool
    from topoagent.tools.homology import ComputePHTool, PersistenceImageTool
    from topoagent.tools.classification import PyTorchClassifierTool

    essential_tools = {
        "image_loader": ImageLoaderTool(),
        "compute_ph": ComputePHTool(),
        "persistence_image": PersistenceImageTool(),
        "pytorch_classifier": PyTorchClassifierTool()
    }

    # Create agent with reflection disabled for speed
    from topoagent.workflow import TopoAgentWorkflow

    workflow = TopoAgentWorkflow(
        model=model,
        tools=essential_tools,
        max_rounds=max_rounds,
        enable_reflection=not disable_reflection,
        enable_short_memory=True,
        enable_long_memory=False  # Disable long memory for speed
    )

    # Create wrapper agent
    class DirectAgent:
        def __init__(self, workflow):
            self.workflow = workflow

        def classify(self, image_path: str, query: str) -> Dict[str, Any]:
            from topoagent.state import create_initial_state

            initial_state = create_initial_state(
                query=query,
                image_path=image_path,
                max_rounds=max_rounds
            )

            # Use stream to get results - collect all state updates
            collected_state = {}
            for chunk in self.workflow.workflow.stream(initial_state):
                # Each chunk updates part of the state
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict):
                        collected_state.update(node_output)

            # Extract prediction from short_term_memory (classifier output)
            prediction = "unknown"
            confidence = 0.0
            tools_used = []
            rounds_used = 0

            short_term_memory = collected_state.get("short_term_memory", [])
            tools_used = [t for t, _ in short_term_memory]
            rounds_used = collected_state.get("current_round", 1) - 1

            # Look for pytorch_classifier output in memory
            for tool_name, output in short_term_memory:
                if tool_name == "pytorch_classifier" and isinstance(output, dict):
                    if output.get("success"):
                        prediction = output.get("predicted_class", "unknown")
                        confidence = output.get("confidence", 0.0)
                        # Convert numeric prediction to class index
                        if isinstance(prediction, str) and prediction.isdigit():
                            prediction = int(prediction)
                        break

            # If no classifier output, try to parse from final_answer
            if prediction == "unknown":
                final_answer = collected_state.get("final_answer", "")
                if final_answer:
                    # Try to extract class number
                    import re
                    match = re.search(r'class[:\s]*(\d+)', final_answer.lower())
                    if match:
                        prediction = int(match.group(1))
                    # Or try extracting from "predicted class: X"
                    match2 = re.search(r'predicted.*?(\d+)', final_answer.lower())
                    if match2 and prediction == "unknown":
                        prediction = int(match2.group(1))

            return {
                "classification": prediction,
                "confidence": float(confidence) if confidence else 0.0,
                "tools_used": tools_used,
                "rounds_used": rounds_used
            }

    agent = DirectAgent(workflow)

    # Sample indices
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)

    # Create temp directory
    temp_dir = Path("temp_topoagent_eval")
    temp_dir.mkdir(exist_ok=True)

    # Evaluate
    predictions = []
    ground_truths = []
    confidences = []
    timings = []
    errors = []
    tool_sequences = []

    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        label = int(label.squeeze())

        # Save image
        img_path = temp_dir / f"sample_{idx}.png"
        if hasattr(img, 'save'):
            img.save(img_path)
        else:
            from PIL import Image
            Image.fromarray(np.array(img).squeeze()).save(img_path)

        # Classify with timeout
        result = evaluate_single_sample_with_timeout(
            agent,
            str(img_path),
            f"Classify this {dataset.replace('mnist', '')} image",
            timeout_per_sample
        )

        # Extract prediction
        if result["success"]:
            pred = result["classification"]
            conf = result.get("confidence", 0.0)
        else:
            pred = "error"
            conf = 0.0
            errors.append({
                "sample_idx": int(idx),
                "error": result.get("error", "unknown")
            })

        predictions.append(pred)
        ground_truths.append(label)
        confidences.append(conf)
        timings.append(result["time"])
        tool_sequences.append(result.get("tools_used", []))

        if verbose:
            status = "OK" if result["success"] else "ERR"
            print(f"  [{i+1}/{n_samples}] Sample {idx}: {status} ({result['time']:.1f}s)")

    # Cleanup
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()

    # Compute metrics
    from scripts.evaluate import compute_comprehensive_metrics

    metrics = compute_comprehensive_metrics(
        predictions, ground_truths, info['label'], n_classes
    )

    total_time = time.time() - start_time

    results = {
        "method": "topoagent_v2_direct",
        "dataset": dataset,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "model": model_name,
        "reflection_enabled": not disable_reflection,
        "max_rounds": max_rounds,
        "seed": seed,
        "metrics": metrics,
        "timing": {
            "total_seconds": total_time,
            "avg_per_sample": np.mean(timings) if timings else 0.0,
            "min_per_sample": np.min(timings) if timings else 0.0,
            "max_per_sample": np.max(timings) if timings else 0.0
        },
        "confidence": {
            "avg": np.mean(confidences) if confidences else 0.0,
            "min": np.min(confidences) if confidences else 0.0,
            "max": np.max(confidences) if confidences else 0.0
        },
        "tool_usage": {},
        "errors": errors,
        "n_errors": len(errors),
        "n_timeouts": sum(1 for e in errors if "timeout" in e.get("error", "").lower()),
        "timestamp": datetime.now().isoformat()
    }

    # Count tool usage
    for seq in tool_sequences:
        for tool in seq:
            results["tool_usage"][tool] = results["tool_usage"].get(tool, 0) + 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results Summary")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Macro F1: {metrics['macro_f1']*100:.2f}%")
        print(f"Errors: {len(errors)} ({len(errors)/n_samples*100:.1f}%)")
        print(f"Total time: {total_time:.1f}s")
        print(f"Avg time/sample: {np.mean(timings):.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="TopoAgent Direct Evaluation")
    parser.add_argument("--dataset", "-d", type=str, default="dermamnist")
    parser.add_argument("--n-samples", "-n", type=int, default=20)
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--with-reflection", action="store_true")
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    results = evaluate_topoagent_direct(
        dataset=args.dataset,
        n_samples=args.n_samples,
        model_name=args.model,
        temperature=args.temperature,
        max_rounds=args.max_rounds,
        timeout_per_sample=args.timeout,
        seed=args.seed,
        disable_reflection=not args.with_reflection
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
