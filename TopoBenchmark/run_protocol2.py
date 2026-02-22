#!/usr/bin/env python3
"""Protocol 2: End-to-End TopoAgent Evaluation.

Phase A: Agent demonstration on sample images (verbose, per-image, ~3 images)
Phase B: Batch evaluation with 6 classifiers (efficient, per-dataset, n=1000-5000)

Usage:
    # Debug mode (small, verbose, CPU classifiers only)
    python TopoBenchmark/run_protocol2.py --dataset BloodMNIST --n-demo 2 --n-eval 100 --verbose

    # Full evaluation (3 datasets, 6 classifiers)
    python TopoBenchmark/run_protocol2.py --dataset BloodMNIST,Kvasir,DermaMNIST --n-eval 5000

    # All 26 datasets (cluster mode)
    python TopoBenchmark/run_protocol2.py --all --n-eval 5000

    # Benchmark-only (skip agent demo)
    python TopoBenchmark/run_protocol2.py --dataset BloodMNIST --no-demo --n-eval 1000

    # Compare multiple methods
    python TopoBenchmark/run_protocol2.py --dataset BloodMNIST --methods topoagent_v4,topoagent_v5 --n-eval 1000
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Constants
# =============================================================================

ALL_CLASSIFIERS = ["TabPFN", "XGBoost", "CatBoost", "RandomForest", "TabM", "RealMLP"]

# Methods supported in Protocol 2
# "topoagent_v4" — Full ReAct + Reflection loop (agentic)
# "topoagent_v5" — Reasoning-first deterministic (v5 workflow)
# "fixed_<descriptor>" — Fixed descriptor baselines
SUPPORTED_METHODS = ["topoagent_v4", "topoagent_v5"]

OUTPUT_DIR = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2"


# =============================================================================
# Helper: Save Sample Images
# =============================================================================

def save_sample_images(dataset_name: str, n_samples: int = 3) -> List[Tuple[str, int]]:
    """Load dataset and save sample images as PNG for agent processing.

    Returns:
        List of (image_path, label) tuples
    """
    output_dir = OUTPUT_DIR / "sample_images" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    b4_dir = PROJECT_ROOT / "RuleBenchmark" / "benchmark4"
    if str(b4_dir) not in sys.path:
        sys.path.insert(0, str(b4_dir))
    from data_loader import load_dataset

    images, labels, class_names = load_dataset(dataset_name, n_samples=n_samples, seed=42)

    from PIL import Image
    samples = []
    for i in range(min(n_samples, len(images))):
        img = images[i]
        label = int(labels[i])

        if img.ndim == 2:
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        elif img.ndim == 3 and img.shape[2] == 3:
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
        else:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))

        fname = f"sample_{i}_label{label}.png"
        fpath = output_dir / fname
        pil_img.save(fpath)
        samples.append((str(fpath), label))

    return samples


def build_query(dataset_name: str) -> str:
    """Build a natural query for the agent."""
    from TopoBenchmark.config import DATASET_DESCRIPTIONS
    desc = DATASET_DESCRIPTIONS[dataset_name]

    return (
        f"Select the best TDA descriptor for classifying images from the {dataset_name} dataset.\n\n"
        f"Domain: {desc['domain']}\n"
        f"Description: {desc['description']}\n"
        f"What matters: {desc['what_matters']}\n"
        f"Number of classes: {desc['n_classes']}\n"
        f"Object type: {desc['object_type']}\n"
        f"Color mode: {desc.get('color_mode', 'grayscale')}\n\n"
        f"Follow the pipeline: (1) image_loader, (2) compute_ph, (3) one descriptor tool.\n"
        f"Your goal is to select and execute the optimal descriptor — not to classify the image.\n"
    )


# =============================================================================
# Phase A: Agent Demonstration (per-image, verbose)
# =============================================================================

def run_agent_demo_v4(
    dataset_name: str,
    n_demo: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase A: Run v4 ReAct + Reflection agent on sample images with verbose output.

    Returns:
        Dict with descriptor_selected, reasoning_traces, llm_calls, elapsed_s
    """
    from topoagent.agent import create_topoagent
    from topoagent.tools.descriptors import get_all_descriptors
    from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS
    from topoagent.workflow import TopoAgentWorkflow

    # Create agent
    agent = create_topoagent(
        workflow_version="v4",
        skills_mode=True,
        temperature=0.3,
    )

    # Add all descriptor tools (default agent only has 16 basic tools)
    descriptor_tools = get_all_descriptors()
    for desc_name in SUPPORTED_DESCRIPTORS:
        if desc_name in descriptor_tools and desc_name not in agent.tools:
            agent.tools[desc_name] = descriptor_tools[desc_name]

    print(f"Agent tools: {len(agent.tools)} ({sorted(agent.tools.keys())})")

    # Rebuild workflow with verbose flag and full tool set
    agent.workflow = TopoAgentWorkflow(
        model=agent.model,
        tools=agent.tools,
        max_rounds=4,
        skills_mode=True,
        verbose=verbose,
    )

    # Save sample images
    samples = save_sample_images(dataset_name, n_samples=n_demo)
    query = build_query(dataset_name)

    results = []
    descriptor_votes = []

    for idx, (image_path, label) in enumerate(samples):
        image_name = Path(image_path).name
        print(f"\n{'╔' + '═' * 68 + '╗'}")
        print(f"║  {dataset_name} — Image {idx + 1}/{n_demo}: {image_name:<42} ║")
        print(f"║  True label: {label:<55} ║")
        print(f"{'╚' + '═' * 68 + '╝'}")

        start = time.time()
        try:
            final_state = agent.workflow.invoke(query=query, image_path=image_path)
            elapsed = time.time() - start
            success = True
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"success": False, "error": str(e), "elapsed": elapsed})
            continue

        # Extract descriptor from skill state
        descriptor = final_state.get("skill_descriptor")
        n_llm = len(final_state.get("llm_interactions", []))
        reasoning = final_state.get("reasoning_trace", [])

        # Fallback: parse descriptor from LLM tool calls if skill_descriptor not set
        if not descriptor:
            from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS as _SD
            for interaction in final_state.get("llm_interactions", []):
                if interaction.tool_calls:
                    for tc in interaction.tool_calls:
                        if tc["name"] in _SD:
                            descriptor = tc["name"]
                            break
                if descriptor:
                    break

        # Fallback: parse from final_answer text
        if not descriptor:
            answer = final_state.get("final_answer", "")
            if answer:
                from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS as _SD2
                for d in _SD2:
                    if d in answer.lower().replace(" ", "_"):
                        descriptor = d
                        break

        descriptor = descriptor or "unknown"

        descriptor_votes.append(descriptor)

        results.append({
            "image": image_name,
            "label": label,
            "descriptor": descriptor,
            "n_llm_calls": n_llm,
            "elapsed": elapsed,
            "success": success,
            "reasoning_trace": reasoning,
        })

        # Print summary for this image
        print(f"\n  {'─' * 60}")
        print(f"  RESULT: descriptor={descriptor}, llm_calls={n_llm}, elapsed={elapsed:.1f}s")

    # Determine consensus descriptor
    if descriptor_votes:
        from collections import Counter
        consensus = Counter(descriptor_votes).most_common(1)[0][0]
    else:
        consensus = "unknown"

    print(f"\n{'═' * 70}")
    print(f"  PHASE A SUMMARY ({dataset_name})")
    print(f"{'═' * 70}")
    print(f"  Descriptor votes: {descriptor_votes}")
    print(f"  Consensus: {consensus}")
    print(f"  Total LLM calls: {sum(r.get('n_llm_calls', 0) for r in results)}")
    total_elapsed = sum(r.get('elapsed', 0) for r in results)
    print(f"  Total time: {total_elapsed:.1f}s")

    return {
        "method": "topoagent_v4",
        "descriptor_selected": consensus,
        "descriptor_votes": descriptor_votes,
        "demo_results": results,
        "n_images": n_demo,
        "total_llm_calls": sum(r.get('n_llm_calls', 0) for r in results),
        "total_elapsed_s": total_elapsed,
    }


def run_agent_demo_v5(
    dataset_name: str,
    n_demo: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase A: Run v5 reasoning-first agent on sample images.

    Returns:
        Dict with descriptor_selected, reasoning_traces, etc.
    """
    from scripts.test_protocol2_small import create_v5_agent, build_query as build_v5_query

    agent = create_v5_agent(verify_mode="quick")
    samples = save_sample_images(dataset_name, n_samples=n_demo)
    query = build_v5_query(dataset_name)

    results = []
    descriptor_votes = []

    for idx, (image_path, label) in enumerate(samples):
        image_name = Path(image_path).name
        print(f"\n{'╔' + '═' * 68 + '╗'}")
        print(f"║  {dataset_name} — Image {idx + 1}/{n_demo}: {image_name:<42} ║")
        print(f"{'╚' + '═' * 68 + '╝'}")

        start = time.time()
        try:
            final_state = agent.workflow.invoke(
                query=query,
                image_path=image_path,
                verify_mode="quick",
                dataset_name=dataset_name,
            )
            elapsed = time.time() - start
            success = True
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  ERROR: {e}")
            results.append({"success": False, "error": str(e), "elapsed": elapsed})
            continue

        descriptor = final_state.get("skill_descriptor", "unknown")
        n_llm = len(final_state.get("llm_interactions", []))
        plan = final_state.get("plan", {})
        trace = final_state.get("execution_trace", [])

        descriptor_votes.append(descriptor)

        if verbose:
            print(f"\n  Plan: {json.dumps(plan, indent=2, default=str)[:500]}")
            for entry in trace:
                status = "OK" if entry.get("success") else "FAIL"
                print(f"  [{status}] {entry.get('step', '?')}: "
                      f"{', '.join(f'{k}={v}' for k, v in entry.items() if k not in ('step', 'success'))}")

        results.append({
            "image": image_name,
            "label": label,
            "descriptor": descriptor,
            "plan": plan,
            "execution_trace": trace,
            "n_llm_calls": n_llm,
            "elapsed": elapsed,
            "success": success,
        })

        print(f"\n  RESULT: descriptor={descriptor}, llm_calls={n_llm}, elapsed={elapsed:.1f}s")

    # Consensus
    if descriptor_votes:
        from collections import Counter
        consensus = Counter(descriptor_votes).most_common(1)[0][0]
    else:
        consensus = "unknown"

    total_elapsed = sum(r.get('elapsed', 0) for r in results)

    return {
        "method": "topoagent_v5",
        "descriptor_selected": consensus,
        "descriptor_votes": descriptor_votes,
        "demo_results": results,
        "n_images": n_demo,
        "total_llm_calls": sum(r.get('n_llm_calls', 0) for r in results),
        "total_elapsed_s": total_elapsed,
    }


# =============================================================================
# Phase B: Batch Evaluation (efficient, 6 classifiers)
# =============================================================================

def run_batch_evaluation(
    dataset_name: str,
    descriptor: str,
    n_eval: int = 1000,
    cv_folds: int = 5,
    classifiers: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase B: Batch feature extraction + 6-classifier evaluation.

    Uses the benchmark_runner infrastructure for feature extraction
    and the classifier_wrapper for all 6 classifiers.

    Returns:
        Dict with per_classifier accuracies and best_classifier info
    """
    if classifiers is None:
        classifiers = ALL_CLASSIFIERS

    from topoagent.benchmark_runner import TopoBenchmarkRunner, IMAGE_BASED_DESCRIPTORS, _ensure_benchmark4_path
    from TopoBenchmark.config import DATASET_DESCRIPTIONS

    desc = DATASET_DESCRIPTIONS.get(dataset_name, {})
    object_type = desc.get("object_type", "surface_lesions")
    color_mode = desc.get("color_mode", "grayscale")

    print(f"\n  Phase B: Extracting features for {descriptor} on {dataset_name} (n={n_eval})...")

    # Create a runner with no model (not needed for batch eval)
    runner = TopoBenchmarkRunner(model=None, lookup_mode=False)

    start = time.time()
    best_accuracy, details = runner._batch_extract_and_evaluate(
        dataset_name=dataset_name,
        descriptor=descriptor,
        object_type=object_type,
        color_mode=color_mode,
        n_eval=n_eval,
        cv_folds=cv_folds,
        seed=seed,
        classifiers=classifiers,
    )
    elapsed = time.time() - start

    # Print results table
    per_clf = details.get("per_classifier", {})
    print(f"\n  {'─' * 60}")
    print(f"  CLASSIFIER RESULTS ({dataset_name} / {descriptor})")
    print(f"  {'─' * 60}")
    print(f"  {'Classifier':<15} {'Bal.Acc':>10} {'Std':>10} {'Status':>10}")
    print(f"  {'-' * 15} {'-' * 10} {'-' * 10} {'-' * 10}")

    for clf_name in classifiers:
        if clf_name in per_clf:
            entry = per_clf[clf_name]
            acc = entry.get("accuracy", 0)
            std = entry.get("std", 0)
            error = entry.get("error", "")
            if error:
                print(f"  {clf_name:<15} {'--':>10} {'--':>10} {'ERROR':>10}")
            else:
                marker = " <-- BEST" if clf_name == details.get("best_classifier") else ""
                print(f"  {clf_name:<15} {acc:>9.1%} {std:>9.3f} {'OK':>10}{marker}")
        else:
            print(f"  {clf_name:<15} {'--':>10} {'--':>10} {'SKIP':>10}")

    print(f"\n  Feature dim: {details.get('feature_dim', '?')}")
    print(f"  Best: {details.get('best_classifier', '?')} ({best_accuracy:.1%})")
    print(f"  Elapsed: {elapsed:.1f}s")

    details["elapsed_s"] = elapsed
    return {
        "descriptor": descriptor,
        "best_accuracy": best_accuracy,
        "best_classifier": details.get("best_classifier", "?"),
        "per_classifier": per_clf,
        "feature_dim": details.get("feature_dim", 0),
        "n_eval": n_eval,
        "cv_folds": cv_folds,
        "elapsed_s": elapsed,
    }


# =============================================================================
# Oracle: Compute per-dataset best accuracy
# =============================================================================

def get_oracle_accuracy(dataset_name: str) -> Tuple[str, float]:
    """Get oracle (best descriptor + best classifier) accuracy from assets."""
    assets_dir = PROJECT_ROOT / "results" / "topobenchmark" / "assets"
    oracle_path = assets_dir / "oracle.json"
    if oracle_path.exists():
        with open(oracle_path) as f:
            oracle_data = json.load(f)
        # oracle.json has {mean_accuracy, per_dataset: {dataset: {accuracy, descriptor}}}
        per_ds = oracle_data.get("per_dataset", oracle_data)
        if dataset_name in per_ds:
            entry = per_ds[dataset_name]
            return entry["descriptor"], entry["accuracy"]
    return "unknown", 0.0


# =============================================================================
# Main: Protocol 2 Pipeline
# =============================================================================

def run_protocol2(
    datasets: List[str],
    methods: List[str],
    n_demo: int = 3,
    n_eval: int = 1000,
    cv_folds: int = 5,
    classifiers: Optional[List[str]] = None,
    verbose: bool = False,
    skip_demo: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Protocol 2: Combined Phase A (agent demo) + Phase B (batch eval).

    For each dataset and method:
      1. Phase A: Run agent on sample images (verbose, per-image)
      2. Phase B: Extract features + run 6 classifiers (efficient, batch)

    Returns:
        Full results dictionary
    """
    if classifiers is None:
        classifiers = ALL_CLASSIFIERS

    all_results = []
    timestamp_start = datetime.now()

    for dataset in datasets:
        print(f"\n{'#' * 70}")
        print(f"#  DATASET: {dataset}")
        print(f"{'#' * 70}")

        for method in methods:
            print(f"\n{'=' * 70}")
            print(f"  Method: {method} on {dataset}")
            print(f"{'=' * 70}")

            # Phase A: Agent demo (descriptor selection)
            demo_result = None
            descriptor_selected = None

            if not skip_demo:
                try:
                    if method == "topoagent_v4":
                        demo_result = run_agent_demo_v4(
                            dataset_name=dataset,
                            n_demo=n_demo,
                            verbose=verbose,
                        )
                    elif method == "topoagent_v5":
                        demo_result = run_agent_demo_v5(
                            dataset_name=dataset,
                            n_demo=n_demo,
                            verbose=verbose,
                        )
                    else:
                        print(f"  Unknown method for Phase A: {method}")

                    if demo_result:
                        descriptor_selected = demo_result["descriptor_selected"]
                except Exception as e:
                    print(f"  Phase A FAILED: {e}")
                    import traceback
                    traceback.print_exc()

            # Fallback: use benchmark_runner's selection if demo failed
            if descriptor_selected is None or descriptor_selected == "unknown":
                print(f"  Phase A skipped or failed, using benchmark_runner selection...")
                try:
                    from topoagent.benchmark_runner import TopoBenchmarkRunner
                    from langchain_openai import ChatOpenAI

                    api_key = os.getenv("OPENAI_API_KEY")
                    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3) if api_key else None

                    runner = TopoBenchmarkRunner(model=model, lookup_mode=False)
                    from TopoBenchmark.config import DATASET_DESCRIPTIONS
                    desc = DATASET_DESCRIPTIONS.get(dataset, {})
                    context = runner._build_dataset_context_light(dataset)
                    selection = runner._select_topoagent(context) if model else {"descriptor_choice": "template_functions"}
                    descriptor_selected = selection.get("descriptor_choice", "template_functions")
                    print(f"  Fallback selection: {descriptor_selected}")
                except Exception as e2:
                    print(f"  Fallback also failed: {e2}")
                    descriptor_selected = "template_functions"

            # Phase B: Batch evaluation
            batch_result = None
            try:
                batch_result = run_batch_evaluation(
                    dataset_name=dataset,
                    descriptor=descriptor_selected,
                    n_eval=n_eval,
                    cv_folds=cv_folds,
                    classifiers=classifiers,
                    seed=seed,
                )
            except Exception as e:
                print(f"  Phase B FAILED: {e}")
                import traceback
                traceback.print_exc()
                batch_result = {
                    "error": str(e),
                    "best_accuracy": 0.0,
                    "per_classifier": {},
                }

            # Get oracle for regret
            oracle_desc, oracle_acc = get_oracle_accuracy(dataset)

            # Combine results
            best_acc = batch_result.get("best_accuracy", 0.0) if batch_result else 0.0

            result = {
                "dataset": dataset,
                "method": method,
                "descriptor_selected": descriptor_selected,
                "per_classifier": batch_result.get("per_classifier", {}) if batch_result else {},
                "best_classifier": batch_result.get("best_classifier", "?") if batch_result else "?",
                "best_accuracy": best_acc,
                "oracle_descriptor": oracle_desc,
                "oracle_accuracy": oracle_acc,
                "regret": oracle_acc - best_acc,
                "feature_dim": batch_result.get("feature_dim", 0) if batch_result else 0,
                "n_eval": n_eval,
                "cv_folds": cv_folds,
                "classifiers": classifiers,
                "demo_elapsed_s": demo_result.get("total_elapsed_s", 0) if demo_result else 0,
                "eval_elapsed_s": batch_result.get("elapsed_s", 0) if batch_result else 0,
                "llm_calls": demo_result.get("total_llm_calls", 0) if demo_result else 0,
                "reasoning_trace": [
                    r.get("reasoning_trace", [])
                    for r in (demo_result.get("demo_results", []) if demo_result else [])
                ],
                "timestamp": datetime.now().isoformat(),
            }

            all_results.append(result)

            # Print result row
            print(f"\n  RESULT: {method}/{dataset}")
            print(f"    Descriptor: {descriptor_selected}")
            print(f"    Best: {result['best_classifier']} ({best_acc:.1%})")
            print(f"    Oracle: {oracle_desc} ({oracle_acc:.1%})")
            print(f"    Regret: {result['regret']:.3f}")

    # Print summary table
    _print_summary_table(all_results, datasets, methods)

    # Build output
    output = {
        "config": {
            "datasets": datasets,
            "methods": methods,
            "n_demo": n_demo,
            "n_eval": n_eval,
            "cv_folds": cv_folds,
            "classifiers": classifiers,
            "verbose": verbose,
            "seed": seed,
            "timestamp": timestamp_start.isoformat(),
        },
        "results": all_results,
    }

    return output


def _print_summary_table(results: List[Dict], datasets: List[str], methods: List[str]):
    """Print a summary table: methods x datasets."""
    print(f"\n{'═' * 80}")
    print("PROTOCOL 2 SUMMARY")
    print(f"{'═' * 80}")

    # Build lookup: method -> dataset -> result
    lookup = {}
    for r in results:
        key = (r["method"], r["dataset"])
        lookup[key] = r

    # Header
    ds_short = [d[:12] for d in datasets]
    header = f"{'Method':<18}"
    for ds in ds_short:
        header += f" {ds:>12}"
    header += f" {'Avg':>8} {'Regret':>8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<18}"
        accs = []
        regrets = []
        for dataset in datasets:
            r = lookup.get((method, dataset))
            if r and r.get("best_accuracy", 0) > 0:
                acc = r["best_accuracy"]
                accs.append(acc)
                regrets.append(r.get("regret", 0))
                row += f" {acc:>11.1%}"
            else:
                row += f" {'--':>12}"

        if accs:
            row += f" {np.mean(accs):>7.1%}"
            row += f" {np.mean(regrets):>7.3f}"
        else:
            row += f" {'--':>8} {'--':>8}"

        print(row)

    # Per-classifier breakdown for each result
    print(f"\n{'─' * 80}")
    print("PER-CLASSIFIER BREAKDOWN")
    print(f"{'─' * 80}")
    for r in results:
        per_clf = r.get("per_classifier", {})
        if per_clf:
            print(f"\n  {r['method']}/{r['dataset']} ({r['descriptor_selected']}):")
            for clf_name, entry in sorted(per_clf.items(), key=lambda x: -x[1].get("accuracy", 0)):
                if entry.get("error"):
                    print(f"    {clf_name:<15}: ERROR — {entry['error'][:60]}")
                else:
                    acc = entry.get("accuracy", 0)
                    std = entry.get("std", 0)
                    best_marker = " *BEST*" if clf_name == r.get("best_classifier") else ""
                    print(f"    {clf_name:<15}: {acc:.1%} +/- {std:.3f}{best_marker}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Protocol 2: End-to-End TopoAgent Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=str, default="BloodMNIST",
        help="Dataset name(s), comma-separated (default: BloodMNIST)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all 26 datasets",
    )
    parser.add_argument(
        "--methods", type=str, default="topoagent_v4",
        help="Methods to evaluate, comma-separated (default: topoagent_v4)",
    )
    parser.add_argument(
        "--n-demo", type=int, default=3,
        help="Number of demo images per dataset for Phase A (default: 3)",
    )
    parser.add_argument(
        "--n-eval", type=int, default=1000,
        help="Number of evaluation images for Phase B (default: 1000)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--classifiers", type=str, default=None,
        help="Classifiers to use, comma-separated (default: all 6)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full agent traces (prompts, responses, tool outputs)",
    )
    parser.add_argument(
        "--no-demo", action="store_true",
        help="Skip Phase A (agent demo), use benchmark_runner for descriptor selection",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Parse datasets
    if args.all:
        from TopoBenchmark.config import DATASET_DESCRIPTIONS
        datasets = list(DATASET_DESCRIPTIONS.keys())
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",")]

    # Parse classifiers
    classifiers = None
    if args.classifiers:
        classifiers = [c.strip() for c in args.classifiers.split(",")]

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.no_demo:
        print("WARNING: OPENAI_API_KEY not set. Phase A (agent demo) will fail.")
        print("  Set the key or use --no-demo to skip Phase A.")

    print(f"\n{'═' * 70}")
    print(f"  PROTOCOL 2: End-to-End TopoAgent Evaluation")
    print(f"{'═' * 70}")
    print(f"  Datasets:    {datasets}")
    print(f"  Methods:     {methods}")
    print(f"  Phase A:     {'SKIP' if args.no_demo else f'{args.n_demo} demo images, verbose={args.verbose}'}")
    print(f"  Phase B:     n_eval={args.n_eval}, cv_folds={args.cv_folds}")
    print(f"  Classifiers: {classifiers or ALL_CLASSIFIERS}")
    print(f"  Seed:        {args.seed}")
    print(f"{'═' * 70}")

    # Run
    output = run_protocol2(
        datasets=datasets,
        methods=methods,
        n_demo=args.n_demo,
        n_eval=args.n_eval,
        cv_folds=args.cv_folds,
        classifiers=classifiers,
        verbose=args.verbose,
        skip_demo=args.no_demo,
        seed=args.seed,
    )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_slug = "_".join(datasets[:3])
    if len(datasets) > 3:
        ds_slug += f"_+{len(datasets) - 3}more"
    results_path = OUTPUT_DIR / f"protocol2_{ds_slug}_{timestamp}.json"

    serializable = json.loads(json.dumps(output, default=str))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return output


if __name__ == "__main__":
    main()
