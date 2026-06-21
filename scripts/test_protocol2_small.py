#!/usr/bin/env python3
"""Protocol 2: TopoAgent v5 evaluation — User Mode and Benchmark Mode.

User Mode (default): Per-image descriptor selection via v5 workflow (2-3 LLM calls).
Benchmark Mode: Per-dataset descriptor selection + batch CV evaluation (1 LLM call).

Usage:
    # User Mode (per-image)
    python scripts/test_protocol2_small.py --mode user --dataset BloodMNIST --n-samples 1
    python scripts/test_protocol2_small.py --mode user --dataset DermaMNIST --verify-mode thorough

    # Benchmark Mode — Live (per-dataset, extract+classify)
    python scripts/test_protocol2_small.py --mode benchmark --dataset BloodMNIST
    python scripts/test_protocol2_small.py --mode benchmark --dataset BloodMNIST --methods topoagent,zeroshot,oracle

    # Benchmark Mode — Lookup (pre-computed, fast ~30s)
    python scripts/test_protocol2_small.py --mode benchmark --lookup --dataset BloodMNIST,PathMNIST
    python scripts/test_protocol2_small.py --mode benchmark --lookup --methods meta_learner,rule_based,sba,oracle

    # Experiment A (zero-shot, no LODO)
    python scripts/test_protocol2_small.py --mode benchmark --lookup --experiment A

    # Experiment B (skills + LODO)
    python scripts/test_protocol2_small.py --mode benchmark --lookup --experiment B
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


def save_sample_images(dataset_name: str, n_samples: int = 3, output_dir: Path = None):
    """Load dataset and save sample images as PNG files.

    Returns:
        List of (image_path, label) tuples
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2" / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    from RuleBenchmark.benchmark4.data_loader import load_dataset
    images, labels, class_names = load_dataset(dataset_name, n_samples=n_samples, seed=42)

    print(f"Loaded {len(images)} images from {dataset_name}")
    print(f"  Shape: {images[0].shape}, dtype: {images[0].dtype}")
    print(f"  Labels: {labels[:n_samples].tolist()}")

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

        fname = f"{dataset_name}_sample_{i}_label{label}.png"
        fpath = output_dir / fname
        pil_img.save(fpath)
        samples.append((str(fpath), label))
        print(f"  Saved: {fname} (label={label})")

    return samples


def create_v5_agent(verify_mode="quick"):
    """Create TopoAgent with v5 workflow (no monkey-patches needed)."""
    from topoagent.agent import create_topoagent
    from topoagent.tools.descriptors import get_all_descriptors
    from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS

    agent = create_topoagent(
        workflow_version="v5",
        verify_mode=verify_mode,
        temperature=0.3,
    )

    # Build focused tool set with all supported descriptors
    descriptor_tools = get_all_descriptors()
    essential_tools = {}
    essential_tools["image_loader"] = agent.tools["image_loader"]
    essential_tools["compute_ph"] = agent.tools["compute_ph"]
    for desc_name in SUPPORTED_DESCRIPTORS:
        if desc_name in descriptor_tools:
            essential_tools[desc_name] = descriptor_tools[desc_name]
        elif desc_name in agent.tools:
            essential_tools[desc_name] = agent.tools[desc_name]
    for clf in ["knn_classifier", "pytorch_classifier"]:
        if clf in agent.tools:
            essential_tools[clf] = agent.tools[clf]
    agent.tools = essential_tools

    # Rebuild v5 workflow with updated tools
    from topoagent.workflow import TopoAgentWorkflowV5
    agent.workflow = TopoAgentWorkflowV5(
        model=agent.model,
        tools=agent.tools,
        verify_mode=verify_mode,
    )

    print(f"Agent created with {len(agent.tools)} tools (v5 workflow, verify_mode={verify_mode})")
    print(f"Tools: {sorted(agent.tools.keys())}")
    return agent


def build_query(dataset_name: str) -> str:
    """Build a natural query for the agent."""
    from TopoBenchmark.config import DATASET_DESCRIPTIONS
    desc = DATASET_DESCRIPTIONS[dataset_name]

    return (
        f"Find the optimal topology feature for this medical image from the {dataset_name} dataset.\n\n"
        f"Domain: {desc['domain']}\n"
        f"Description: {desc['description']}\n"
        f"What matters: {desc['what_matters']}\n"
        f"Number of classes: {desc['n_classes']}\n"
        f"Object type: {desc['object_type']}\n"
    )


def run_single_image(agent, image_path: str, query: str, label: int,
                     verify_mode: str = "quick", dataset_name: str = None):
    """Run the v5 workflow on a single image and display detailed trace."""

    print(f"\n{'='*80}")
    print(f"IMAGE: {Path(image_path).name}")
    print(f"TRUE LABEL: {label}")
    print(f"{'='*80}")

    start = time.time()
    try:
        final_state = agent.workflow.invoke(
            query=query,
            image_path=image_path,
            verify_mode=verify_mode,
            dataset_name=dataset_name,
        )
        elapsed = time.time() - start
        success = True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "elapsed": elapsed}

    # =========================================================================
    # Print v5 trace
    # =========================================================================

    plan = final_state.get("plan", {})
    trace = final_state.get("execution_trace", [])
    verification = final_state.get("verification", {})
    report = final_state.get("report", "")
    feature_vector = final_state.get("feature_vector")

    # 1. PLAN
    print(f"\n{'─'*60}")
    print("=== ANALYZE & PLAN ===")
    print(f"{'─'*60}")
    if plan:
        print(json.dumps(plan, indent=2, default=str))
    else:
        print("No plan generated.")

    # 2. EXECUTION TRACE
    print(f"\n{'─'*60}")
    print("=== EXTRACT FEATURES ===")
    print(f"{'─'*60}")
    for entry in trace:
        step = entry.get("step", "unknown")
        ok = entry.get("success", False)
        status = "OK" if ok else "FAIL"
        details = {k: v for k, v in entry.items() if k not in ("step", "success")}
        print(f"  [{status}] {step}: {details}")

    if feature_vector is not None:
        if isinstance(feature_vector, np.ndarray):
            print(f"  Feature vector: shape={feature_vector.shape}, dtype={feature_vector.dtype}")
            print(f"    min={feature_vector.min():.4f}, max={feature_vector.max():.4f}, "
                  f"mean={feature_vector.mean():.4f}, std={feature_vector.std():.4f}")
        else:
            print(f"  Feature vector: len={len(feature_vector)}")

    # 3. VERIFICATION
    print(f"\n{'─'*60}")
    print(f"=== VERIFY ({verification.get('mode', 'none')}) ===")
    print(f"{'─'*60}")
    if verification:
        # Print verification result
        for k, v in verification.items():
            if k == "results" and isinstance(v, dict):
                print(f"  {k}:")
                for desc, res in v.items():
                    acc = res.get("accuracy", 0)
                    std = res.get("std", 0)
                    dim = res.get("dim", "?")
                    print(f"    {desc}: {acc:.1%} +/- {std:.1%} (dim={dim})")
            else:
                print(f"  {k}: {v}")
    else:
        print("  No verification performed.")

    # 4. REPORT
    print(f"\n{'─'*60}")
    print("=== OUTPUT REPORT ===")
    print(f"{'─'*60}")
    if report:
        print(report[:2000])
        if len(report) > 2000:
            print("...[truncated]")
    else:
        print("No report generated.")

    # 5. LLM INTERACTIONS
    print(f"\n{'─'*60}")
    n_llm = len(final_state.get("llm_interactions", []))
    print(f"LLM CALLS: {n_llm}")
    print(f"ELAPSED: {elapsed:.1f}s")
    for interaction in final_state.get("llm_interactions", []):
        print(f"  - [{interaction.step}] {len(interaction.response)} chars")

    # 6. SKILL STATE
    print(f"\n{'─'*60}")
    print("SKILL STATE")
    print(f"{'─'*60}")
    print(f"  Descriptor: {final_state.get('skill_descriptor', 'none')}")
    print(f"  Color mode: {final_state.get('skill_color_mode', 'none')}")
    params = final_state.get('skill_params', {})
    if params:
        print(f"  Params: {json.dumps({k:v for k,v in params.items() if k not in ('total_dim','classifier','color_mode','dim')}, default=str)}")
        print(f"  Total dim: {params.get('total_dim', '?')}")
        print(f"  Classifier: {params.get('classifier', '?')}")

    # Build result
    result = {
        "image_path": image_path,
        "true_label": label,
        "plan": plan,
        "execution_trace": trace,
        "verification": verification,
        "skill_descriptor": final_state.get("skill_descriptor"),
        "skill_params": final_state.get("skill_params"),
        "retry_used": final_state.get("retry_used", False),
        "n_llm_calls": n_llm,
        "elapsed": elapsed,
        "success": success,
    }

    return result


def run_benchmark_mode(args):
    """Benchmark Mode (Mode A): Per-dataset descriptor selection + batch CV evaluation."""
    from topoagent.benchmark_runner import TopoBenchmarkRunner, NO_LLM_METHODS
    from topoagent.memory.skill_memory import SkillMemory

    # Parse datasets
    datasets = [d.strip() for d in args.dataset.split(",")]

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",")]

    # Determine experiment type
    experiment = getattr(args, "experiment", "A")
    lookup = getattr(args, "lookup", False)
    lodo = (experiment == "B")

    print(f"\n{'='*80}")
    print(f"BENCHMARK MODE — Experiment {experiment} "
          f"({len(datasets)} datasets, {'lookup' if lookup else 'live'} mode"
          f"{', LODO' if lodo else ''})")
    print(f"{'='*80}")
    print(f"  Datasets: {datasets}")
    print(f"  Methods: {methods}")
    if not lookup:
        print(f"  n_eval: {args.n_eval}")
        print(f"  cv_folds: {args.cv_folds}")
    else:
        print(f"  Mode: lookup (pre-computed benchmark4 results)")

    # Check if any methods need an LLM
    needs_llm = any(
        m not in NO_LLM_METHODS and not m.startswith("fixed_")
        for m in methods
    )

    # Create model only if needed
    model = None
    if needs_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set (required for LLM methods)")
            sys.exit(1)
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

    skill_memory = SkillMemory()
    runner = TopoBenchmarkRunner(model=model, skill_memory=skill_memory,
                                 lookup_mode=lookup)

    # Run evaluation
    all_output = runner.evaluate_all(
        datasets=datasets,
        methods=methods,
        n_eval=args.n_eval,
        cv_folds=args.cv_folds,
        seed=args.seed,
        lodo=lodo,
    )

    # Save results
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_slug = "_".join(datasets[:3])
    results_path = output_dir / f"benchmark_{ds_slug}_{timestamp}.json"

    # Serialize (strip non-serializable _interaction objects)
    serializable = json.loads(json.dumps(all_output, default=str))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print accuracy matrix table (datasets as columns)
    summary = all_output.get("summary", {})
    print(f"\n{'='*80}")
    print("ACCURACY MATRIX (Balanced Accuracy %)")
    print(f"{'='*80}")
    table = runner.format_matrix_table(summary, datasets)
    print(table)

    # Print per-method aggregate metrics
    per_method = summary.get("per_method", {})
    if per_method:
        print(f"\n{'─'*80}")
        print("AGGREGATE METRICS")
        print(f"{'─'*80}")
        print(f"{'Method':<26} {'Top-1':>7} {'Top-3':>7} {'LLM':>5}")
        print(f"{'-'*26} {'-'*7} {'-'*7} {'-'*5}")
        for method, stats in per_method.items():
            print(f"{method:<26} "
                  f"{stats['top1_rate']:>6.1%} "
                  f"{stats['top3_rate']:>6.1%} "
                  f"{stats['total_llm_calls']:>5}")

    # Print descriptor choices per dataset
    desc_matrix = summary.get("descriptor_matrix", {})
    if desc_matrix:
        print(f"\n{'─'*80}")
        print("DESCRIPTOR CHOICES")
        print(f"{'─'*80}")
        for method, ds_descs in desc_matrix.items():
            for ds, desc in ds_descs.items():
                print(f"  {method}/{ds}: {desc}")

    return all_output


def run_user_mode(args):
    """User Mode (Mode B): Per-image descriptor selection via v5 workflow."""
    # Step 1: Save sample images
    print(f"\n{'='*80}")
    print(f"STEP 1: Loading {args.n_samples} sample(s) from {args.dataset}")
    print(f"{'='*80}")
    samples = save_sample_images(args.dataset, n_samples=args.n_samples)

    # Step 2: Create v5 agent
    print(f"\n{'='*80}")
    print(f"STEP 2: Creating TopoAgent v5 (verify_mode={args.verify_mode})")
    print(f"{'='*80}")
    agent = create_v5_agent(verify_mode=args.verify_mode)

    # Step 3: Build query
    query = build_query(args.dataset)
    print(f"\nQuery:\n{query}")

    # Step 4: Run on each image
    all_results = []
    for image_path, label in samples:
        result = run_single_image(
            agent, image_path, query, label,
            verify_mode=args.verify_mode,
            dataset_name=args.dataset,
        )
        all_results.append(result)

        # LEARN — update skill memory after each run
        if result.get("success") and result.get("skill_descriptor"):
            try:
                agent.workflow.skill_memory.record_reflection(
                    context=f"{result.get('plan', {}).get('object_type', 'unknown')} on {args.dataset}",
                    choice=result["skill_descriptor"],
                    outcome="success" if result["success"] else "failure",
                    reflection_text=f"Used {result['skill_descriptor']} on {args.dataset}.",
                    lesson=f"For {args.dataset}, {result['skill_descriptor']} extraction succeeded in {result['elapsed']:.1f}s.",
                )
                print(f"\n  [LEARN] Stored reflection for {result['skill_descriptor']} on {args.dataset}")
            except Exception as e:
                print(f"\n  [LEARN] Failed to store reflection: {e}")

    # Save results
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"v5_test_{args.dataset}_{timestamp}.json"

    # Make results JSON-serializable
    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if hasattr(v, "item"):
                sr[k] = v.item()
            elif isinstance(v, dict):
                sr[k] = {kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in v.items()}
            elif isinstance(v, np.ndarray):
                sr[k] = f"ndarray shape={v.shape}"
            else:
                sr[k] = v
        serializable.append(sr)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in all_results:
        img = Path(r.get("image_path", "unknown")).name if r.get("image_path") else "unknown"
        desc = r.get("skill_descriptor", "unknown")
        llm_calls = r.get("n_llm_calls", 0)
        elapsed = r.get("elapsed", 0)
        success = r.get("success", False)
        retry = r.get("retry_used", False)
        plan = r.get("plan", {})
        ot = plan.get("object_type", "?")
        chain = plan.get("reasoning_chain", "?")
        print(f"  {img}: success={success}, descriptor={desc}, object_type={ot}, "
              f"chain={chain}, retry={retry}, llm_calls={llm_calls}, elapsed={elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Protocol 2: TopoAgent v5 Evaluation")
    parser.add_argument("--mode", type=str, default="user",
                        choices=["user", "benchmark"],
                        help="Evaluation mode: 'user' (per-image) or 'benchmark' (per-dataset)")
    parser.add_argument("--dataset", type=str, default="BloodMNIST",
                        help="Dataset name(s). Comma-separated for benchmark mode (default: BloodMNIST)")

    # User mode args
    parser.add_argument("--n-samples", type=int, default=1,
                        help="[User mode] Number of sample images (default: 1)")
    parser.add_argument("--verify-mode", type=str, default="quick",
                        choices=["quick", "thorough"],
                        help="[User mode] Verification mode (default: quick)")

    # Benchmark mode args
    parser.add_argument("--methods", type=str,
                        default="topoagent,topoagent_no_skills,meta_learner,rule_based,sba,zeroshot,oracle",
                        help="[Benchmark] Comma-separated methods")
    parser.add_argument("--n-eval", type=int, default=1000,
                        help="[Benchmark] Number of evaluation images per dataset (default: 1000)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="[Benchmark] Number of CV folds (default: 5)")
    parser.add_argument("--experiment", type=str, default="A", choices=["A", "B"],
                        help="[Benchmark] A=zero-shot (no LODO), B=skills+LODO (default: A)")
    parser.add_argument("--lookup", action="store_true",
                        help="[Benchmark] Use pre-computed benchmark4 results (requires precompute_benchmark_assets.py)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # Check API key for user mode (benchmark mode checks inside run_benchmark_mode)
    if args.mode == "user":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set")
            sys.exit(1)
        print("API key: set")

    if args.mode == "benchmark":
        run_benchmark_mode(args)
    else:
        run_user_mode(args)


if __name__ == "__main__":
    main()
