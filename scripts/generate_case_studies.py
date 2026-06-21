#!/usr/bin/env python3
"""Generate paper-ready case studies showing the full TopoAgent reasoning pipeline.

For each of 5 representative datasets (one per object type), this script:
1. Saves a representative image as PNG
2. Runs the agent with skills_mode=True
3. Extracts the full reasoning trace: LLM prompts/responses, tool outputs
4. Formats into structured case study JSON
5. Generates a LaTeX table summarizing all 5 case studies
6. Saves to results/case_studies/

Usage:
    python scripts/generate_case_studies.py
    python scripts/generate_case_studies.py --dataset BloodMNIST --verbose
    python scripts/generate_case_studies.py --output results/case_studies/
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DATASETS = [
    {"name": "BloodMNIST", "object_type": "discrete_cells", "color_mode": "per_channel"},
    {"name": "PathMNIST", "object_type": "glands_lumens", "color_mode": "per_channel"},
    {"name": "RetinaMNIST", "object_type": "vessel_trees", "color_mode": "per_channel"},
    {"name": "DermaMNIST", "object_type": "surface_lesions", "color_mode": "per_channel"},
    {"name": "OrganAMNIST", "object_type": "organ_shape", "color_mode": "grayscale"},
]


# =============================================================================
# Helpers
# =============================================================================

def save_sample_images(dataset_name: str, output_dir: Path, n_samples: int = 1, seed: int = 42) -> List[Tuple[str, int]]:
    """Save N sample images from a dataset as PNGs. Returns [(path, label), ...]."""
    output_dir.mkdir(parents=True, exist_ok=True)

    b4_dir = PROJECT_ROOT / "RuleBenchmark" / "benchmark4"
    if str(b4_dir) not in sys.path:
        sys.path.insert(0, str(b4_dir))
    from data_loader import load_dataset

    images, labels, class_names = load_dataset(dataset_name, n_samples=n_samples, seed=seed)

    from PIL import Image
    results = []
    for idx in range(len(images)):
        img = images[idx]
        label = int(labels[idx])
        if img.ndim == 2:
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        elif img.ndim == 3 and img.shape[2] == 3:
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
        else:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))

        suffix = f"_{idx}" if n_samples > 1 else ""
        fpath = output_dir / f"{dataset_name}_case_study{suffix}.png"
        pil_img.save(fpath)
        results.append((str(fpath), label))
    return results


def build_query(dataset_name: str) -> str:
    """Build a query for the agent that includes dataset context."""
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


def load_accuracy_lookup() -> Dict[str, Dict[str, float]]:
    """Load Benchmark4 accuracy lookup."""
    lookup_path = PROJECT_ROOT / "results" / "topobenchmark" / "assets" / "accuracy_lookup.json"
    if lookup_path.exists():
        with open(lookup_path) as f:
            return json.load(f)
    return {}


def get_oracle(dataset_name: str, accuracy_lookup: Dict) -> Tuple[str, float]:
    """Get oracle (best descriptor, accuracy) for a dataset."""
    ds_data = accuracy_lookup.get(dataset_name, {})
    if not ds_data:
        return "unknown", 0.0
    best = max(ds_data.items(), key=lambda x: x[1])
    return best[0], best[1]


def extract_ph_stats_from_state(state: Dict) -> Dict:
    """Extract PH statistics from agent state."""
    ph_stats = state.get("_ph_stats") or {}
    if ph_stats:
        return ph_stats

    # Fallback: extract from short-term memory
    for tool_name, output in state.get("short_term_memory", []):
        if tool_name == "compute_ph" and isinstance(output, dict):
            stats = output.get("statistics", {})
            return {
                "H0_count": stats.get("H0", {}).get("count", 0),
                "H1_count": stats.get("H1", {}).get("count", 0),
            }
    return {}


def extract_case_study(
    dataset_name: str,
    ds_config: Dict,
    final_state: Dict,
    query: str,
    accuracy_lookup: Dict,
) -> Dict[str, Any]:
    """Extract a structured case study from the agent's final state."""

    plan = final_state.get("_plan") or {}
    report = final_state.get("final_answer") or {}
    feature_quality = final_state.get("_feature_quality") or {}
    ph_stats = extract_ph_stats_from_state(final_state)
    interactions = final_state.get("llm_interactions", [])
    descriptor = final_state.get("skill_descriptor", "unknown")
    skill_params = final_state.get("skill_params") or {}

    # Get accuracy from lookup
    ds_accuracy = accuracy_lookup.get(dataset_name, {})
    agent_accuracy = ds_accuracy.get(descriptor, 0.0)
    oracle_descriptor, oracle_accuracy = get_oracle(dataset_name, accuracy_lookup)
    regret = oracle_accuracy - agent_accuracy

    # Extract LLM interactions
    plan_interaction = None
    verify_interaction = None
    for interaction in interactions:
        if interaction.step == "plan_descriptor":
            plan_interaction = interaction
        elif interaction.step == "verify_and_reflect":
            verify_interaction = interaction

    # Filter parameters
    safe_params = {
        k: v for k, v in skill_params.items()
        if k not in ("total_dim", "classifier", "color_mode", "dim")
    }

    # Verification data
    verification = report.get("verification", {}) if isinstance(report, dict) else {}

    case_study = {
        "dataset": dataset_name,
        "object_type": ds_config["object_type"],
        "color_mode": ds_config["color_mode"],

        "input": {
            "query": query[:200] + "..." if len(query) > 200 else query,
            "ph_statistics": ph_stats,
            "color_mode": final_state.get("skill_color_mode", ds_config["color_mode"]),
        },

        "reasoning": {
            "object_type": plan.get("object_type", "unknown"),
            "reasoning_chain": plan.get("reasoning_chain", "unknown"),
            "image_analysis": plan.get("image_analysis", ""),
            "descriptor_choice": plan.get("descriptor_choice", descriptor),
            "descriptor_rationale": plan.get("descriptor_rationale", ""),
            "alternative": plan.get("alternative_descriptor", ""),
            "alternative_rationale": plan.get("alternative_rationale", ""),
        },

        "action": {
            "tools_executed": [name for name, _ in final_state.get("short_term_memory", [])],
            "parameters": safe_params,
            "feature_quality": {
                "dimension": feature_quality.get("dimension", 0),
                "sparsity": round(feature_quality.get("sparsity", 0), 1),
                "variance": feature_quality.get("variance", 0),
                "nan_count": feature_quality.get("nan_count", 0),
            },
        },

        "reflection": {
            "ph_confirms_object_type": verification.get("ph_confirms_object_type", None),
            "quality_ok": verification.get("quality_ok", None),
            "dimension_correct": verification.get("dimension_correct", None),
            "issues": verification.get("issues", []),
            "suggestion": verification.get("suggestion", ""),
            "error_analysis": report.get("error_analysis", "") if isinstance(report, dict) else "",
            "experience": report.get("experience", "") if isinstance(report, dict) else "",
        },

        "output": {
            "descriptor": descriptor,
            "accuracy": round(agent_accuracy * 100, 2) if agent_accuracy else None,
            "oracle_descriptor": oracle_descriptor,
            "oracle_accuracy": round(oracle_accuracy * 100, 2) if oracle_accuracy else None,
            "regret": round(regret * 100, 2) if oracle_accuracy else None,
        },

        "metadata": {
            "n_llm_calls": report.get("n_llm_calls", len(interactions)) if isinstance(report, dict) else len(interactions),
            "n_tools_executed": report.get("n_tools_executed", len(final_state.get("short_term_memory", []))) if isinstance(report, dict) else 0,
            "retry_used": report.get("retry_used", False) if isinstance(report, dict) else False,
            "total_time_s": report.get("total_time_seconds", 0) if isinstance(report, dict) else 0,
        },

        "llm_traces": {
            "plan_prompt_chars": len(plan_interaction.prompt) if plan_interaction else 0,
            "plan_response_chars": len(plan_interaction.response) if plan_interaction else 0,
            "verify_prompt_chars": len(verify_interaction.prompt) if verify_interaction else 0,
            "verify_response_chars": len(verify_interaction.response) if verify_interaction else 0,
        },
    }

    return case_study


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_aggregate_table(case_studies: List[Dict]) -> str:
    """Generate a LaTeX table with per-dataset aggregates for multi-image runs."""
    from collections import defaultdict, Counter

    by_dataset = defaultdict(list)
    for cs in case_studies:
        by_dataset[cs["dataset"]].append(cs)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{TopoAgent consistency and accuracy across 5 MedMNIST datasets "
        r"($n$ images per dataset). Consistency = fraction selecting the same descriptor. "
        r"Regret = Oracle $-$ Agent.}",
        r"\label{tab:case_studies}",
        r"\small",
        r"\begin{tabular}{llclccccc}",
        r"\toprule",
        r"Dataset & Object Type & $n$ & Descriptor & Consist. & Acc.\ (\%) & Oracle (\%) & Regret (\%) & Avg.\ Time (s) \\",
        r"\midrule",
    ]

    all_regrets = []
    for ds_name in [d["name"] for d in DATASETS]:
        studies = by_dataset.get(ds_name, [])
        if not studies:
            continue
        n = len(studies)
        obj_type = studies[0]["object_type"].replace("_", r"\_")
        desc_counts = Counter(cs["output"]["descriptor"] for cs in studies)
        top_desc, top_count = desc_counts.most_common(1)[0]
        consist = f"{top_count}/{n}"

        accs = [cs["output"]["accuracy"] for cs in studies if cs["output"]["accuracy"] is not None]
        acc = f"{accs[0]:.1f}" if accs else "--"

        oracles = [cs["output"]["oracle_accuracy"] for cs in studies if cs["output"]["oracle_accuracy"] is not None]
        oracle = f"{oracles[0]:.1f}" if oracles else "--"

        regrets = [cs["output"]["regret"] for cs in studies if cs["output"]["regret"] is not None]
        regret = f"{np.mean(regrets):.2f}" if regrets else "--"
        all_regrets.extend(regrets)

        times = [cs["metadata"]["total_time_s"] for cs in studies if cs["metadata"]["total_time_s"]]
        avg_time = f"{np.mean(times):.1f}" if times else "--"

        desc_tex = top_desc.replace("_", r"\_")

        lines.append(
            f"  {ds_name.replace('MNIST', '')} & {obj_type} & {n} & {desc_tex} & "
            f"{consist} & {acc} & {oracle} & {regret} & {avg_time} \\\\"
        )

    if all_regrets:
        lines.append(r"\midrule")
        mean_r = f"{np.mean(all_regrets):.2f}"
        lines.append(f"  \\textbf{{Mean}} & & & & & & & {mean_r} & \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_latex_table(case_studies: List[Dict]) -> str:
    """Generate a LaTeX table summarizing all case studies."""

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{TopoAgent case studies across 5 representative MedMNIST datasets. "
        r"The agent selects a descriptor through structured reasoning (2 LLM calls), "
        r"achieving near-oracle accuracy. Regret = Oracle $-$ Agent.}",
        r"\label{tab:case_studies}",
        r"\small",
        r"\begin{tabular}{llllcccc}",
        r"\toprule",
        r"Dataset & Object Type & Reasoning & Descriptor & Acc.\ (\%) & Oracle (\%) & Regret (\%) & Time (s) \\",
        r"\midrule",
    ]

    for cs in case_studies:
        dataset = cs["dataset"].replace("MNIST", "")
        obj_type = cs["object_type"].replace("_", r"\_")
        chain = cs["reasoning"]["reasoning_chain"].replace("_", r"\_")
        desc = cs["output"]["descriptor"].replace("_", r"\_")
        acc = f'{cs["output"]["accuracy"]:.1f}' if cs["output"]["accuracy"] is not None else "--"
        oracle = f'{cs["output"]["oracle_accuracy"]:.1f}' if cs["output"]["oracle_accuracy"] is not None else "--"
        regret = f'{cs["output"]["regret"]:.2f}' if cs["output"]["regret"] is not None else "--"
        time_s = f'{cs["metadata"]["total_time_s"]:.1f}' if cs["metadata"]["total_time_s"] else "--"

        lines.append(
            f"  {dataset} & {obj_type} & {chain} & {desc} & {acc} & {oracle} & {regret} & {time_s} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


def generate_summary_table(case_studies: List[Dict]) -> str:
    """Generate a plaintext summary table."""
    header = f"{'Dataset':<14} {'#':>3} {'Object Type':<17} {'Chain':<18} {'Descriptor':<24} {'Acc%':>6} {'Oracle%':>8} {'Regret%':>8} {'Time':>6} {'LLM':>4} {'Retry':>5}"
    sep = "-" * len(header)
    rows = [header, sep]

    for cs in case_studies:
        acc = f'{cs["output"]["accuracy"]:.1f}' if cs["output"]["accuracy"] is not None else "--"
        oracle = f'{cs["output"]["oracle_accuracy"]:.1f}' if cs["output"]["oracle_accuracy"] is not None else "--"
        regret = f'{cs["output"]["regret"]:.2f}' if cs["output"]["regret"] is not None else "--"
        time_s = f'{cs["metadata"]["total_time_s"]:.1f}' if cs["metadata"]["total_time_s"] else "--"
        retry = "Yes" if cs["metadata"]["retry_used"] else "No"
        sample_idx = cs.get("sample_index", 0)

        rows.append(
            f'{cs["dataset"]:<14} {sample_idx:>3} {cs["object_type"]:<17} '
            f'{cs["reasoning"]["reasoning_chain"]:<18} '
            f'{cs["output"]["descriptor"]:<24} '
            f'{acc:>6} {oracle:>8} {regret:>8} {time_s:>6} '
            f'{cs["metadata"]["n_llm_calls"]:>4} {retry:>5}'
        )

    # Mean regret
    regrets = [cs["output"]["regret"] for cs in case_studies if cs["output"]["regret"] is not None]
    if regrets:
        rows.append(sep)
        rows.append(f"{'Mean regret:':<84} {np.mean(regrets):>8.2f}")

    return "\n".join(rows)


def generate_aggregate_table(case_studies: List[Dict]) -> str:
    """Generate a per-dataset aggregate summary for multi-image runs."""
    from collections import defaultdict, Counter

    by_dataset = defaultdict(list)
    for cs in case_studies:
        by_dataset[cs["dataset"]].append(cs)

    header = (f"{'Dataset':<14} {'N':>3} {'Object Type':<17} {'Top Descriptor':<24} "
              f"{'Consist':>7} {'Acc%':>6} {'Oracle%':>8} {'Regret%':>8} {'AvgTime':>8} {'Retries':>7}")
    sep = "-" * len(header)
    rows = [header, sep]

    all_regrets = []
    for ds_name, studies in by_dataset.items():
        n = len(studies)
        obj_type = studies[0]["object_type"]

        # Most common descriptor
        desc_counts = Counter(cs["output"]["descriptor"] for cs in studies)
        top_desc, top_count = desc_counts.most_common(1)[0]
        consistency = f"{top_count}/{n}"

        # Accuracy (all same descriptor → same accuracy from lookup)
        accs = [cs["output"]["accuracy"] for cs in studies if cs["output"]["accuracy"] is not None]
        acc_str = f"{accs[0]:.1f}" if accs else "--"

        oracles = [cs["output"]["oracle_accuracy"] for cs in studies if cs["output"]["oracle_accuracy"] is not None]
        oracle_str = f"{oracles[0]:.1f}" if oracles else "--"

        regrets = [cs["output"]["regret"] for cs in studies if cs["output"]["regret"] is not None]
        regret_str = f"{np.mean(regrets):.2f}" if regrets else "--"
        all_regrets.extend(regrets)

        times = [cs["metadata"]["total_time_s"] for cs in studies if cs["metadata"]["total_time_s"]]
        avg_time = f"{np.mean(times):.1f}" if times else "--"

        retries = sum(1 for cs in studies if cs["metadata"]["retry_used"])
        retry_str = f"{retries}/{n}"

        rows.append(
            f"{ds_name:<14} {n:>3} {obj_type:<17} {top_desc:<24} "
            f"{consistency:>7} {acc_str:>6} {oracle_str:>8} {regret_str:>8} {avg_time:>8} {retry_str:>7}"
        )

    if all_regrets:
        rows.append(sep)
        rows.append(f"{'Mean regret across all:':<77} {np.mean(all_regrets):>8.2f}")

    return "\n".join(rows)


# =============================================================================
# Main
# =============================================================================

def run_case_studies(
    datasets: Optional[List[Dict]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    n_samples: int = 1,
) -> List[Dict]:
    """Run case study generation for specified datasets.

    Args:
        datasets: List of dataset configs (default: all 5)
        output_dir: Where to save results
        verbose: Print full LLM prompts/responses
        n_samples: Number of images per dataset
    """
    from topoagent.agent import create_topoagent
    from topoagent.workflow import TopoAgentWorkflow

    if datasets is None:
        datasets = DATASETS
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "case_studies"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Load accuracy lookup
    accuracy_lookup = load_accuracy_lookup()
    if not accuracy_lookup:
        print("WARNING: No accuracy_lookup.json found. Accuracy/regret will be unavailable.")

    # Create agent
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    agent = create_topoagent(
        workflow_version="v4",
        skills_mode=True,
        temperature=0.3,
    )

    # Rebuild workflow with verbose flag
    agent.workflow = TopoAgentWorkflow(
        model=agent.model,
        tools=agent.tools,
        max_rounds=4,
        skills_mode=True,
        verbose=verbose,
    )

    print(f"Agent created with {len(agent.tools)} tools")
    print(f"Output directory: {output_dir}")
    print(f"Images per dataset: {n_samples}")
    total_runs = len(datasets) * n_samples
    print(f"Total agent runs: {total_runs} (~{total_runs * 2} LLM calls)")

    case_studies = []
    global_start = time.time()

    for ds_config in datasets:
        ds_name = ds_config["name"]
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name} (object_type={ds_config['object_type']}, n={n_samples})")
        print(f"{'='*70}")

        # Save sample images
        try:
            image_samples = save_sample_images(ds_name, image_dir, n_samples=n_samples)
            print(f"  Saved {len(image_samples)} images")
        except Exception as e:
            print(f"  ERROR saving sample images: {e}")
            continue

        query = build_query(ds_name)

        for sample_idx, (image_path, label) in enumerate(image_samples):
            progress = len(case_studies) + 1
            print(f"\n  --- Sample {sample_idx+1}/{n_samples} (global {progress}/{total_runs}) "
                  f"label={label} ---")

            start = time.time()
            try:
                final_state = agent.workflow.invoke(query=query, image_path=image_path)
                elapsed = time.time() - start
            except Exception as e:
                print(f"    ERROR running agent: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Extract case study
            try:
                case_study = extract_case_study(
                    dataset_name=ds_name,
                    ds_config=ds_config,
                    final_state=final_state,
                    query=query,
                    accuracy_lookup=accuracy_lookup,
                )
                case_study["sample_index"] = sample_idx
                case_study["ground_truth_label"] = label
                case_studies.append(case_study)

                # Print compact summary
                desc = case_study["output"]["descriptor"]
                chain = case_study["reasoning"]["reasoning_chain"]
                retry = "RETRY" if case_study["metadata"]["retry_used"] else ""
                t = case_study["metadata"]["total_time_s"]
                print(f"    {desc} ({chain}) {elapsed:.1f}s {retry}")

            except Exception as e:
                print(f"    ERROR extracting case study: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Per-dataset mini-summary
        ds_studies = [cs for cs in case_studies if cs["dataset"] == ds_name]
        if ds_studies:
            from collections import Counter
            desc_counts = Counter(cs["output"]["descriptor"] for cs in ds_studies)
            top_desc, top_count = desc_counts.most_common(1)[0]
            print(f"\n  {ds_name} summary: {top_desc} chosen {top_count}/{len(ds_studies)}")

    global_elapsed = time.time() - global_start

    if not case_studies:
        print("\nNo case studies generated!")
        return []

    # Save combined results
    combined_path = output_dir / "all_case_studies.json"
    with open(combined_path, "w") as f:
        json.dump(case_studies, f, indent=2, default=str)
    print(f"\nSaved combined: {combined_path}")

    # Generate and save per-image summary table
    summary = generate_summary_table(case_studies)
    summary_path = output_dir / "summary_table.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved per-image summary: {summary_path}")

    # Generate and save aggregate table (per-dataset)
    if n_samples > 1:
        agg = generate_aggregate_table(case_studies)
        print(f"\n{agg}")
        agg_path = output_dir / "aggregate_table.txt"
        with open(agg_path, "w") as f:
            f.write(agg)
        print(f"\nSaved aggregate: {agg_path}")
    else:
        print(f"\n{summary}")

    # Generate and save LaTeX table (aggregate for multi-image, per-image for single)
    if n_samples > 1:
        latex = generate_latex_aggregate_table(case_studies)
    else:
        latex = generate_latex_table(case_studies)
    latex_path = output_dir / "case_studies_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")

    print(f"\nTotal time: {global_elapsed:.0f}s ({global_elapsed/60:.1f} min) for {len(case_studies)} runs")

    return case_studies


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready case studies")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only for this dataset (e.g., BloodMNIST)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results/case_studies/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full LLM prompts and responses")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of images per dataset (default: 1)")
    args = parser.parse_args()

    # Filter datasets
    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["name"] == args.dataset]
        if not datasets:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            print(f"Available: {[d['name'] for d in DATASETS]}")
            sys.exit(1)

    output_dir = Path(args.output) if args.output else None

    run_case_studies(
        datasets=datasets,
        output_dir=output_dir,
        verbose=args.verbose,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
