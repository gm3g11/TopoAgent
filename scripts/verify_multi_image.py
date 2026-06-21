#!/usr/bin/env python3
"""Multi-image verification of TopoAgent optimized pipeline.

Runs the agent on N images per dataset and verifies output quality.

Usage:
    python scripts/verify_multi_image.py --n-images 10
    python scripts/verify_multi_image.py --n-images 10 --dataset BloodMNIST
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

DATASETS = [
    {"name": "BloodMNIST", "object_type": "discrete_cells", "color_mode": "per_channel"},
    {"name": "PathMNIST", "object_type": "glands_lumens", "color_mode": "per_channel"},
    {"name": "RetinaMNIST", "object_type": "vessel_trees", "color_mode": "per_channel"},
    {"name": "DermaMNIST", "object_type": "surface_lesions", "color_mode": "per_channel"},
    {"name": "OrganAMNIST", "object_type": "organ_shape", "color_mode": "grayscale"},
]

VALID_REASONING_CHAINS = {
    "h0_dominant", "h1_important", "noisy_ph", "shape_silhouette", "color_diagnostic"
}


def save_sample_images(dataset_name, n_images=10):
    """Save N sample images. Returns list of (path, label)."""
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2" / "verify_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    b4_dir = PROJECT_ROOT / "RuleBenchmark" / "benchmark4"
    if str(b4_dir) not in sys.path:
        sys.path.insert(0, str(b4_dir))
    from data_loader import load_dataset
    from PIL import Image

    images, labels, class_names = load_dataset(dataset_name, n_samples=n_images, seed=42)
    results = []
    for i in range(len(images)):
        img = images[i]
        label = int(labels[i])
        if img.ndim == 2:
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        elif img.ndim == 3 and img.shape[2] == 3:
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
        else:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
        fpath = output_dir / f"{dataset_name}_verify_{i}.png"
        pil_img.save(fpath)
        results.append((str(fpath), label))
    return results


def build_query(dataset_name):
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


def verify_single_output(state, config):
    """Verify a single agent run. Returns dict of check results."""
    from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS

    checks = {}

    # 1. LLM call count
    interactions = state.get("llm_interactions", [])
    n_llm = len(interactions)
    checks["n_llm_calls"] = n_llm
    checks["efficiency_ok"] = n_llm <= 3

    # 2. Plan quality
    plan = state.get("_plan") or {}
    checks["has_plan"] = bool(plan)
    checks["descriptor_choice"] = plan.get("descriptor_choice", "NONE")
    checks["descriptor_valid"] = plan.get("descriptor_choice") in SUPPORTED_DESCRIPTORS
    checks["reasoning_chain"] = plan.get("reasoning_chain", "NONE")
    checks["chain_valid"] = plan.get("reasoning_chain") in VALID_REASONING_CHAINS

    rationale = plan.get("descriptor_rationale", "")
    checks["has_because"] = "because" in rationale.lower()
    checks["image_analysis_len"] = len(plan.get("image_analysis", ""))

    # 3. PH stats
    ph = state.get("_ph_stats") or {}
    checks["h0_count"] = ph.get("H0_count", 0)
    checks["h1_count"] = ph.get("H1_count", 0)
    checks["h0_max_pers"] = ph.get("H0_max_persistence", 0)
    checks["h1_max_pers"] = ph.get("H1_max_persistence", 0)
    checks["ph_ok"] = checks["h0_count"] > 0

    # 4. Execution success
    stm = state.get("short_term_memory", [])
    tool_names = [n for n, _ in stm]
    checks["tools_run"] = tool_names

    desc_outputs = [(n, o) for n, o in stm if n in SUPPORTED_DESCRIPTORS]
    if desc_outputs:
        desc_name, desc_out = desc_outputs[0]
        checks["descriptor_executed"] = desc_name
        checks["descriptor_success"] = desc_out.get("success", False) if isinstance(desc_out, dict) else False
        fv = desc_out.get("combined_vector") or desc_out.get("feature_vector") if isinstance(desc_out, dict) else None
        checks["feature_dim"] = len(fv) if fv is not None and hasattr(fv, '__len__') else 0
    else:
        checks["descriptor_executed"] = "NONE"
        checks["descriptor_success"] = False
        checks["feature_dim"] = 0

    # 5. Feature quality
    fq = state.get("_feature_quality") or {}
    checks["sparsity"] = fq.get("sparsity", -1)
    checks["variance"] = fq.get("variance", -1)
    checks["nan_count"] = fq.get("nan_count", -1)

    # 6. Reflection quality
    ltm = state.get("long_term_memory", [])
    if ltm:
        last = ltm[-1]
        ea = last.error_analysis or ""
        exp = last.experience or ""
        checks["reflection_error_analysis"] = ea[:100]
        checks["reflection_experience"] = exp[:100]
        checks["reflection_generic"] = ea.startswith("No ") and exp.startswith("No ")
    else:
        checks["reflection_error_analysis"] = ""
        checks["reflection_experience"] = ""
        checks["reflection_generic"] = True

    # 7. Report quality
    fa = state.get("final_answer")
    if isinstance(fa, dict):
        checks["report_structured"] = True
        checks["report_reasoning_len"] = len(fa.get("reasoning", ""))
        checks["report_alts_len"] = len(fa.get("alternatives_considered", ""))
        checks["report_descriptor"] = fa.get("descriptor", "NONE")
    else:
        checks["report_structured"] = False
        checks["report_reasoning_len"] = 0
        checks["report_alts_len"] = 0
        checks["report_descriptor"] = "NONE"

    # Overall pass
    checks["overall_pass"] = (
        checks["efficiency_ok"]
        and checks["descriptor_valid"]
        and checks["descriptor_success"]
        and checks["feature_dim"] > 0
        and checks["report_structured"]
        and checks["report_reasoning_len"] > 50
        and not checks["reflection_generic"]
    )

    return checks


def run_multi_image(datasets=None, n_images=10):
    from topoagent.agent import create_topoagent
    from topoagent.workflow import TopoAgentWorkflow

    if datasets is None:
        datasets = DATASETS

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    agent = create_topoagent(
        workflow_version="v4",
        skills_mode=True,
        temperature=0.3,
    )
    agent.workflow = TopoAgentWorkflow(
        model=agent.model,
        tools=agent.tools,
        max_rounds=4,
        skills_mode=True,
        verbose=False,
    )
    print(f"Agent: {len(agent.tools)} tools")

    all_results = {}
    grand_total = 0
    grand_pass = 0

    for ds_config in datasets:
        ds_name = ds_config["name"]
        print(f"\n{'='*70}")
        print(f"  {ds_name} ({ds_config['object_type']}) — {n_images} images")
        print(f"{'='*70}")

        # Save images
        try:
            samples = save_sample_images(ds_name, n_images)
            print(f"  Saved {len(samples)} images")
        except Exception as e:
            print(f"  ERROR saving images: {e}")
            all_results[ds_name] = {"error": str(e)}
            continue

        query = build_query(ds_name)
        ds_checks = []
        ds_times = []

        for idx, (img_path, label) in enumerate(samples):
            t0 = time.time()
            try:
                state = agent.workflow.invoke(query=query, image_path=img_path)
                elapsed = time.time() - t0
                checks = verify_single_output(state, ds_config)
                checks["label"] = label
                checks["elapsed"] = round(elapsed, 1)
                ds_checks.append(checks)
                ds_times.append(elapsed)

                status = "OK" if checks["overall_pass"] else "FAIL"
                desc = checks["descriptor_choice"]
                dim = checks["feature_dim"]
                n_llm = checks["n_llm_calls"]
                print(f"  [{idx+1:2d}/{n_images}] {status} | {desc:<24} | dim={dim:4d} | LLM={n_llm} | {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{idx+1:2d}/{n_images}] ERROR | {str(e)[:60]} | {elapsed:.1f}s")
                ds_checks.append({
                    "overall_pass": False,
                    "error": str(e),
                    "elapsed": round(elapsed, 1),
                    "label": label,
                })
                ds_times.append(elapsed)

        # Summarize dataset
        n_pass = sum(1 for c in ds_checks if c.get("overall_pass"))
        n_total = len(ds_checks)
        grand_total += n_total
        grand_pass += n_pass

        descriptor_counts = Counter(c.get("descriptor_choice", "?") for c in ds_checks)
        chain_counts = Counter(c.get("reasoning_chain", "?") for c in ds_checks)
        llm_counts = [c.get("n_llm_calls", 0) for c in ds_checks if "n_llm_calls" in c]
        dims = [c.get("feature_dim", 0) for c in ds_checks if c.get("feature_dim", 0) > 0]

        avg_time = np.mean(ds_times) if ds_times else 0

        print(f"\n  --- {ds_name} Summary ---")
        print(f"  Pass rate:    {n_pass}/{n_total}")
        print(f"  Avg time:     {avg_time:.1f}s")
        print(f"  LLM calls:    {Counter(llm_counts)}")
        print(f"  Descriptors:  {dict(descriptor_counts)}")
        print(f"  Chains:       {dict(chain_counts)}")
        if dims:
            print(f"  Feature dims: min={min(dims)}, max={max(dims)}, mean={np.mean(dims):.0f}")

        # Check for DermaMNIST leakage
        if ds_name != "DermaMNIST":
            leaks = 0
            for c in ds_checks:
                ea = c.get("reflection_error_analysis", "").lower()
                exp = c.get("reflection_experience", "").lower()
                if "dermamnist" in ea or "dermamnist" in exp or "skin lesion" in ea:
                    leaks += 1
            if leaks:
                print(f"  WARNING: {leaks} images had DermaMNIST leakage in reflections")

        # Report quality
        structured = sum(1 for c in ds_checks if c.get("report_structured"))
        has_because = sum(1 for c in ds_checks if c.get("has_because"))
        non_generic = sum(1 for c in ds_checks if not c.get("reflection_generic", True))
        print(f"  Structured reports: {structured}/{n_total}")
        print(f"  BECAUSE rationale:  {has_because}/{n_total}")
        print(f"  Non-generic reflect: {non_generic}/{n_total}")

        all_results[ds_name] = {
            "n_pass": n_pass,
            "n_total": n_total,
            "avg_time": round(avg_time, 1),
            "descriptor_counts": dict(descriptor_counts),
            "chain_counts": dict(chain_counts),
            "checks": ds_checks,
        }

    # Grand summary
    print(f"\n{'='*70}")
    print(f"  GRAND SUMMARY: {grand_pass}/{grand_total} passed across {len(datasets)} datasets")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Dataset':<16} | {'Pass':>6} | {'Time':>6} | {'LLM':>4} | {'Descriptor':>24} | {'BECAUSE':>7} | {'Reflect':>7}")
    print("-" * 90)
    for ds_config in datasets:
        ds_name = ds_config["name"]
        r = all_results.get(ds_name, {})
        if "error" in r:
            print(f"{ds_name:<16} | ERROR")
            continue
        n_pass = r["n_pass"]
        n_total = r["n_total"]
        avg_t = r["avg_time"]
        top_desc = max(r["descriptor_counts"], key=r["descriptor_counts"].get) if r["descriptor_counts"] else "?"
        checks = r.get("checks", [])
        avg_llm = np.mean([c.get("n_llm_calls", 0) for c in checks if "n_llm_calls" in c])
        because_pct = sum(1 for c in checks if c.get("has_because")) / max(len(checks), 1) * 100
        reflect_pct = sum(1 for c in checks if not c.get("reflection_generic", True)) / max(len(checks), 1) * 100
        print(f"{ds_name:<16} | {n_pass:>2}/{n_total:<2} | {avg_t:>5.1f}s | {avg_llm:>3.1f} | {top_desc:>24} | {because_pct:>5.0f}% | {reflect_pct:>5.0f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-image TopoAgent verification")
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if args.dataset:
        datasets = [d for d in DATASETS if d["name"] == args.dataset]
        if not datasets:
            print(f"Unknown: {args.dataset}. Available: {[d['name'] for d in DATASETS]}")
            sys.exit(1)
    else:
        datasets = None

    results = run_multi_image(datasets=datasets, n_images=args.n_images)

    # Save
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"verify_multi_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(json.loads(json.dumps(results, default=str)), f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
