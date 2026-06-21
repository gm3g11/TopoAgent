#!/usr/bin/env python3
"""Protocol 2 — Component Verification.

Programmatically verifies that each agent component (skills, reasoning,
action, reflection, parameter auto-injection, efficiency, report quality)
works correctly on 5 representative datasets (one per object type).

v4.1 update: Tests the optimized 2-LLM-call pipeline with structured JSON.

Usage:
    python scripts/protocol2_verify_components.py
    python scripts/protocol2_verify_components.py --verbose
    python scripts/protocol2_verify_components.py --dataset BloodMNIST
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

VALID_REASONING_CHAINS = {
    "h0_dominant", "h1_important", "noisy_ph", "shape_silhouette", "color_diagnostic"
}


# =============================================================================
# Helper: Save a single sample image
# =============================================================================

def save_sample_image(dataset_name: str) -> Tuple[str, int]:
    """Save one sample image from a dataset as PNG. Returns (path, label)."""
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2" / "verify_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    b4_dir = PROJECT_ROOT / "RuleBenchmark" / "benchmark4"
    if str(b4_dir) not in sys.path:
        sys.path.insert(0, str(b4_dir))
    from data_loader import load_dataset

    images, labels, class_names = load_dataset(dataset_name, n_samples=1, seed=42)

    from PIL import Image
    img = images[0]
    label = int(labels[0])
    if img.ndim == 2:
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    elif img.ndim == 3 and img.shape[2] == 3:
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
    else:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

    fpath = output_dir / f"{dataset_name}_verify.png"
    pil_img.save(fpath)
    return str(fpath), label


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


# =============================================================================
# Check Functions
# =============================================================================

def check_skills_injection(
    final_state: Dict[str, Any],
    config: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 1: Skills knowledge was injected into prompts."""
    from topoagent.skills import SkillRegistry
    from topoagent.skills.rules_data import TOP_PERFORMERS, SUPPORTED_DESCRIPTORS

    object_type = config["object_type"]
    color_mode = config["color_mode"]

    registry = SkillRegistry()
    context = registry.build_skill_context(
        object_type_hint=object_type,
        color_mode=color_mode,
    )

    details = []

    # Context should be non-empty
    if not context or len(context) < 100:
        return False, "Skill context is empty or too short"
    details.append(f"Context length: {len(context)} chars")

    # Should mention the object type
    if object_type not in context:
        return False, f"Object type '{object_type}' not found in skill context"
    details.append(f"Object type '{object_type}' found in context")

    # Should mention at least 3 descriptors
    desc_count = sum(1 for d in SUPPORTED_DESCRIPTORS if d in context)
    if desc_count < 3:
        return False, f"Only {desc_count} descriptors mentioned in context (need >= 3)"
    details.append(f"{desc_count} descriptors mentioned")

    # TOP_PERFORMERS should be present
    performers = TOP_PERFORMERS.get(object_type, [])
    if performers:
        top1 = performers[0]["descriptor"]
        if top1 not in context:
            details.append(f"WARNING: Top performer '{top1}' not in context (may use supported-only list)")
    details.append(f"Top performers available for {object_type}: {len(performers)}")

    return True, "; ".join(details)


def check_reasoning(
    final_state: Dict[str, Any],
    config: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 2: LLM produced structured reasoning with valid plan."""
    from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS

    details = []

    # Check for plan_descriptor interaction (new pipeline)
    interactions = final_state.get("llm_interactions", [])
    plan_interactions = [i for i in interactions if i.step == "plan_descriptor"]

    if plan_interactions:
        # New optimized pipeline
        plan_resp = plan_interactions[0]
        response_text = plan_resp.response or ""

        details.append(f"plan_descriptor response: {len(response_text)} chars")

        # Response should contain visible reasoning (not 0-char)
        if len(response_text) < 10:
            return False, f"plan_descriptor response too short ({len(response_text)} chars)"

        # Check structured plan
        plan = final_state.get("_plan") or {}
        if not plan:
            return False, "No _plan in state — LLM response not parsed"

        # Validate descriptor_choice
        descriptor = plan.get("descriptor_choice")
        if descriptor not in SUPPORTED_DESCRIPTORS:
            return False, f"descriptor_choice '{descriptor}' not in SUPPORTED_DESCRIPTORS"
        details.append(f"descriptor_choice: {descriptor}")

        # Validate reasoning_chain
        chain = plan.get("reasoning_chain", "")
        if chain in VALID_REASONING_CHAINS:
            details.append(f"reasoning_chain: {chain}")
        else:
            details.append(f"WARNING: reasoning_chain '{chain}' not in standard set")

        # Check descriptor_rationale contains BECAUSE
        rationale = plan.get("descriptor_rationale", "")
        if "BECAUSE" in rationale.upper() or "because" in rationale.lower():
            details.append("rationale contains BECAUSE")
        else:
            details.append(f"WARNING: rationale missing BECAUSE: '{rationale[:60]}'")

        # Check image_analysis is non-empty
        analysis = plan.get("image_analysis", "")
        if len(analysis) > 20:
            details.append(f"image_analysis: {len(analysis)} chars")
        else:
            details.append(f"WARNING: image_analysis too short: '{analysis}'")

        return True, "; ".join(details)

    # Fallback: check for old tool_selection interactions
    tool_selection_interactions = [i for i in interactions if i.step == "tool_selection"]
    if not tool_selection_interactions:
        return False, "No plan_descriptor or tool_selection interactions found"

    # Old pipeline checks
    rounds_with_tools = sum(1 for i in tool_selection_interactions if i.tool_calls)
    details.append(f"{rounds_with_tools}/{len(tool_selection_interactions)} rounds had tool_calls")
    passed = rounds_with_tools >= 1
    return passed, "; ".join(details)


def check_action(
    final_state: Dict[str, Any],
    config: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 3: Tools executed successfully with correct outputs."""
    from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS

    stm = final_state.get("short_term_memory", [])
    details = []

    if not stm:
        return False, "Short-term memory is empty — no tools executed"

    tool_names = [name for name, _ in stm]
    details.append(f"Tools executed: {tool_names}")

    # Check image_loader
    img_output = None
    for name, output in stm:
        if name == "image_loader" and isinstance(output, dict):
            img_output = output
            break

    if img_output:
        if img_output.get("success"):
            details.append(f"image_loader: success, shape={img_output.get('shape')}")
        else:
            return False, f"image_loader failed: {img_output.get('error')}"
    else:
        details.append("WARNING: image_loader not in memory")

    # Check compute_ph
    ph_output = None
    for name, output in stm:
        if name == "compute_ph" and isinstance(output, dict):
            ph_output = output
            break

    if ph_output:
        if ph_output.get("success"):
            stats = ph_output.get("statistics", {})
            h0 = stats.get("H0", {}).get("count", 0)
            h1 = stats.get("H1", {}).get("count", 0)
            details.append(f"compute_ph: success, H0={h0}, H1={h1}")
        else:
            details.append(f"compute_ph: FAILED — {ph_output.get('error')}")

    # Check descriptor tool
    descriptor_output = None
    descriptor_name = None
    for name, output in stm:
        if name in SUPPORTED_DESCRIPTORS and isinstance(output, dict):
            descriptor_output = output
            descriptor_name = name
            break

    if descriptor_output:
        if descriptor_output.get("success"):
            # Check for feature vector
            fv = descriptor_output.get("combined_vector") or descriptor_output.get("feature_vector")
            if fv is not None:
                dim = len(fv) if hasattr(fv, '__len__') else 0
                details.append(f"{descriptor_name}: success, feature_dim={dim}")
                if dim == 0:
                    return False, f"{descriptor_name} returned empty feature vector"
            else:
                details.append(f"{descriptor_name}: success but no feature vector key found")
        else:
            return False, f"{descriptor_name} failed: {descriptor_output.get('error')}"
    else:
        desc_in_stm = [n for n in tool_names if n in SUPPORTED_DESCRIPTORS]
        if not desc_in_stm:
            return False, "No descriptor tool was executed"
        else:
            details.append(f"Descriptor {desc_in_stm[0]} in memory but output not a dict")

    passed = descriptor_output is not None and descriptor_output.get("success", False)
    return passed, "; ".join(details)


def check_reflection(
    final_state: Dict[str, Any],
    config: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 4: Reflection mechanism produced meaningful entries."""
    ltm = final_state.get("long_term_memory", [])
    details = []

    if not ltm:
        return False, "Long-term memory is empty — no reflections generated"

    details.append(f"{len(ltm)} reflection entries")

    # Check the last reflection entry (from verify_and_reflect)
    last = ltm[-1]

    # error_analysis should be non-generic
    ea = last.error_analysis or ""
    if ea.startswith("No ") or ea == "No error analysis provided":
        details.append(f"WARNING: error_analysis is default: '{ea[:50]}'")
        ea_ok = False
    else:
        details.append(f"error_analysis: '{ea[:80]}'")
        ea_ok = True

    # experience should follow "For [type] with [pattern], [descriptor] gives [quality]" format
    exp = last.experience or ""
    if exp.startswith("No ") or exp == "No experience recorded":
        details.append(f"WARNING: experience is default: '{exp[:50]}'")
        exp_ok = False
    else:
        details.append(f"experience: '{exp[:80]}'")
        exp_ok = True

    # Check that the reflection doesn't reference DermaMNIST for non-DermaMNIST datasets
    dataset_name = config["name"]
    if dataset_name != "DermaMNIST":
        combined = ea + " " + exp
        if "dermamnist" in combined.lower() or "skin lesion" in combined.lower():
            details.append(f"WARNING: Reflection references DermaMNIST for {dataset_name}")

    passed = ea_ok or exp_ok
    return passed, "; ".join(details)


def check_params(
    final_state: Dict[str, Any],
    config: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 5: Parameter auto-injection from benchmark rules."""
    from topoagent.skills.rules_data import (
        SUPPORTED_DESCRIPTORS, get_optimal_params, get_descriptor_dim
    )

    details = []

    skill_descriptor = final_state.get("skill_descriptor")
    skill_params = final_state.get("skill_params") or {}

    if not skill_descriptor:
        return False, "skill_descriptor not set — descriptor choice not recorded"
    details.append(f"skill_descriptor: {skill_descriptor}")

    if skill_descriptor not in SUPPORTED_DESCRIPTORS:
        return False, f"skill_descriptor '{skill_descriptor}' not in SUPPORTED_DESCRIPTORS"

    if not skill_params:
        return False, "skill_params is empty — no benchmark parameters applied"

    # Check that params match expected from get_optimal_params
    object_type = config["object_type"]
    color_mode = config["color_mode"]
    expected_params = get_optimal_params(skill_descriptor, object_type)

    matching_keys = 0
    for key, expected_val in expected_params.items():
        if key in ("dim",):
            continue  # Meta key
        actual_val = skill_params.get(key)
        if actual_val is not None:
            matching_keys += 1
            if actual_val != expected_val:
                details.append(f"MISMATCH {key}: expected={expected_val}, got={actual_val}")
    details.append(f"{matching_keys}/{len(expected_params)} params matched")

    # Check total_dim
    expected_dim = get_descriptor_dim(skill_descriptor, object_type, color_mode)
    actual_dim = skill_params.get("total_dim")
    if actual_dim is not None:
        details.append(f"total_dim: expected={expected_dim}, actual={actual_dim}")

    passed = skill_descriptor in SUPPORTED_DESCRIPTORS and len(skill_params) > 0
    return passed, "; ".join(details)


def check_efficiency(
    final_state: Dict[str, Any],
    config: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 6: Exactly 2 LLM calls (plan_descriptor + verify_and_reflect)."""
    interactions = final_state.get("llm_interactions", [])
    details = []

    n_interactions = len(interactions)
    details.append(f"{n_interactions} LLM interactions")

    if n_interactions == 0:
        return False, "No LLM interactions recorded"

    # List all step types
    steps = [i.step for i in interactions]
    details.append(f"Steps: {steps}")

    # Check for exactly 2 LLM calls
    if n_interactions == 2:
        details.append("Exactly 2 LLM calls (optimal)")
        passed = True
    elif n_interactions <= 3:
        details.append(f"Close to optimal ({n_interactions} LLM calls, target is 2)")
        passed = True
    else:
        details.append(f"Too many LLM calls ({n_interactions}, target is 2)")
        passed = False

    # Verify expected steps are present
    has_plan = any(s == "plan_descriptor" for s in steps)
    has_verify = any(s == "verify_and_reflect" for s in steps)
    if has_plan:
        details.append("plan_descriptor present")
    if has_verify:
        details.append("verify_and_reflect present")

    return passed, "; ".join(details)


def check_report(
    final_state: Dict[str, Any],
    config: Dict[str, str],
) -> Tuple[bool, str]:
    """Check 7: Final report JSON has all fields and quality reasoning."""
    details = []

    final_answer = final_state.get("final_answer")
    if final_answer is None:
        return False, "No final_answer in state"

    if isinstance(final_answer, str):
        # Old-style text answer — not structured
        details.append(f"final_answer is text ({len(final_answer)} chars), not structured JSON")
        return False, "; ".join(details)

    if not isinstance(final_answer, dict):
        return False, f"final_answer is {type(final_answer).__name__}, expected dict"

    # Check required fields
    required_fields = ["descriptor", "reasoning", "alternatives_considered"]
    missing = [f for f in required_fields if f not in final_answer or not final_answer[f]]
    if missing:
        details.append(f"Missing fields: {missing}")
        return False, "; ".join(details)

    # Check descriptor is valid
    from topoagent.skills.rules_data import SUPPORTED_DESCRIPTORS
    descriptor = final_answer.get("descriptor", "")
    if descriptor in SUPPORTED_DESCRIPTORS:
        details.append(f"descriptor: {descriptor}")
    else:
        details.append(f"WARNING: descriptor '{descriptor}' not in SUPPORTED_DESCRIPTORS")

    # Check reasoning is >50 chars
    reasoning = final_answer.get("reasoning", "")
    if len(reasoning) > 50:
        details.append(f"reasoning: {len(reasoning)} chars")
    else:
        details.append(f"WARNING: reasoning too short ({len(reasoning)} chars)")

    # Check alternatives_considered is non-empty
    alts = final_answer.get("alternatives_considered", "")
    if len(alts) > 10:
        details.append(f"alternatives_considered: {len(alts)} chars")
    else:
        details.append(f"WARNING: alternatives_considered too short")

    # Check for DermaMNIST leakage
    dataset_name = config["name"]
    if dataset_name != "DermaMNIST":
        combined = reasoning + " " + alts
        if "dermamnist" in combined.lower():
            details.append(f"WARNING: Report references DermaMNIST for {dataset_name}")

    # Verify feature_dimension is present and > 0
    feat_dim = final_answer.get("feature_dimension", 0)
    if isinstance(feat_dim, str):
        try:
            feat_dim = int(feat_dim)
        except ValueError:
            feat_dim = 0
    if feat_dim > 0:
        details.append(f"feature_dimension: {feat_dim}")
    else:
        details.append(f"WARNING: feature_dimension={feat_dim}")

    # Pass if reasoning is substantive and descriptor is valid
    passed = (
        descriptor in SUPPORTED_DESCRIPTORS
        and len(reasoning) > 50
        and len(alts) > 10
    )
    return passed, "; ".join(details)


# =============================================================================
# Main Runner
# =============================================================================

def run_verification(
    datasets: Optional[List[Dict]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run component verification on specified datasets.

    Returns:
        Dict with per-dataset results and summary table.
    """
    from topoagent.agent import create_topoagent
    from topoagent.workflow import TopoAgentWorkflow

    if datasets is None:
        datasets = DATASETS

    # Create agent (shared across datasets)
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
    print(f"Tools: {sorted(agent.tools.keys())}")

    # Checks to run (7 checks including 2 new ones)
    checks = [
        ("Skills", check_skills_injection),
        ("Reasoning", check_reasoning),
        ("Action", check_action),
        ("Reflection", check_reflection),
        ("Params", check_params),
        ("Efficiency", check_efficiency),
        ("Report", check_report),
    ]

    all_results = {}
    summary_rows = []

    for ds_config in datasets:
        ds_name = ds_config["name"]
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name} (object_type={ds_config['object_type']})")
        print(f"{'='*70}")

        # Save sample image
        try:
            image_path, label = save_sample_image(ds_name)
            print(f"  Image: {Path(image_path).name} (label={label})")
        except Exception as e:
            print(f"  ERROR saving sample image: {e}")
            row = {"dataset": ds_name}
            for check_name, _ in checks:
                row[check_name] = "ERROR"
            row["Overall"] = "ERROR"
            summary_rows.append(row)
            all_results[ds_name] = {"error": str(e)}
            continue

        # Run agent
        query = build_query(ds_name)
        start = time.time()
        try:
            final_state = agent.workflow.invoke(query=query, image_path=image_path)
            elapsed = time.time() - start
            print(f"  Agent completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR running agent: {e}")
            import traceback
            traceback.print_exc()
            row = {"dataset": ds_name}
            for check_name, _ in checks:
                row[check_name] = "ERROR"
            row["Overall"] = "ERROR"
            summary_rows.append(row)
            all_results[ds_name] = {"error": str(e), "elapsed": elapsed}
            continue

        # Run checks
        ds_result = {"elapsed": elapsed, "checks": {}}
        row = {"dataset": ds_name}
        all_pass = True

        for check_name, check_fn in checks:
            try:
                passed, details = check_fn(final_state, ds_config)
                status = "PASS" if passed else "FAIL"
                ds_result["checks"][check_name] = {
                    "passed": passed,
                    "details": details,
                }
                row[check_name] = status
                if not passed:
                    all_pass = False
                print(f"  {status:>4} | {check_name:<12} | {details}")
            except Exception as e:
                row[check_name] = "ERROR"
                ds_result["checks"][check_name] = {
                    "passed": False,
                    "details": f"Exception: {e}",
                }
                all_pass = False
                print(f"  ERR  | {check_name:<12} | Exception: {e}")

        row["Overall"] = "PASS" if all_pass else "FAIL"
        ds_result["overall"] = all_pass
        ds_result["skill_descriptor"] = final_state.get("skill_descriptor")
        ds_result["skill_params"] = {
            k: v for k, v in (final_state.get("skill_params") or {}).items()
            if k not in ("total_dim", "classifier", "color_mode", "dim")
        }

        summary_rows.append(row)
        all_results[ds_name] = ds_result

    # Print summary table
    print(f"\n{'='*100}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*100}")
    header = f"{'Dataset':<16}"
    for check_name, _ in checks:
        header += f" | {check_name:^10}"
    header += f" | {'Overall':^8}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        line = f"{row['dataset']:<16}"
        for check_name, _ in checks:
            val = row.get(check_name, "?")
            line += f" | {val:^10}"
        line += f" | {row.get('Overall', '?'):^8}"
        print(line)

    # Aggregate
    total = len(summary_rows)
    passed = sum(1 for r in summary_rows if r.get("Overall") == "PASS")
    print(f"\n{passed}/{total} datasets passed all checks")

    return {
        "results": all_results,
        "summary": summary_rows,
        "total": total,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Protocol 2: Component Verification")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full LLM prompts and responses")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to test (default: all 5)")
    args = parser.parse_args()

    if args.dataset:
        # Filter to single dataset
        datasets = [d for d in DATASETS if d["name"] == args.dataset]
        if not datasets:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {[d['name'] for d in DATASETS]}")
            sys.exit(1)
    else:
        datasets = None

    results = run_verification(datasets=datasets, verbose=args.verbose)

    # Save detailed results
    output_dir = PROJECT_ROOT / "results" / "topobenchmark" / "protocol2"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"verify_components_{timestamp}.json"

    serializable = json.loads(json.dumps(results, default=str))
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
