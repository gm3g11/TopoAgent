#!/usr/bin/env python3
"""TopoAgent End-to-End Case Study — Explicit Component Trace.

Runs TopoAgent on a single image and traces EVERY component:
  1. Input Prompt
  2. Skills (Expert Knowledge + Benchmark Rankings)
  3. Long-Term Memory (Learned Rules + Past Reflections)
  4. Short-Term Memory (Tool Outputs: image_loader -> compute_ph -> descriptor)
  5. Reasoning (LLM #1: plan_descriptor)
  6. Action (Deterministic tool execution with auto-injected params)
  7. Reflection (LLM #2: verify_and_reflect)
  8. Output (Feature vector as numpy array)
  9. Classifier Evaluation (3-fold CV on n_eval samples)

Usage:
    python scripts/demo_topoagent_e2e.py --dataset BloodMNIST --index 0
    python scripts/demo_topoagent_e2e.py --dataset ISIC2019 --index 0 --no-eval --agentic
    python scripts/demo_topoagent_e2e.py --all --no-eval --agentic --time-limit 60
    python scripts/demo_topoagent_e2e.py --dataset BloodMNIST --v8 --model claude-sonnet-4-20250514
    python scripts/demo_topoagent_e2e.py --dataset BloodMNIST --v8 --model gemini-2.5-pro-preview-06-05
    python scripts/demo_topoagent_e2e.py --dataset BloodMNIST --v9 --model gpt-4o
"""

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# -- Setup -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# -- Dataset helpers -------------------------------------------------------

from topoagent.skills.rules_data import (
    DATASET_TO_OBJECT_TYPE, DATASET_COLOR_MODE, SUPPORTED_DESCRIPTORS,
)
# All descriptor names (PH-based + image-based) for matching in short_term_memory
SUPPORTED_DESCRIPTORS_ALL = set(SUPPORTED_DESCRIPTORS)

# Auto-populated from rules_data.py — all 26 datasets
DATASET_INFO = {
    name: {
        "object_type": DATASET_TO_OBJECT_TYPE[name],
        "color": DATASET_COLOR_MODE.get(name, "grayscale") == "per_channel",
        "color_mode_str": DATASET_COLOR_MODE.get(name, "grayscale"),
    }
    for name in DATASET_TO_OBJECT_TYPE
}

# Rich descriptions for agent queries (imported lazily to avoid circular deps)
_DATASET_DESCRIPTIONS = None

def _get_dataset_descriptions():
    """Lazily import DATASET_DESCRIPTIONS from TopoBenchmark/config.py."""
    global _DATASET_DESCRIPTIONS
    if _DATASET_DESCRIPTIONS is None:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "topobenchmark_config",
                str(PROJECT_ROOT / "TopoBenchmark" / "config.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            # TopoBenchmark/config.py imports from RuleBenchmark/benchmark4/config.py
            sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))
            spec.loader.exec_module(mod)
            sys.path.pop(0)
            _DATASET_DESCRIPTIONS = mod.DATASET_DESCRIPTIONS
        except Exception:
            _DATASET_DESCRIPTIONS = {}
    return _DATASET_DESCRIPTIONS

IMAGE_BASED_DESCRIPTORS = {
    "minkowski_functionals", "euler_characteristic_curve",
    "euler_characteristic_transform", "edge_histogram", "lbp_texture",
}


def load_dataset_unified(dataset_name: str, n_samples: int = 10):
    """Load images from any of the 26 datasets using benchmark4's loader."""
    sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))
    try:
        from data_loader import load_dataset
    finally:
        sys.path.pop(0)

    color_mode = DATASET_INFO[dataset_name]["color_mode_str"]
    images, labels, class_names = load_dataset(
        dataset_name, n_samples=n_samples, seed=42, color_mode=color_mode,
    )
    n_channels = 3 if images.ndim == 4 and images.shape[-1] == 3 else 1
    return images, labels, class_names, n_channels


def save_temp_image(image_array: np.ndarray, dataset_name: str, index: int) -> str:
    """Save image to results/case_studies/images/ and return the path."""
    outdir = PROJECT_ROOT / "results" / "case_studies" / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dataset_name}_sample{index}.png"
    # Handle float32 [0,1] images from benchmark4 loader
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(image_array)
    img.save(str(path))
    return str(path)


def _build_query(dataset_name: str, object_type: str,
                 n_channels: int = 3, description: dict = None) -> str:
    """Build a clean, generic query — the agent must observe and reason.

    The prompt is intentionally generic so the LLM cannot shortcut by
    recognising the dataset name.  Domain context (description, what_matters)
    is still supplied so the agent can reason about the medical domain.
    """
    if description and isinstance(description, dict):
        # Build context string from safe fields only (no object_type/color_mode)
        parts = []
        if "domain" in description:
            parts.append(f"Domain: {description['domain']}")
        if "description" in description:
            parts.append(description["description"])
        if "what_matters" in description:
            parts.append(f"Key features: {description['what_matters']}")
        if "n_classes" in description:
            parts.append(f"{description['n_classes']} classes")
        desc_part = "; ".join(parts)
    elif description and isinstance(description, str):
        desc_part = description
    else:
        desc_part = ""
    context_line = f"\n\nDataset context: {desc_part}" if desc_part else ""
    return (
        f"Analyze the provided medical image, compute its persistent homology, "
        f"and determine the most suitable topology descriptor to produce a "
        f"fixed-length feature vector."
        f"{context_line}"
    )


# -- Serialization helpers --------------------------------------------------

def _serialize_llm_interactions(interactions):
    """Convert LLM interactions to JSON-serializable dicts.

    Each interaction captures a full LLM call: the prompt sent and response
    received, with step label (observe/act/reflect) and any tool calls.
    """
    serialized = []
    for i, inter in enumerate(interactions):
        if isinstance(inter, dict):
            entry = {
                "index": i,
                "step": inter.get("step", "unknown"),
                "round": inter.get("round", 0),
                "prompt": inter.get("prompt", ""),
                "response": inter.get("response", ""),
                "tool_calls": inter.get("tool_calls"),
            }
        else:
            # LLMInteraction dataclass
            entry = {
                "index": i,
                "step": getattr(inter, "step", "unknown"),
                "round": getattr(inter, "round", 0),
                "prompt": getattr(inter, "prompt", ""),
                "response": getattr(inter, "response", ""),
                "tool_calls": getattr(inter, "tool_calls", None),
            }
        serialized.append(entry)
    return serialized


# -- Main Case Study -------------------------------------------------------

def run_case_study(dataset_name: str, sample_index: int,
                   n_eval: int = 200, verbose: bool = True,
                   skip_eval: bool = False, agentic: bool = False,
                   agentic_v8: bool = False,
                   agentic_v9: bool = False,
                   time_limit: float = 60.0,
                   n_samples: int = 1,
                   model_name: str = "gpt-4o"):
    """Run explicit case study, tracing all agent components.

    When n_samples > 1, runs multiple images sequentially with the SAME agent
    to demonstrate within-session learning (REFLECT writes to LTM, PLAN reads it).
    """
    from topoagent.agent import create_topoagent_auto
    from topoagent.skills import SkillRegistry
    from topoagent.memory.skill_memory import SkillMemory

    info = DATASET_INFO[dataset_name]
    object_type = info["object_type"]
    is_rgb = info["color"]

    if agentic_v9:
        mode_label = "AGENTIC_V9"
    elif agentic_v8:
        mode_label = "AGENTIC_V8"
    elif agentic:
        mode_label = "AGENTIC"
    else:
        mode_label = "SKILLS"

    # For multi-image demo, run all images and aggregate results
    if n_samples > 1 and (agentic_v9 or agentic_v8):
        return _run_multi_image_case_study(
            dataset_name, n_samples=n_samples, n_eval=n_eval,
            skip_eval=skip_eval, time_limit=time_limit,
            model_name=model_name, agentic_v9=agentic_v9,
        )

    print("=" * 80)
    print(f"  TopoAgent Case Study [{mode_label}]: {dataset_name} (sample #{sample_index})")
    print("=" * 80)

    # == STEP 0: Load sample image =========================================
    print("\n[STEP 0] LOAD DATASET")
    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=max(n_eval, 10)
    )
    sample_img = images[sample_index]
    sample_label = labels[sample_index]
    image_path = save_temp_image(sample_img, dataset_name, sample_index)
    print(f"  Image shape: {sample_img.shape}, dtype: {sample_img.dtype}")
    print(f"  Ground truth: class {sample_label} ({class_names[sample_label] if sample_label < len(class_names) else '?'})")
    print(f"  Saved to: {image_path}")

    # == STEP 1: INPUT PROMPT ==============================================
    print("\n" + "=" * 80)
    print("[STEP 1] INPUT PROMPT (User Query)")
    print("=" * 80)
    # Get dataset description for richer query
    descriptions = _get_dataset_descriptions()
    desc_text = descriptions.get(dataset_name, "")
    query = _build_query(dataset_name, object_type,
                         n_channels=n_ch, description=desc_text)
    print(f'  query: "{query[:200]}..."' if len(query) > 200 else f'  query: "{query}"')
    print(f"  image_path: {image_path}")

    # == STEP 2: SKILLS (Expert Knowledge) =================================
    print("\n" + "=" * 80)
    print("[STEP 2] SKILLS (Expert Knowledge + Benchmark Rankings)")
    print("=" * 80)
    registry = SkillRegistry()
    color_mode = "per_channel" if is_rgb else "grayscale"

    skill_knowledge = registry.build_skill_context(
        object_type_hint=object_type,
        color_mode=color_mode,
        include_learned=False,
    )
    lines = skill_knowledge.split("\n")
    print(f"  Total knowledge: {len(skill_knowledge)} chars, {len(lines)} lines")
    print("  --- Skill knowledge preview (first 30 lines) ---")
    for line in lines[:30]:
        print(f"  {line}")
    if len(lines) > 30:
        print(f"  ... ({len(lines) - 30} more lines)")

    from topoagent.skills.rules_data import SUPPORTED_TOP_PERFORMERS
    top_descs = SUPPORTED_TOP_PERFORMERS.get(object_type, [])
    if top_descs:
        print(f"\n  Benchmark top-3 for {object_type}:")
        for i, td in enumerate(top_descs[:3]):
            params = registry.configure_after_selection(
                descriptor=td["descriptor"],
                object_type=object_type,
                color_mode=color_mode,
            )
            print(f"    #{i+1}: {td['descriptor']} ({td['accuracy']:.1%}) "
                  f"-> params={params}")

    # == STEP 3: LONG-TERM MEMORY ==========================================
    print("\n" + "=" * 80)
    print("[STEP 3] LONG-TERM MEMORY (learned_rules.json + reflections.json)")
    print("=" * 80)
    skill_mem = SkillMemory()

    rules = skill_mem.load_learned_rules()
    n_rules = len(rules.get("rules", []))
    updated_rankings = rules.get("updated_rankings", {})
    print(f"  Learned rules on disk: {n_rules} entries")
    print(f"  Updated rankings for: {list(updated_rankings.keys())}")

    relevant_rules = [
        r for r in rules.get("rules", [])
        if r.get("object_type") == object_type
    ]
    if relevant_rules:
        print(f"  Rules for {object_type}:")
        for r in relevant_rules[:3]:
            print(f"    - {r['agent_choice']} (acc={r['agent_accuracy']:.1%}, "
                  f"optimal={r.get('was_optimal', '?')})")

    reflections = skill_mem.load_reflections()
    relevant_refs = [r for r in reflections
                     if object_type in r.get("context", "")]
    print(f"\n  Total reflections on disk: {len(reflections)}")
    print(f"  Relevant to {object_type}: {len(relevant_refs)}")
    if relevant_refs:
        for r in relevant_refs[:2]:
            lesson = r.get("lesson_learned", "")[:120]
            print(f'    - "{lesson}..."')

    learned_ctx = skill_mem.get_learned_context(object_type)
    print(f"\n  Injected into LLM prompt ({len(learned_ctx)} chars):")
    for line in learned_ctx.split("\n")[:8]:
        print(f"    {line}")

    # == STEP 4: RUN TOPOAGENT =============================================
    print("\n" + "=" * 80)
    print(f"[STEP 4] RUN TOPOAGENT ({mode_label} mode, verbose=True)")
    print("=" * 80)
    print(f"  Creating agent with {model_name}, {mode_label.lower()}_mode=True ...")

    t0 = time.time()
    agent = create_topoagent_auto(
        model_name=model_name,
        skills_mode=not (agentic or agentic_v8 or agentic_v9),
        agentic_mode=agentic and not (agentic_v8 or agentic_v9),
        agentic_v8=agentic_v8 and not agentic_v9,
        agentic_v9=agentic_v9,
        time_limit_seconds=time_limit,
    )
    # Enable verbose output on the workflow
    agent.workflow.verbose = True
    result = agent.classify(
        image_path=image_path,
        query=query,
    )
    elapsed = time.time() - t0
    print(f"\n  Agent completed in {elapsed:.1f}s")

    # == STEP 5: REASONING =====================================================
    llm_interactions = result.get("llm_interactions", [])

    if agentic_v9:
        # v9 mode: Show INTERPRET + ANALYZE (hypothesis) + ACT (reconcile)
        v9_interpret = result.get("_v9_interpret_output") or {}
        if v9_interpret:
            print("\n" + "=" * 80)
            print("[STEP 5a] INTERPRET (LLM #1: blind data interpretation)")
            print("=" * 80)
            print(f"  Modality guess:       {v9_interpret.get('modality_guess', '?')}")
            print(f"  Modality evidence:    {v9_interpret.get('modality_evidence', '?')[:200]}")
            print(f"  Object type guess:    {v9_interpret.get('object_type_guess', '?')}")
            print(f"  Object type evidence: {v9_interpret.get('object_type_evidence', '?')[:200]}")
            print(f"  Color diagnostic:     {v9_interpret.get('color_diagnostic', '?')}")
            print(f"  Image profile:        {v9_interpret.get('image_profile', '?')[:200]}")
            print(f"  PH profile:           {v9_interpret.get('ph_profile', '?')[:200]}")
            ot_correct = v9_interpret.get('_object_type_correct')
            if ot_correct is not None:
                mark = "CORRECT" if ot_correct else f"WRONG (GT={object_type})"
                print(f"  Object type check:    {mark}")

        v9_hypothesis = result.get("_v9_hypothesis") or {}
        if v9_hypothesis:
            print("\n" + "=" * 80)
            print("[STEP 5b] ANALYZE (LLM #2: hypothesis formation)")
            print("=" * 80)
            print(f"  Descriptor hypothesis: {v9_hypothesis.get('descriptor_hypothesis', '?')}")
            print(f"  Descriptor reasoning:  {v9_hypothesis.get('descriptor_reasoning', '?')[:300]}")
            print(f"  Color mode:            {v9_hypothesis.get('color_mode', '?')}")
            print(f"  Color reasoning:       {v9_hypothesis.get('color_reasoning', '?')[:200]}")
            print(f"  Alternatives:          {v9_hypothesis.get('alternatives', [])}")
            print(f"  Confidence:            {v9_hypothesis.get('confidence', '?')}")
            param_intuition = v9_hypothesis.get('parameter_intuition', {})
            if param_intuition:
                print(f"  Parameter intuition:   {param_intuition}")

        v9_act = result.get("_v9_act_decision") or {}
        if v9_act:
            print("\n" + "=" * 80)
            print("[STEP 5c] ACT (LLM #3: reconcile hypothesis vs benchmark)")
            print("=" * 80)
            reconciliation = v9_act.get("reconciliation", {})
            print(f"  Hypothesis descriptor: {reconciliation.get('hypothesis_descriptor', '?')}")
            print(f"  Benchmark top:         {reconciliation.get('benchmark_top', '?')}")
            print(f"  Agreement:             {reconciliation.get('agreement', '?')}")
            print(f"  Stance:                {reconciliation.get('stance', '?')}")
            print(f"  Stance reasoning:      {reconciliation.get('stance_reasoning', '?')[:300]}")
            print(f"  Final descriptor:      {v9_act.get('final_descriptor', '?')}")
            print(f"  Final params:          {v9_act.get('final_params', {})}")
            print(f"  Color mode:            {v9_act.get('color_mode', '?')}")
            print(f"  Backup descriptor:     {v9_act.get('backup_descriptor', '?')}")
            param_recon = v9_act.get("param_reconciliation", {})
            if param_recon:
                print(f"  Parameter reconciliation:")
                for pname, pinfo in param_recon.items():
                    if isinstance(pinfo, dict):
                        print(f"    {pname}: value={pinfo.get('value', '?')}, "
                              f"source={pinfo.get('source', '?')}, "
                              f"reasoning={str(pinfo.get('reasoning', ''))[:100]}")

        # Memory stats
        v9_mem_stats = result.get("_v9_memory_stats") or {}
        if v9_mem_stats:
            print(f"\n  Memory stats:")
            print(f"    Entries available:     {v9_mem_stats.get('entries_available', 0)}")
            print(f"    Entries retrieved:     {v9_mem_stats.get('entries_retrieved', 0)}")
            print(f"    Top similarity:       {v9_mem_stats.get('top_similarity', 0):.3f}")
            print(f"    Memory influenced:    {v9_mem_stats.get('memory_influenced', False)}")

        if llm_interactions:
            print(f"\n  ({len(llm_interactions)} LLM interaction(s) recorded)")

    elif agentic_v8:
        # v8.1 mode: Show PERCEIVE_DECIDE + ANALYZE + PLAN phases
        perceive_decisions = result.get("_perceive_decisions") or {}
        if perceive_decisions:
            print("\n" + "=" * 80)
            print("[STEP 5a] PERCEIVE_DECIDE (LLM #1: filtration + denoising)")
            print("=" * 80)
            print(f"  Filtration type:     {perceive_decisions.get('filtration_type', '?')}")
            print(f"  Filtration reasoning: {perceive_decisions.get('filtration_reasoning', '?')}")
            print(f"  Apply denoising:     {perceive_decisions.get('apply_denoising', False)}")
            print(f"  Denoising method:    {perceive_decisions.get('denoising_method', 'N/A')}")
            print(f"  Denoising reasoning: {perceive_decisions.get('denoising_reasoning', '?')}")
            print(f"  Max dimension:       {perceive_decisions.get('max_dimension', 1)}")

        print("\n" + "=" * 80)
        print("[STEP 5b] ANALYZE (LLM #2: synthesize perception)")
        print("=" * 80)
        v8_analysis = result.get("_v8_analysis_context") or {}
        if v8_analysis:
            print(f"  Object type:          {v8_analysis.get('object_type', '?')}")
            print(f"  Color mode:           {v8_analysis.get('color_mode', '?')}")
            print(f"  Image characteristics: {v8_analysis.get('image_characteristics', '?')}")
            print(f"  PH interpretation:    {v8_analysis.get('ph_interpretation', '?')}")
            print(f"  Descriptor intuition: {v8_analysis.get('descriptor_intuition', '?')}")
            print(f"  Tools used summary:   {v8_analysis.get('tools_used_summary', '?')}")
            ot_correct = v8_analysis.get('_object_type_correct')
            if ot_correct is not None:
                mark = "CORRECT" if ot_correct else f"WRONG (GT={object_type})"
                print(f"  Object type check:    {mark}")
        else:
            print("  (No v8 analysis context recorded)")

        print("\n" + "=" * 80)
        print("[STEP 5c] PLAN (LLM #3: select descriptor with LTM)")
        print("=" * 80)
        v8_plan = result.get("_v8_plan_context") or {}
        if v8_plan:
            print(f"  Primary descriptor:   {v8_plan.get('primary_descriptor', '?')}")
            print(f"  Stance:               {v8_plan.get('stance', '?')}")
            print(f"  Backup descriptor:    {v8_plan.get('backup_descriptor', 'none')}")
            print(f"  Request fusion:       {v8_plan.get('request_fusion', False)}")
            print(f"  Parameters:           {v8_plan.get('primary_params', {})}")
            reasoning = v8_plan.get('reasoning', '')
            print(f"  Reasoning:")
            for line in reasoning.split('. '):
                if line.strip():
                    print(f"    - {line.strip()}")
        else:
            print("  (No v8 plan context recorded)")

        # Also show raw LLM interactions if any
        if llm_interactions:
            print(f"\n  ({len(llm_interactions)} LLM interaction(s) recorded)")
    else:
        # v7 / skills mode
        print("\n" + "=" * 80)
        print("[STEP 5] REASONING (LLM #1: plan_descriptor)")
        print("=" * 80)

        plan_json = None
        if len(llm_interactions) >= 1:
            i1 = llm_interactions[0]
            prompt1 = i1.get("prompt", "") if isinstance(i1, dict) else getattr(i1, "prompt", "")
            response1 = i1.get("response", "") if isinstance(i1, dict) else getattr(i1, "response", "")

            print("  [LLM #1 Prompt -- first 50 lines]")
            for line in prompt1.split("\n")[:50]:
                print(f"    {line}")

            print(f"\n  [LLM #1 Response -- {len(response1)} chars]")
            try:
                plan_json = json.loads(response1) if response1.strip().startswith("{") else None
            except (json.JSONDecodeError, AttributeError):
                plan_json = None

            if plan_json:
                print("    Parsed plan JSON:")
                for k, v in plan_json.items():
                    print(f"      {k}: {str(v)[:120]}")
            else:
                for line in response1.split("\n")[:20]:
                    print(f"    {line}")
        else:
            print("  (No LLM interactions recorded)")

    # == STEP 6: ACTION =====================================================
    print("\n" + "=" * 80)
    print("[STEP 6] ACTION (Tool Execution Pipeline)")
    print("=" * 80)
    tools_used = result.get("tools_used", [])
    print(f"  Tools executed in order: {tools_used}")

    report = result.get("report") or result.get("raw_answer") or {}
    if isinstance(report, dict):
        params = report.get("parameters", {})
        print(f"  Descriptor chosen: {report.get('descriptor', 'N/A')}")
        print(f"  Parameters applied: {params}")
        print(f"  Feature dimension: "
              f"{report.get('actual_dimension', report.get('feature_dimension', 'N/A'))}")
        print(f"  Color mode: {report.get('color_mode', 'N/A')}")

    reasoning_trace = result.get("reasoning_trace", [])
    print(f"\n  Reasoning trace ({len(reasoning_trace)} steps):")
    for i, step in enumerate(reasoning_trace):
        print(f"    [{i}] {step}")

    # == STEP 7: SHORT-TERM MEMORY ==========================================
    print("\n" + "=" * 80)
    print("[STEP 7] SHORT-TERM MEMORY (Session-scoped tool outputs)")
    print("=" * 80)
    print("  Short-term memory stores (tool_name, output) tuples.")
    print("  Enables auto-injection: image_array -> compute_ph, "
          "persistence_data -> descriptor.")
    print(f"  Tools in memory: {tools_used}")

    descriptor_name = (report.get("descriptor", "unknown")
                       if isinstance(report, dict)
                       else result.get("classification", "unknown"))

    for i, tool_name in enumerate(tools_used):
        print(f"    [{i}] {tool_name}:")
        if tool_name == "image_loader":
            print("        -> Stores image_array "
                  "(used by compute_ph + image-based descriptors)")
        elif tool_name == "compute_ph":
            print("        -> Stores persistence_data with H0/H1 pairs "
                  "(used by PH-based descriptors)")
            print("        -> Extracts PH statistics for LLM #1 prompt")
        else:
            dim_str = report.get("actual_dimension", "?") if isinstance(report, dict) else "?"
            print(f"        -> Stores feature vector ({dim_str}D)")
            print("        -> This is the final topology descriptor output")

    # == STEP 8: REFLECTION ===================================================
    print("\n" + "=" * 80)

    if agentic_v9:
        # v9 mode: show REFLECT with raw stats evaluation
        v9_reflect = result.get("_v9_reflect_output") or {}
        print("[STEP 8] REFLECTION (v9 LLM-driven evaluation + structured LTM)")
        print("=" * 80)

        reflect_history = result.get("_reflect_history", [])
        for rh in reflect_history:
            rnd = rh.get("round", "?")
            print(f"\n  --- Reflect Round {rnd} ---")
            fq = rh.get("feature_quality", {})
            print(f"    Descriptor:    {rh.get('descriptor', 'unknown')}")
            print(f"    Feature:       dim={fq.get('dimension', '?')}, "
                  f"sparsity={fq.get('sparsity', 0):.1f}%, "
                  f"variance={fq.get('variance', 0):.6f}, "
                  f"NaN={fq.get('nan_count', 0)}")
            print(f"    Quality OK:    {rh.get('quality_ok', '?')}")
            print(f"    Decision:      {rh.get('decision', '?')}")
            print(f"    Reasoning:     {rh.get('reasoning', '')[:250]}")

        if v9_reflect:
            qa = v9_reflect.get("quality_assessment", {})
            if qa:
                print(f"\n  Quality Assessment:")
                print(f"    Overall quality:  {qa.get('overall_quality', '?')}")
                print(f"    Reasoning:        {qa.get('reasoning', '')[:250]}")
            exp = v9_reflect.get("experience_entry", {})
            if exp:
                print(f"\n  Experience Entry (written to LTM):")
                print(f"    Object type:      {exp.get('object_type', '?')}")
                print(f"    Descriptor:       {exp.get('descriptor', '?')}")
                print(f"    Quality:          {exp.get('quality_verdict', '?')}")
                print(f"    Would choose again: {exp.get('would_choose_again', '?')}")
                print(f"    Lesson:           {exp.get('lesson', '?')[:200]}")

        ltm = agent._long_term_memory
        if ltm:
            entries = ltm.get_all()
            print(f"\n  Long-term memory now has {len(entries)} entries")

        retry_used = (report.get("retry_used", False)
                      if isinstance(report, dict) else False)
        print(f"\n  Retry triggered: {retry_used}")

    elif agentic_v8:
        # v8 mode: show LLM-driven reflection with experience entry
        v8_exp = result.get("_v8_reflect_experience") or {}
        print("[STEP 8] REFLECTION (v8 LLM-driven evaluation + LTM write)")
        print("=" * 80)

        reflect_history = result.get("_reflect_history", [])
        for rh in reflect_history:
            rnd = rh.get("round", "?")
            print(f"\n  --- Reflect Round {rnd} ---")
            fq = rh.get("feature_quality", {})
            print(f"    Descriptor:    {rh.get('descriptor', 'unknown')}")
            print(f"    Feature:       dim={fq.get('dimension', '?')}, "
                  f"sparsity={fq.get('sparsity', 0):.1f}%, "
                  f"variance={fq.get('variance', 0):.6f}, "
                  f"NaN={fq.get('nan_count', 0)}")
            print(f"    Quality OK:    {rh.get('quality_ok', '?')}")
            print(f"    Decision:      {rh.get('decision', '?')}")
            print(f"    Reasoning:     {rh.get('reasoning', '')[:250]}")

        if v8_exp:
            print(f"\n  Experience Entry (written to LTM):")
            print(f"    Object type:      {v8_exp.get('object_type', '?')}")
            print(f"    Descriptor:       {v8_exp.get('descriptor', '?')}")
            print(f"    PH profile:       {v8_exp.get('ph_profile_summary', '?')}")
            print(f"    Image profile:    {v8_exp.get('image_profile_summary', '?')}")
            print(f"    Quality:          {v8_exp.get('quality', '?')}")
            print(f"    Lesson:           {v8_exp.get('lesson', '?')}")

        ltm = agent._long_term_memory
        if ltm:
            entries = ltm.get_all()
            print(f"\n  Long-term memory now has {len(entries)} entries")

        retry_used = (report.get("retry_used", False)
                      if isinstance(report, dict) else False)
        print(f"\n  Retry triggered: {retry_used}")

    elif result.get("_reflect_history"):
        reflect_history = result.get("_reflect_history", [])
        # Agentic mode: show each reflect round from _reflect_history
        n_rounds = len(reflect_history)
        print(f"[STEP 8] REFLECTION ({n_rounds} round{'s' if n_rounds > 1 else ''})")
        print("=" * 80)

        for rh in reflect_history:
            rnd = rh.get("round", "?")
            print(f"\n  --- Reflect Round {rnd} ---")
            print(f"    Descriptor:    {rh.get('descriptor', 'unknown')}")
            print(f"    Parameters:    {rh.get('descriptor_params', {})}")
            ph = rh.get("ph_stats", {})
            print(f"    PH stats:      H0={ph.get('H0_count', '?')}, "
                  f"H1={ph.get('H1_count', '?')}, "
                  f"filtration={ph.get('filtration', '?')}")
            fq = rh.get("feature_quality", {})
            print(f"    Feature:       dim={fq.get('dimension', '?')}, "
                  f"sparsity={fq.get('sparsity', 0):.1f}%, "
                  f"variance={fq.get('variance', 0):.6f}, "
                  f"NaN={fq.get('nan_count', 0)}")
            print(f"    Quality OK:    {rh.get('quality_ok', '?')}")
            print(f"    Decision:      {rh.get('decision', '?')}")
            print(f"    Reasoning:     {rh.get('reasoning', '')[:200]}")
            if rh.get("retry_suggestion"):
                print(f"    Retry hint:    {rh['retry_suggestion'][:150]}")

        retry_used = (report.get("retry_used", False)
                      if isinstance(report, dict) else False)
        print(f"\n  Retry triggered: {retry_used}")
        if retry_used:
            print("  -> Reflect flagged quality issues; agent retried with "
                  "different descriptor/PH params")
        else:
            print("  -> Quality OK on first pass, no retry needed")
    else:
        # Skills mode fallback: show LLM #2 (verify_and_reflect)
        print("[STEP 8] REFLECTION (LLM #2: verify_and_reflect)")
        print("=" * 80)

        verify_json = None
        if len(llm_interactions) >= 2:
            i2 = llm_interactions[1]
            prompt2 = (i2.get("prompt", "") if isinstance(i2, dict)
                       else getattr(i2, "prompt", ""))
            response2 = (i2.get("response", "") if isinstance(i2, dict)
                         else getattr(i2, "response", ""))

            print("  [LLM #2 Prompt -- first 40 lines]")
            for line in prompt2.split("\n")[:40]:
                print(f"    {line}")

            print(f"\n  [LLM #2 Response -- {len(response2)} chars]")
            try:
                verify_json = (json.loads(response2)
                               if response2.strip().startswith("{") else None)
            except (json.JSONDecodeError, AttributeError):
                verify_json = None

            if verify_json:
                v = verify_json.get("verification", {})
                print("    Verification:")
                print(f"      ph_confirms_object_type: "
                      f"{v.get('ph_confirms_object_type')}")
                print(f"      dimension_correct: {v.get('dimension_correct')}")
                print(f"      quality_ok: {v.get('quality_ok')}")
                print(f"      issues: {v.get('issues', [])}")
                print(f"      suggestion: {v.get('suggestion')}")

                r = verify_json.get("reflection", {})
                print("    Reflection:")
                print(f"      error_analysis: "
                      f"{r.get('error_analysis', '')[:200]}")
                print(f"      experience: {r.get('experience', '')[:200]}")

                rpt = verify_json.get("report", {})
                print("    Report:")
                for k, val in rpt.items():
                    print(f"      {k}: {str(val)[:120]}")
            else:
                for line in response2.split("\n")[:20]:
                    print(f"    {line}")

            retry_used = (report.get("retry_used", False)
                          if isinstance(report, dict) else False)
            print(f"\n  Retry triggered: {retry_used}")
            if retry_used:
                print("  -> Verification flagged quality issues; "
                      "agent swapped to alternative descriptor")
            else:
                print("  -> Quality OK, no retry needed")
        else:
            print("  (Only 1 LLM interaction -- reflection not recorded)")

    # == STEP 9: OUTPUT (Feature Vector) ====================================
    print("\n" + "=" * 80)
    print("[STEP 9] OUTPUT (Topology Feature Vector)")
    print("=" * 80)

    # First try to get the feature vector from the agentic pipeline result
    # (which already has per-channel concatenation if applicable).
    feature_vector = None
    stm = result.get("short_term_memory", [])
    for tool_name_stm, output_stm in reversed(stm):
        if tool_name_stm in SUPPORTED_DESCRIPTORS_ALL and isinstance(output_stm, dict):
            fv = output_stm.get("combined_vector") or output_stm.get("feature_vector")
            if fv is not None:
                feature_vector = fv
                break
    # Fallback: re-extract if pipeline didn't produce a vector
    if feature_vector is None:
        feature_vector = _extract_feature_vector(
            agent, image_path, descriptor_name, registry, object_type, color_mode
        )

    if feature_vector is not None:
        arr = np.asarray(feature_vector, dtype=np.float64)
        print(f"  Descriptor: {descriptor_name}")
        print(f"  Feature vector shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        print(f"  Statistics:")
        print(f"    mean={arr.mean():.6f}, std={arr.std():.6f}")
        print(f"    min={arr.min():.6f}, max={arr.max():.6f}")
        sparsity = 100 * np.mean(arr == 0)
        print(f"    sparsity={sparsity:.1f}%")
        print(f"    non-zero={np.count_nonzero(arr)}/{len(arr)}")
        print(f"  First 20 values: {arr[:20].tolist()}")

        outdir = PROJECT_ROOT / "results" / "case_studies"
        outdir.mkdir(parents=True, exist_ok=True)
        npy_path = outdir / f"{dataset_name}_sample{sample_index}_features.npy"
        np.save(str(npy_path), arr)
        print(f"  Saved to: {npy_path}")
    else:
        arr = None
        sparsity = 0.0
        print(f"  WARNING: Could not extract feature vector for {descriptor_name}")

    # == STEP 10: CLASSIFIER EVALUATION =====================================
    accuracy = 0.0
    if not skip_eval:
        print("\n" + "=" * 80)
        print(f"[STEP 10] CLASSIFIER EVALUATION ({n_eval} samples, 3-fold CV)")
        print("=" * 80)
        accuracy = _run_classifier_eval(
            dataset_name, descriptor_name, registry, object_type,
            color_mode, n_eval
        )
    else:
        print("\n[STEP 10] CLASSIFIER EVALUATION -- skipped (--no-eval)")

    # == SUMMARY ============================================================
    print("\n" + "=" * 80)
    print("  CASE STUDY SUMMARY")
    print("=" * 80)

    # Extract decisions from result (agentic v7/v8 fields)
    decisions_data = {}
    if isinstance(report, dict):
        decisions_data = {
            "object_type": report.get("observe_object_type", ""),
            "object_type_reasoning": report.get("observe_reasoning", ""),
            "object_type_correct": report.get("object_type_correct"),
            "color_mode": report.get("observe_color_mode", ""),
            "color_mode_reasoning": "",
            "filtration_type": report.get("observe_filtration", ""),
            "filtration_reasoning": "",
            "ph_interpretation": report.get("ph_interpretation", ""),
            "descriptor": descriptor_name,
            "benchmark_stance": report.get("benchmark_stance", ""),
        }
        # Try to get full reasoning from _observe_decisions in result
        obs_dec = result.get("_observe_decisions") or {}
        if obs_dec:
            decisions_data["object_type_reasoning"] = obs_dec.get("object_type_reasoning", "")
            decisions_data["color_mode_reasoning"] = obs_dec.get("color_mode_reasoning", "")
            decisions_data["filtration_reasoning"] = obs_dec.get("filtration_reasoning", "")
            decisions_data["object_type_correct"] = obs_dec.get("_object_type_correct")
        # v8.1: Perceive decisions
        perceive_dec = result.get("_perceive_decisions") or {}
        if perceive_dec:
            decisions_data["perceive_filtration"] = perceive_dec.get("filtration_type", "")
            decisions_data["perceive_filtration_reasoning"] = perceive_dec.get("filtration_reasoning", "")
            decisions_data["perceive_denoising"] = perceive_dec.get("apply_denoising", False)
            decisions_data["perceive_denoising_reasoning"] = perceive_dec.get("denoising_reasoning", "")
        # PH signals
        ph_signals = result.get("_ph_signals") or []
        decisions_data["ph_signals"] = [
            {"name": s["name"], "recommends": s["recommends"],
             "metrics": s.get("metrics", {})}
            for s in ph_signals
        ]
        # v9: hypothesis-first agentic fields
        if agentic_v9:
            decisions_data["v9_interpret"] = result.get("_v9_interpret_output")
            decisions_data["v9_hypothesis"] = result.get("_v9_hypothesis")
            decisions_data["v9_act_decision"] = result.get("_v9_act_decision")
            decisions_data["v9_reflect"] = result.get("_v9_reflect_output")
            decisions_data["v9_memory_stats"] = result.get("_v9_memory_stats")

    summary = {
        "dataset": dataset_name,
        "sample_index": sample_index,
        "object_type": object_type,
        "color_mode": color_mode,
        "decisions": decisions_data,
        "input": {
            "query": query,
            "image_path": image_path,
            "image_shape": list(sample_img.shape),
            "ground_truth_label": int(sample_label),
            "ground_truth_class": (class_names[sample_label]
                                   if sample_label < len(class_names) else str(sample_label)),
        },
        "skills": {
            "knowledge_chars": len(skill_knowledge),
            "knowledge_text": skill_knowledge,
            "top_3_descriptors": [
                {"descriptor": td["descriptor"], "accuracy": td["accuracy"]}
                for td in top_descs[:3]
            ] if top_descs else [],
        },
        "long_term_memory": {
            "total_rules": n_rules,
            "relevant_rules": len(relevant_rules),
            "total_reflections": len(reflections),
            "relevant_reflections": len(relevant_refs),
            "injected_chars": len(learned_ctx),
            "learned_context_text": learned_ctx,
            "relevant_rules_detail": [
                {
                    "object_type": r.get("object_type"),
                    "agent_choice": r.get("agent_choice"),
                    "agent_accuracy": r.get("agent_accuracy"),
                    "was_optimal": r.get("was_optimal"),
                }
                for r in relevant_rules[:5]
            ],
            "relevant_reflections_detail": [
                {
                    "context": r.get("context", "")[:200],
                    "lesson_learned": r.get("lesson_learned", "")[:300],
                }
                for r in relevant_refs[:3]
            ],
        },
        "reasoning": {
            "object_type": (report.get("object_type", object_type)
                           if isinstance(report, dict) else object_type),
            "reasoning_chain": (report.get("reasoning_chain", "")
                               if isinstance(report, dict) else ""),
            "descriptor_choice": descriptor_name,
            "descriptor_rationale": (report.get("descriptor_rationale", "")
                                    if isinstance(report, dict) else ""),
            "alternative": (report.get("alternatives_considered", "")
                           if isinstance(report, dict) else ""),
        },
        "action": {
            "tools_executed": tools_used,
            "n_llm_calls": (report.get("n_llm_calls", len(llm_interactions))
                           if isinstance(report, dict) else len(llm_interactions)),
            "n_tools": len(tools_used),
            "parameters": (report.get("parameters", {})
                          if isinstance(report, dict) else {}),
        },
        "short_term_memory": {
            "entries": len(tools_used),
            "auto_injections": [
                "image_array -> compute_ph",
                (f"persistence_data -> {descriptor_name}"
                 if descriptor_name not in IMAGE_BASED_DESCRIPTORS
                 else f"image_array -> {descriptor_name}"),
            ],
        },
        "reflection": {
            "quality_ok": (report.get("quality_ok", True)
                          if isinstance(report, dict) else True),
            "error_analysis": (report.get("error_analysis", "")
                              if isinstance(report, dict) else ""),
            "experience": (report.get("experience", "")
                          if isinstance(report, dict) else ""),
            "retry_used": (report.get("retry_used", False)
                          if isinstance(report, dict) else False),
            "reflect_rounds": reflect_history,
        },
        "output": {
            "descriptor": descriptor_name,
            "feature_dimension": int(arr.shape[0]) if arr is not None else 0,
            "sparsity_pct": round(float(sparsity), 1),
            "feature_mean": round(float(arr.mean()), 6) if arr is not None else 0,
            "feature_std": round(float(arr.std()), 6) if arr is not None else 0,
        },
        "classifier": {
            "method": "XGBoost (3-fold CV)",
            "n_samples": n_eval,
            "accuracy": accuracy,
        },
        "llm_interactions": _serialize_llm_interactions(llm_interactions),
        "metadata": {
            "total_time_seconds": round(elapsed, 2),
            "n_llm_calls": len(llm_interactions),
            "model": model_name,
        },
    }

    print(f"  Dataset:           {dataset_name}")
    print(f"  Object type (GT):  {object_type}")
    if decisions_data.get("object_type"):
        ot_correct = decisions_data.get("object_type_correct")
        ot_mark = " OK" if ot_correct else f" WRONG (GT={object_type})" if ot_correct is False else ""
        print(f"  Object type (LLM): {decisions_data['object_type']}{ot_mark}")
    print(f"  Color mode:        {color_mode}")
    if decisions_data.get("benchmark_stance"):
        print(f"  Benchmark stance:  {decisions_data['benchmark_stance']}")
    print(f"  Descriptor:        {descriptor_name}")
    print(f"  Reasoning chain:   {summary['reasoning']['reasoning_chain'][:120]}")
    print(f"  Tools:             {' -> '.join(tools_used)}")
    print(f"  Feature dim:       {summary['output']['feature_dimension']}D")
    print(f"  Sparsity:          {summary['output']['sparsity_pct']:.1f}%")
    print(f"  LLM calls:         {len(llm_interactions)}")
    print(f"  Retry:             {summary['reflection']['retry_used']}")
    print(f"  Quality OK:        {summary['reflection']['quality_ok']}")
    print(f"  Accuracy (CV):     {accuracy:.1f}%")
    print(f"  Total time:        {elapsed:.1f}s")
    print(f"  Long-term memory:  {n_rules} rules, {len(reflections)} reflections")

    outdir = PROJECT_ROOT / "results" / "case_studies"
    json_path = outdir / f"{dataset_name}_explicit_case_study.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Full case study saved to: {json_path}")

    return summary


# -- Multi-image within-session learning -----------------------------------

def _run_multi_image_case_study(dataset_name: str, n_samples: int = 3,
                                 n_eval: int = 200, skip_eval: bool = False,
                                 time_limit: float = 60.0,
                                 model_name: str = "gpt-4o",
                                 agentic_v9: bool = False):
    """Run n_samples images sequentially with the SAME agent.

    Demonstrates within-session learning:
      Image 1: LTM empty → PLAN has no past experience → fresh decision
      Image 2: LTM has 1 entry → PLAN reads it → may change decision
      Image 3: LTM has 2 entries → richer context → demonstrates learning

    Returns:
        Aggregated summary with per-image results and LTM growth trace.
    """
    from topoagent.agent import create_topoagent_auto
    from topoagent.skills import SkillRegistry
    from topoagent.memory.skill_memory import SkillMemory

    info = DATASET_INFO[dataset_name]
    object_type = info["object_type"]
    is_rgb = info["color"]
    color_mode = "per_channel" if is_rgb else "grayscale"

    mode_tag = "AGENTIC_V9" if agentic_v9 else "AGENTIC_V8"
    print("=" * 80)
    print(f"  TopoAgent Multi-Image Case Study [{mode_tag}]")
    print(f"  Dataset: {dataset_name} ({n_samples} images, within-session learning)")
    print("=" * 80)

    # Load dataset — only load what we need (skip_eval → just n_samples images)
    n_load = n_samples + 5 if skip_eval else max(n_eval, n_samples + 5)
    print(f"\n[LOAD] Loading dataset ({n_load} images)...")
    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=n_load
    )

    # Create ONE agent for all images — shared LTM across images
    print(f"\n[INIT] Creating agent with shared LTM (model={model_name})...")
    agent = create_topoagent_auto(
        model_name=model_name,
        agentic_v8=not agentic_v9,
        agentic_v9=agentic_v9,
        time_limit_seconds=time_limit,
    )
    agent.workflow.verbose = True

    # Clear LTM so this dataset starts fresh (no leakage from previous datasets)
    if agent._long_term_memory is not None:
        agent._long_term_memory.clear()
        print("  LTM cleared for fresh within-session learning")

    # Get dataset description for queries
    descriptions = _get_dataset_descriptions()
    desc_text = descriptions.get(dataset_name, "")

    registry = SkillRegistry()
    per_image_results = []

    for img_idx in range(n_samples):
        print("\n" + "=" * 80)
        print(f"  IMAGE {img_idx + 1}/{n_samples}")
        print("=" * 80)

        # Check LTM before this image
        ltm = agent._long_term_memory
        ltm_entries_before = len(ltm.get_all()) if ltm else 0
        print(f"  LTM entries available: {ltm_entries_before}")

        sample_img = images[img_idx]
        sample_label = labels[img_idx]
        image_path = save_temp_image(sample_img, dataset_name, img_idx)
        print(f"  Image shape: {sample_img.shape}")
        print(f"  Ground truth: class {sample_label} "
              f"({class_names[sample_label] if sample_label < len(class_names) else '?'})")

        query = _build_query(dataset_name, object_type,
                             n_channels=n_ch, description=desc_text)

        t0 = time.time()
        result = agent.classify(image_path=image_path, query=query)
        elapsed = time.time() - t0

        # Extract key info
        report = result.get("report") or result.get("raw_answer") or {}
        llm_interactions = result.get("llm_interactions", [])
        reflect_history = result.get("_reflect_history", [])
        v8_analysis = result.get("_v8_analysis_context") or {}
        v8_plan = result.get("_v8_plan_context") or {}
        perceive_decisions = result.get("_perceive_decisions") or {}
        # v9-specific
        v9_interpret = result.get("_v9_interpret_output") or {}
        v9_hypothesis = result.get("_v9_hypothesis") or {}
        v9_act = result.get("_v9_act_decision") or {}
        v9_mem_stats = result.get("_v9_memory_stats") or {}

        # Descriptor used
        descriptor_name = (report.get("descriptor", "unknown")
                           if isinstance(report, dict) else "unknown")

        # LTM after this image
        ltm_entries_after = len(ltm.get_all()) if ltm else 0

        img_result = {
            "image_index": img_idx,
            "ground_truth_label": int(sample_label),
            "ground_truth_class": (class_names[sample_label]
                                   if sample_label < len(class_names) else str(sample_label)),
            "ltm_entries_available": ltm_entries_before,
            "ltm_entries_after": ltm_entries_after,
            "perceive_decisions": {
                "filtration_type": perceive_decisions.get("filtration_type", "sublevel"),
                "apply_denoising": perceive_decisions.get("apply_denoising", False),
                "filtration_reasoning": perceive_decisions.get("filtration_reasoning", ""),
                "denoising_reasoning": perceive_decisions.get("denoising_reasoning", ""),
            },
            "analysis": {
                "object_type": (v9_interpret.get("object_type_guess", "?")
                                if agentic_v9 else v8_analysis.get("object_type", "?")),
                "color_mode": (v9_hypothesis.get("color_mode", "?")
                               if agentic_v9 else v8_analysis.get("color_mode", "?")),
                "descriptor_intuition": (v9_hypothesis.get("descriptor_reasoning", "")
                                         if agentic_v9 else v8_analysis.get("descriptor_intuition", "")),
            },
            "plan": {
                "descriptor": descriptor_name,
                "stance": (v9_act.get("reconciliation", {}).get("stance", "?")
                           if agentic_v9 else v8_plan.get("stance", "?")),
                "backup": (v9_act.get("backup_descriptor", "?")
                           if agentic_v9 else v8_plan.get("backup_descriptor", "?")),
                "reasoning": ((v9_act.get("reconciliation", {}).get("stance_reasoning", "")[:200])
                              if agentic_v9 else v8_plan.get("reasoning", "")[:200]),
            },
            "reflect": {
                "n_rounds": len(reflect_history),
                "quality_ok": reflect_history[-1].get("quality_ok", True) if reflect_history else True,
                "decision": reflect_history[-1].get("decision", "COMPLETE") if reflect_history else "COMPLETE",
            },
            "n_llm_calls": len(llm_interactions),
            "tools_used": result.get("tools_used", []),
            "time_seconds": round(elapsed, 2),
        }
        per_image_results.append(img_result)

        print(f"\n  --- Image {img_idx + 1} Summary ---")
        print(f"  LTM: {ltm_entries_before} → {ltm_entries_after} entries")
        if agentic_v9:
            stance = v9_act.get("reconciliation", {}).get("stance", "?")
            hypothesis = v9_hypothesis.get("descriptor_hypothesis", "?")
            print(f"  Hypothesis: {hypothesis}")
            print(f"  Descriptor: {descriptor_name} (stance={stance})")
            if v9_mem_stats:
                print(f"  Memory influence: {v9_mem_stats.get('memory_influenced', False)} "
                      f"(sim={v9_mem_stats.get('top_similarity', 0):.3f})")
        else:
            print(f"  Filtration: {perceive_decisions.get('filtration_type', 'sublevel')}")
            print(f"  Denoising: {perceive_decisions.get('apply_denoising', False)}")
            print(f"  Descriptor: {descriptor_name} (stance={v8_plan.get('stance', '?')})")
        print(f"  LLM calls: {len(llm_interactions)}")
        print(f"  Reflect decision: {img_result['reflect']['decision']}")
        print(f"  Time: {elapsed:.1f}s")

    # Aggregate summary
    print("\n" + "=" * 80)
    print(f"  MULTI-IMAGE SUMMARY ({n_samples} images)")
    print("=" * 80)

    descriptors_used = [r["plan"]["descriptor"] for r in per_image_results]
    llm_call_counts = [r["n_llm_calls"] for r in per_image_results]
    ltm_growth = [r["ltm_entries_after"] - r["ltm_entries_available"] for r in per_image_results]
    retries = sum(1 for r in per_image_results if r["reflect"]["decision"] != "COMPLETE"
                  or r["reflect"]["n_rounds"] > 1)

    for r in per_image_results:
        print(f"  Image {r['image_index'] + 1}: "
              f"LTM={r['ltm_entries_available']}→{r['ltm_entries_after']}, "
              f"descriptor={r['plan']['descriptor']}, "
              f"stance={r['plan']['stance']}, "
              f"LLM_calls={r['n_llm_calls']}, "
              f"time={r['time_seconds']:.1f}s")

    print(f"\n  Descriptors used: {descriptors_used}")
    print(f"  LLM calls per image: {llm_call_counts} (range: {min(llm_call_counts)}-{max(llm_call_counts)})")
    print(f"  LTM growth per image: {ltm_growth}")
    print(f"  Retries triggered: {retries}/{n_samples}")

    # Within-session learning evidence
    if len(per_image_results) >= 2:
        ltm0 = per_image_results[0]["ltm_entries_available"]
        ltm1 = per_image_results[1]["ltm_entries_available"]
        print(f"\n  Within-session learning:")
        print(f"    Image 1 LTM entries available: {ltm0}")
        print(f"    Image 2 LTM entries available: {ltm1}")
        if ltm1 > ltm0:
            print(f"    ✓ Image 2 had access to {ltm1 - ltm0} new LTM entries from Image 1")
        else:
            print(f"    ⚠ No new LTM entries between images (REFLECT may not have written)")

    summary = {
        "dataset": dataset_name,
        "object_type": object_type,
        "color_mode": color_mode,
        "n_samples": n_samples,
        "per_image_results": per_image_results,
        "aggregate": {
            "descriptors_used": descriptors_used,
            "unique_descriptors": len(set(descriptors_used)),
            "llm_call_counts": llm_call_counts,
            "ltm_growth": ltm_growth,
            "retries": retries,
            "total_time": sum(r["time_seconds"] for r in per_image_results),
        },
    }

    outdir = PROJECT_ROOT / "results" / "case_studies"
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{dataset_name}_multi_image_case_study.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved to: {json_path}")

    return summary


# -- Helpers ----------------------------------------------------------------

def _extract_feature_vector(agent, image_path, descriptor_name,
                            registry, object_type, color_mode):
    """Re-extract feature vector by running the tool pipeline directly."""
    try:
        tools = agent.workflow.tools

        img_result = tools["image_loader"].invoke({
            "image_path": image_path,
            "normalize": True,
            "grayscale": True,
        })
        if not img_result.get("success"):
            return None

        image_array = img_result["image_array"]

        if descriptor_name in IMAGE_BASED_DESCRIPTORS:
            params = registry.configure_after_selection(
                descriptor_name, object_type, color_mode)
            tool_args = {"image_array": image_array}
            for k, v in params.items():
                if k not in ("total_dim", "classifier", "color_mode", "dim"):
                    tool_args[k] = v
            result = tools[descriptor_name].invoke(tool_args)
        else:
            ph_result = tools["compute_ph"].invoke({
                "image_array": image_array,
                "filtration_type": "sublevel",
                "max_dimension": 1,
            })
            if not ph_result.get("success"):
                return None

            params = registry.configure_after_selection(
                descriptor_name, object_type, color_mode)
            tool_args = {"persistence_data": ph_result["persistence"]}
            for k, v in params.items():
                if k not in ("total_dim", "classifier", "color_mode", "dim"):
                    tool_args[k] = v
            result = tools[descriptor_name].invoke(tool_args)

        if result.get("success"):
            return result.get("combined_vector") or result.get("feature_vector")
        return None
    except Exception as e:
        print(f"  Feature extraction error: {e}")
        return None


def _run_classifier_eval(dataset_name, descriptor_name, registry,
                         object_type, color_mode, n_eval):
    """Run 3-fold stratified CV with XGBoost on extracted features."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import balanced_accuracy_score
    from topoagent.agent import create_topoagent_auto

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not available, skipping classifier eval")
        return 0.0

    print(f"  Loading {n_eval} samples from {dataset_name}...")
    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=n_eval)

    # Create a fresh agent just for tool access (no LLM calls)
    agent = create_topoagent_auto(
        model_name="gpt-4o", skills_mode=True)
    tools = agent.workflow.tools

    params = registry.configure_after_selection(
        descriptor_name, object_type, color_mode)

    is_per_channel = color_mode == "per_channel"
    is_image_based = descriptor_name in IMAGE_BASED_DESCRIPTORS
    clean_params = {k: v for k, v in params.items()
                    if k not in ("total_dim", "classifier", "color_mode", "dim")}

    print(f"  Extracting {descriptor_name} features for {len(images)} images "
          f"(color_mode={color_mode})...")
    t0 = time.time()

    features_list = []
    valid_labels = []

    for i, (img, lbl) in enumerate(zip(images, labels)):
        if i > 0 and i % 100 == 0:
            print(f"    ... {i}/{len(images)} ({time.time()-t0:.0f}s)")

        try:
            if is_per_channel and img.ndim == 3 and img.shape[2] == 3:
                # --- Per-channel extraction: R, G, B separately ---
                rgb = img.astype(np.float32)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                channel_vectors = []

                for ch_idx in range(3):
                    ch_array = rgb[:, :, ch_idx].tolist()
                    ch_args = dict(clean_params)

                    if is_image_based:
                        ch_args["image_array"] = ch_array
                    else:
                        ch_ph = tools["compute_ph"].invoke({
                            "image_array": ch_array,
                            "filtration_type": "sublevel",
                            "max_dimension": 1,
                        })
                        if not ch_ph.get("success"):
                            break
                        ch_args["persistence_data"] = ch_ph["persistence"]

                    ch_result = tools[descriptor_name].invoke(ch_args)
                    fv = ch_result.get("combined_vector") or ch_result.get("feature_vector")
                    if fv is not None:
                        channel_vectors.append(np.asarray(fv, dtype=np.float64))
                    else:
                        break

                if len(channel_vectors) == 3:
                    features_list.append(np.concatenate(channel_vectors))
                    valid_labels.append(lbl)

            else:
                # --- Grayscale extraction ---
                # Handle float32 images for PIL
                img_save = img
                if img_save.dtype != np.uint8:
                    img_save = (img_save * 255).clip(0, 255).astype(np.uint8)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    Image.fromarray(img_save).save(f.name)
                    tmp_path = f.name

                try:
                    img_result = tools["image_loader"].invoke({
                        "image_path": tmp_path,
                        "normalize": True,
                        "grayscale": True,
                    })
                    if not img_result.get("success"):
                        continue

                    image_array = img_result["image_array"]

                    if is_image_based:
                        tool_args = {"image_array": image_array}
                        tool_args.update(clean_params)
                        result = tools[descriptor_name].invoke(tool_args)
                    else:
                        ph_result = tools["compute_ph"].invoke({
                            "image_array": image_array,
                            "filtration_type": "sublevel",
                            "max_dimension": 1,
                        })
                        if not ph_result.get("success"):
                            continue
                        tool_args = {"persistence_data": ph_result["persistence"]}
                        tool_args.update(clean_params)
                        result = tools[descriptor_name].invoke(tool_args)

                    if result.get("success"):
                        fv = result.get("combined_vector") or result.get("feature_vector")
                        if fv is not None:
                            features_list.append(np.asarray(fv, dtype=np.float64))
                            valid_labels.append(lbl)
                finally:
                    os.unlink(tmp_path)
        except Exception:
            pass

    if len(features_list) < 10:
        print(f"  Only {len(features_list)} valid features -- not enough for CV")
        return 0.0

    X = np.array(features_list)
    y = np.array(valid_labels)
    print(f"  Feature matrix: {X.shape}, labels: {y.shape}")
    print(f"  Extraction took: {time.time()-t0:.1f}s")

    X = np.clip(X, -1e10, 1e10)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_train_s = np.clip(X_train_s, -100, 100)
        X_test_s = np.clip(X_test_s, -100, 100)

        clf = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="mlogloss", verbosity=0,
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc = balanced_accuracy_score(y_test, y_pred) * 100
        fold_accs.append(acc)
        print(f"    Fold {fold_i+1}: {acc:.1f}%")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"  Mean accuracy: {mean_acc:.1f}% (+/- {std_acc:.1f}%)")
    return round(mean_acc, 2)


# -- Entry Point ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TopoAgent E2E Case Study")
    parser.add_argument("--dataset", type=str, default="BloodMNIST",
                        choices=list(DATASET_INFO.keys()))
    parser.add_argument("--index", type=int, default=0,
                        help="Sample index within the dataset")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of samples for classifier evaluation")
    parser.add_argument("--all", action="store_true",
                        help="Run case study for all 26 datasets")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip classifier evaluation (faster)")
    parser.add_argument("--agentic", action="store_true",
                        help="Use agentic v7 mode (LLM drives tool execution)")
    parser.add_argument("--v8", action="store_true",
                        help="Use agentic v8 mode (5-phase pipeline with LTM)")
    parser.add_argument("--v9", action="store_true",
                        help="Use agentic v9 mode (6-phase hypothesis-first pipeline)")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="Time limit in seconds for agentic pipeline (default: 60)")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of images per dataset for multi-image demo (default: 1)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model name (default: gpt-4o). "
                             "Examples: gpt-4o, claude-sonnet-4-20250514, "
                             "gemini-2.5-pro-preview-06-05")
    args = parser.parse_args()

    if args.all:
        summaries = []
        for ds in DATASET_INFO:
            try:
                s = run_case_study(
                    ds, sample_index=0, n_eval=args.n_eval,
                    skip_eval=args.no_eval, agentic=args.agentic,
                    agentic_v8=args.v8,
                    agentic_v9=args.v9,
                    time_limit=args.time_limit,
                    n_samples=args.n_samples,
                    model_name=args.model,
                )
                summaries.append(s)
            except Exception as e:
                print(f"\n  ERROR on {ds}: {e}")
                summaries.append({
                    "dataset": ds,
                    "object_type": DATASET_INFO[ds]["object_type"],
                    "color_mode": DATASET_INFO[ds]["color_mode_str"],
                    "error": str(e),
                    "reasoning": {"object_type": DATASET_INFO[ds]["object_type"],
                                  "descriptor_choice": "ERROR"},
                    "output": {"feature_dimension": 0},
                    "classifier": {"accuracy": 0.0},
                    "metadata": {"total_time_seconds": 0, "n_llm_calls": 0},
                    "reflection": {"quality_ok": False},
                })

        # Detect multi-image format vs single-image format
        is_multi = args.n_samples > 1 and (args.v8 or args.v9)

        print("\n" + "=" * 110)
        print(f"  COMBINED RESULTS (all {len(summaries)} datasets"
              f"{f', {args.n_samples} images each' if is_multi else ''})")
        print("=" * 110)

        if is_multi:
            # Multi-image summary table
            header = (f"  {'Dataset':<22} {'ObjType':<16} {'Descriptors':<30} "
                      f"{'Stances':<20} {'LLM':>5} {'Retry':>5} "
                      f"{'Time':>7}")
            print(header)
            print("  " + "-" * 108)
            total_retries = 0
            total_llm_calls = 0
            all_descriptors = set()
            for s in summaries:
                ds_name = s.get('dataset', '?')
                if 'error' in s:
                    print(f"  {ds_name:<22} ERROR: {s['error'][:60]}")
                    continue
                obj_type = s.get('object_type', '?')
                agg = s.get('aggregate', {})
                descs = agg.get('descriptors_used', [])
                all_descriptors.update(descs)
                unique_descs = list(dict.fromkeys(descs))  # preserve order, dedupe
                desc_str = ','.join(unique_descs)
                stances = [r.get('plan', {}).get('stance', '?')
                           for r in s.get('per_image_results', [])]
                unique_stances = list(dict.fromkeys(stances))
                stance_str = ','.join(unique_stances)
                llm_calls = agg.get('llm_call_counts', [])
                n_llm = sum(llm_calls)
                retries = agg.get('retries', 0)
                total_retries += retries
                total_llm_calls += n_llm
                t_sec = agg.get('total_time', 0)
                print(f"  {ds_name:<22} {obj_type:<16} {desc_str:<30} "
                      f"{stance_str:<20} {n_llm:>5} {retries:>5} "
                      f"{t_sec:>6.1f}s")

            print(f"\n  Paper metrics:")
            print(f"    Datasets: {len(summaries)}")
            print(f"    Unique descriptors used: {len(all_descriptors)} — {sorted(all_descriptors)}")
            print(f"    Total LLM calls: {total_llm_calls}")
            print(f"    Total retries: {total_retries}")
        else:
            # Single-image summary table (legacy)
            header = (f"  {'Dataset':<22} {'ObjType(LLM)':<18} {'Stance':<8} "
                      f"{'Descriptor':<25} {'Dim':>5} "
                      f"{'Time':>6} {'LLM':>4} {'OT':>4}")
            print(header)
            print("  " + "-" * 100)
            n_ot_correct = 0
            n_follow = 0
            n_total = 0
            for s in summaries:
                ds_name = s.get('dataset', '?')
                dec = s.get('decisions', {})
                obj_type = dec.get('object_type', '') or s.get('object_type', '?')
                stance = dec.get('benchmark_stance', '')[:7]
                ot_correct = dec.get('object_type_correct')
                ot_mark = 'OK' if ot_correct else 'MISS' if ot_correct is False else '?'
                desc = s.get('reasoning', {}).get('descriptor_choice', 'ERROR')
                dim = s.get('output', {}).get('feature_dimension', 0)
                t_sec = s.get('metadata', {}).get('total_time_seconds', 0)
                n_llm = s.get('metadata', {}).get('n_llm_calls', 0)
                print(f"  {ds_name:<22} {obj_type:<18} {stance:<8} "
                      f"{desc:<25} {dim:>5}D "
                      f"{t_sec:>5.1f}s {n_llm:>3} "
                      f"{ot_mark:>4}")
                n_total += 1
                if ot_correct is True:
                    n_ot_correct += 1
                if stance.upper().startswith('FOLLOW'):
                    n_follow += 1

            print(f"\n  Paper metrics:")
            print(f"    Object type accuracy: {n_ot_correct}/{n_total}")
            print(f"    FOLLOW rate: {n_follow}/{n_total}")
            print(f"    DEVIATE rate: {n_total - n_follow}/{n_total}")

        outdir = PROJECT_ROOT / "results" / "case_studies"
        outdir.mkdir(parents=True, exist_ok=True)
        json_name = (f"all_multi_image_{args.n_samples}samples.json"
                     if is_multi else "all_explicit_case_studies.json")
        with open(outdir / json_name, "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"\n  Saved to: {outdir / json_name}")
    else:
        run_case_study(
            args.dataset, sample_index=args.index,
            n_eval=args.n_eval, skip_eval=args.no_eval,
            agentic=args.agentic, agentic_v8=args.v8,
            agentic_v9=args.v9,
            time_limit=args.time_limit,
            n_samples=args.n_samples,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
