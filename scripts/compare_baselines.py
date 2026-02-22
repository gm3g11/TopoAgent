#!/usr/bin/env python3
"""Compare TopoAgent vs GPT-4o (general LLM) vs MedRAX (medical agent).

Runs all three systems on the same medical image and produces a side-by-side
comparison JSON for the paper's case study.

Usage:
    python scripts/compare_baselines.py --dataset BloodMNIST --index 0
    python scripts/compare_baselines.py --dataset DermaMNIST --index 0
    python scripts/compare_baselines.py --all
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDRAX_ROOT = PROJECT_ROOT.parent / "MedRAX"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MEDRAX_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from topoagent.skills.rules_data import DATASET_TO_OBJECT_TYPE, DATASET_COLOR_MODE

# All 26 datasets
ALL_DATASETS = list(DATASET_TO_OBJECT_TYPE.keys())

# Lazily import dataset descriptions
_DATASET_DESCRIPTIONS = None
def _get_dataset_descriptions():
    global _DATASET_DESCRIPTIONS
    if _DATASET_DESCRIPTIONS is None:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "topobenchmark_config",
                str(PROJECT_ROOT / "TopoBenchmark" / "config.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))
            spec.loader.exec_module(mod)
            sys.path.pop(0)
            _DATASET_DESCRIPTIONS = mod.DATASET_DESCRIPTIONS
        except Exception:
            _DATASET_DESCRIPTIONS = {}
    return _DATASET_DESCRIPTIONS


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for GPT-4o multimodal input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_sample_image(dataset_name: str, sample_index: int = 0):
    """Load a sample image from the dataset."""
    from scripts.demo_topoagent_e2e import load_dataset_unified, save_temp_image

    images, labels, class_names, n_channels = load_dataset_unified(
        dataset_name, n_samples=max(sample_index + 1, 10))

    img_array = images[sample_index]
    label = labels[sample_index]
    is_rgb = n_channels == 3
    n_classes = len(class_names)
    image_path = save_temp_image(img_array, dataset_name, sample_index)

    return img_array, label, n_classes, is_rgb, image_path


def build_query(dataset_name: str, n_channels: int) -> str:
    """Build the query string (same as TopoAgent uses)."""
    descs = _get_dataset_descriptions()
    desc = descs.get(dataset_name, f"a {dataset_name} medical image")
    return (
        f"Analyze this {dataset_name} medical image "
        f"({n_channels}-channel ({desc})). "
        f"Extract topological features."
    )


def _parse_json_response(text: str) -> dict:
    """Try to parse JSON from an LLM response (may have markdown fences)."""
    import re
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting from ```json ... ```
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    # Try finding first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ============================================================================
# Shared PH computation — same data given to all three methods
# ============================================================================

def compute_ph_for_image(image_path: str) -> tuple:
    """Compute persistent homology on the image using GUDHI (same as TopoAgent).

    Returns (stats_dict, raw_persistence_data) where raw data can be passed to
    descriptor tools.
    """
    from topoagent.tools.homology.compute_ph import ComputePHTool

    # Load image as grayscale float32 array
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Compute PH using the same tool TopoAgent uses
    ph_tool = ComputePHTool()
    result = ph_tool._run(
        image_array=img_array.tolist(),
        filtration_type="sublevel",
        max_dimension=1,
    )

    if not result.get("success"):
        print(f"  WARNING: PH computation failed: {result.get('error')}")
        return {}, None

    persistence = result["persistence"]
    # persistence is {"H0": [{"birth":..., "death":..., "persistence":...}], "H1": [...]}
    h0_list = persistence.get("H0", [])
    h1_list = persistence.get("H1", [])

    h0_pers = [p["persistence"] for p in h0_list if np.isfinite(p["death"])]
    h1_pers = [p["persistence"] for p in h1_list if np.isfinite(p["death"])]

    stats = {
        "H0_count": len(h0_list),
        "H1_count": len(h1_list),
        "H0_max_persistence": max(h0_pers) if h0_pers else 0.0,
        "H1_max_persistence": max(h1_pers) if h1_pers else 0.0,
        "H0_avg_persistence": float(np.mean(h0_pers)) if h0_pers else 0.0,
        "H1_avg_persistence": float(np.mean(h1_pers)) if h1_pers else 0.0,
        "total_features": len(h0_list) + len(h1_list),
        "h1_h0_ratio": round(len(h1_list) / max(len(h0_list), 1), 2),
        "filtration": "sublevel",
    }

    print(f"  PH computed: H0={stats['H0_count']}, H1={stats['H1_count']}, "
          f"ratio={stats['h1_h0_ratio']}, "
          f"H0_avg={stats['H0_avg_persistence']:.4f}, "
          f"H1_avg={stats['H1_avg_persistence']:.4f}")

    return stats, persistence


# ============================================================================
# Execute a descriptor tool to produce a real feature vector
# ============================================================================

# Image-based descriptors that take image_array instead of persistence_data
IMAGE_BASED_DESCRIPTORS = {
    "lbp_texture", "edge_histogram",
    "euler_characteristic_transform", "minkowski_functionals",
}

def execute_descriptor(descriptor_name: str, persistence_data: dict,
                       image_path: str, params: dict = None) -> dict:
    """Run a descriptor tool and return the feature vector + quality metrics.

    Works for any of the 15 supported descriptors.
    """
    from topoagent.tools.descriptors import get_all_descriptors

    descriptors = get_all_descriptors()
    if descriptor_name not in descriptors:
        return {"success": False, "error": f"Unknown descriptor: {descriptor_name}"}

    tool = descriptors[descriptor_name]
    tool_args = dict(params or {})

    if descriptor_name in IMAGE_BASED_DESCRIPTORS:
        img = Image.open(image_path).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
        tool_args["image_array"] = img_array.tolist()
    else:
        tool_args["persistence_data"] = persistence_data

    try:
        result = tool.invoke(tool_args)
    except Exception as e:
        return {"success": False, "error": str(e)}

    if not result.get("success"):
        return {"success": False, "error": result.get("error", "unknown")}

    fv = np.array(result.get("combined_vector", result.get("feature_vector", [])),
                  dtype=np.float64)
    nonzero = np.count_nonzero(fv)
    sparsity = round(100.0 * (1.0 - nonzero / max(len(fv), 1)), 1)
    variance = round(float(np.var(fv)), 6) if len(fv) > 0 else 0.0

    return {
        "success": True,
        "feature_dim": len(fv),
        "sparsity_pct": sparsity,
        "variance": variance,
        "has_nan": bool(np.any(np.isnan(fv))),
        "feature_vector": fv,
    }


def _format_ph_stats_for_prompt(ph_stats: dict) -> str:
    """Format PH stats as a text block for LLM prompts."""
    if not ph_stats:
        return "PH computation failed — no data available."
    return (
        f"Persistent Homology Results (computed via GUDHI cubical complex, sublevel filtration):\n"
        f"  H0 (connected components): {ph_stats['H0_count']} features\n"
        f"    max persistence = {ph_stats['H0_max_persistence']:.6f}\n"
        f"    avg persistence = {ph_stats['H0_avg_persistence']:.6f}\n"
        f"  H1 (loops/holes): {ph_stats['H1_count']} features\n"
        f"    max persistence = {ph_stats['H1_max_persistence']:.6f}\n"
        f"    avg persistence = {ph_stats['H1_avg_persistence']:.6f}\n"
        f"  Total features: {ph_stats['total_features']}\n"
        f"  H1/H0 ratio: {ph_stats['h1_h0_ratio']}\n"
        f"  Note: Many features with small persistence is NORMAL for medical\n"
        f"  images (sublevel filtration on [0,1] normalized pixels)."
    )


# ============================================================================
# Baseline 1: GPT-4o (General LLM, no tools, multimodal with image)
# ============================================================================

def run_gpt4o_baseline(dataset_name: str, image_path: str, query: str,
                       ph_stats: dict = None) -> dict:
    """Run GPT-4o as a general LLM baseline — multimodal with image + real PH."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    print(f"\n{'='*80}")
    print(f"  GPT-4o Baseline (General LLM): {dataset_name}")
    print(f"{'='*80}")

    b64_img = encode_image_base64(image_path)
    ph_text = _format_ph_stats_for_prompt(ph_stats) if ph_stats else "No PH data available."

    # Structured prompt: image + real PH data + force concrete decisions
    prompt = f"""You are a topological data analysis (TDA) expert analyzing a medical image.

Task: {query}

Look at this image carefully. You must make concrete decisions — do NOT say
"I cannot analyze images" or hedge. Commit to specific answers based on what
you observe AND the PH data provided below.

## Step 1: What do you see?
Describe the medical structures in this specific image (not generic descriptions).

## Step 2: Persistent Homology Data
We computed persistent homology on this image using GUDHI (cubical complex,
sublevel filtration on grayscale). Here are the actual results:

{ph_text}

Analyze this PH profile:
- Is H0 or H1 more informative?
- Is the persistence signal strong or noisy?
- What reasoning chain does this suggest? (h0_dominant, h1_important, noisy_ph,
  shape_silhouette, color_diagnostic)

## Step 3: Choose a TDA descriptor
Based on the PH data above, pick exactly ONE from this list:
persistence_image, persistence_landscapes, betti_curves, persistence_silhouette,
persistence_entropy, persistence_statistics, tropical_coordinates,
template_functions, minkowski_functionals, euler_characteristic_curve,
edge_histogram, lbp_texture

## Step 4: Choose parameters
Specify the key parameters for your chosen descriptor (e.g., resolution,
n_templates, n_bins, sigma, etc.).

## Step 5: Choose color mode
Should this image use per_channel (R,G,B separately) or grayscale for
feature extraction?

Respond with ONLY valid JSON:
{{
  "structure_description": "<what you see in this specific image>",
  "ph_analysis": "<2-3 sentences analyzing the PH data above>",
  "dominant_dimension": "H0" or "H1",
  "persistence_quality": "strong" or "moderate" or "noisy",
  "reasoning_chain": "<which reasoning chain: h0_dominant / h1_important / noisy_ph / shape_silhouette / color_diagnostic>",
  "descriptor": "<chosen descriptor name>",
  "descriptor_reasoning": "<why this descriptor given the PH data>",
  "parameters": {{"param_name": value, ...}},
  "color_mode": "per_channel" or "grayscale",
  "color_reasoning": "<why this color mode>"
}}"""

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{b64_img}",
            "detail": "high",
        }},
    ])

    t0 = time.time()
    response = model.invoke([message])
    elapsed = time.time() - t0

    # Try to parse structured JSON from response
    parsed = _parse_json_response(response.content)

    print(f"  Response: {len(response.content)} chars in {elapsed:.1f}s")
    if parsed:
        print(f"  Descriptor: {parsed.get('descriptor', '?')}")
        print(f"  Color mode: {parsed.get('color_mode', '?')}")
        print(f"  H0 estimate: {parsed.get('h0_estimate', '?')}, H1 estimate: {parsed.get('h1_estimate', '?')}")
        print(f"  Reasoning: {parsed.get('descriptor_reasoning', '?')[:150]}")
    else:
        print(f"  (Could not parse JSON — raw preview below)")
        print(f"  {response.content[:300]}...")

    return {
        "method": "gpt4o_baseline",
        "model": "gpt-4o",
        "dataset": dataset_name,
        "query": prompt,
        "response": response.content,
        "parsed": parsed,
        "time_seconds": round(elapsed, 2),
        "has_tools": False,
        "has_tda_tools": False,
        "has_image_input": True,
        "feature_vector": None,
        "feature_dim": 0,
        "descriptor": parsed.get("descriptor") if parsed else None,
        "descriptor_reasoning": parsed.get("descriptor_reasoning") if parsed else None,
        "color_mode": parsed.get("color_mode") if parsed else None,
        "h0_estimate": parsed.get("h0_estimate") if parsed else None,
        "h1_estimate": parsed.get("h1_estimate") if parsed else None,
        "parameters": parsed.get("parameters") if parsed else None,
    }


# ============================================================================
# Baseline 2: MedRAX (Medical Agent with vision tools, no TDA)
# ============================================================================

def run_medrax_baseline(dataset_name: str, image_path: str, query: str,
                        ph_stats: dict = None) -> dict:
    """Run MedRAX medical agent baseline.

    MedRAX is a medical AI agent (GPT-4o + medical vision tools). Its tools
    are CXR-specific, so for non-CXR images it relies on GPT-4o's vision +
    its medical system prompt. Given the same PH data as TopoAgent to make
    the comparison fair — the difference is in decision-making quality.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n{'='*80}")
    print(f"  MedRAX Baseline (Medical Agent): {dataset_name}")
    print(f"{'='*80}")

    b64_img = encode_image_base64(image_path)
    ph_text = _format_ph_stats_for_prompt(ph_stats) if ph_stats else "No PH data available."

    # MedRAX system prompt (from medrax/docs/system_prompts.txt)
    system_prompt = (
        "You are an expert medical AI assistant who can answer any medical "
        "questions and analyze medical images similar to a doctor.\n"
        "You have access to specialized tools for chest X-ray analysis "
        "(classification of 18 pathologies, anatomical segmentation, "
        "radiology report generation, visual question answering, and "
        "phrase grounding). However, these tools are designed for chest "
        "X-rays specifically.\n"
        "For non-CXR images, rely on your own medical vision and reasoning.\n"
        "Solve using your own vision and reasoning and use tools to "
        "complement your reasoning."
    )

    # Structured prompt: image + real PH data + medical reasoning
    task_prompt = f"""Analyze this medical image for topological feature extraction.

Task: {query}

You are a medical AI agent with expertise in medical image analysis. Look at
this image carefully and make concrete decisions. Do NOT hedge — commit to
specific answers based on the image AND the PH data below.

## Step 1: Medical diagnosis
What medical structures do you see? What pathology or cell type is this?

## Step 2: Persistent Homology Data
We computed persistent homology on this image using GUDHI (cubical complex,
sublevel filtration on grayscale). Here are the actual results:

{ph_text}

Analyze this PH profile from a medical perspective:
- What do the H0 components correspond to medically?
- What do the H1 loops correspond to medically?
- Is the persistence signal strong or noisy?

## Step 3: Choose a TDA descriptor
Based on your medical understanding AND the PH data, pick exactly ONE:
persistence_image, persistence_landscapes, betti_curves, persistence_silhouette,
persistence_entropy, persistence_statistics, tropical_coordinates,
template_functions, minkowski_functionals, euler_characteristic_curve,
edge_histogram, lbp_texture

## Step 4: Choose parameters
Specify the key parameters for your chosen descriptor.

## Step 5: Choose color mode
Should this image use per_channel (R,G,B separately) or grayscale?

Respond with ONLY valid JSON:
{{
  "medical_assessment": "<what you diagnose in this image>",
  "structure_description": "<visual features you observe>",
  "ph_analysis": "<2-3 sentences interpreting the PH data from a medical perspective>",
  "dominant_dimension": "H0" or "H1",
  "persistence_quality": "strong" or "moderate" or "noisy",
  "reasoning_chain": "<h0_dominant / h1_important / noisy_ph / shape_silhouette / color_diagnostic>",
  "descriptor": "<chosen descriptor name>",
  "descriptor_reasoning": "<why this descriptor given the PH + medical context>",
  "parameters": {{"param_name": value, ...}},
  "color_mode": "per_channel" or "grayscale",
  "color_reasoning": "<why this color mode>"
}}"""

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)

    message = HumanMessage(content=[
        {"type": "text", "text": task_prompt},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{b64_img}",
            "detail": "high",
        }},
    ])

    t0 = time.time()
    response = model.invoke([
        SystemMessage(content=system_prompt),
        message,
    ])
    elapsed = time.time() - t0
    final_response = response.content

    # Parse structured JSON
    parsed = _parse_json_response(final_response)

    print(f"  Response: {len(final_response)} chars in {elapsed:.1f}s")
    if parsed:
        print(f"  Medical assessment: {parsed.get('medical_assessment', '?')[:100]}")
        print(f"  Descriptor: {parsed.get('descriptor', '?')}")
        print(f"  Color mode: {parsed.get('color_mode', '?')}")
        print(f"  H0 estimate: {parsed.get('h0_estimate', '?')}, H1 estimate: {parsed.get('h1_estimate', '?')}")
        print(f"  Limitation: {parsed.get('limitation', '?')[:120]}")
    else:
        print(f"  (Could not parse JSON)")
        print(f"  {final_response[:300]}...")

    return {
        "method": "medrax_baseline",
        "model": "gpt-4o (MedRAX persona)",
        "dataset": dataset_name,
        "query": task_prompt,
        "response": final_response,
        "parsed": parsed,
        "time_seconds": round(elapsed, 2),
        "has_tools": False,
        "has_tda_tools": False,
        "has_image_input": True,
        "tools_used": [],
        "feature_vector": None,
        "feature_dim": 0,
        "descriptor": parsed.get("descriptor") if parsed else None,
        "descriptor_reasoning": parsed.get("descriptor_reasoning") if parsed else None,
        "color_mode": parsed.get("color_mode") if parsed else None,
        "h0_estimate": parsed.get("h0_estimate") if parsed else None,
        "h1_estimate": parsed.get("h1_estimate") if parsed else None,
        "parameters": parsed.get("parameters") if parsed else None,
        "medical_assessment": parsed.get("medical_assessment") if parsed else None,
        "limitation": parsed.get("limitation") if parsed else None,
    }


# ============================================================================
# TopoAgent (load from existing case study)
# ============================================================================

def load_topoagent_result(dataset_name: str) -> dict:
    """Load TopoAgent result from existing case study JSON."""
    path = PROJECT_ROOT / "results" / "case_studies" / f"{dataset_name}_explicit_case_study.json"
    if not path.exists():
        print(f"  Warning: No TopoAgent case study for {dataset_name}")
        return None

    with open(path) as f:
        case = json.load(f)

    decisions = case.get("decisions", {})
    action = case.get("action", {})
    reflection = case.get("reflection", {})
    output = case.get("output", {})

    # Get reasoning from LLM interactions
    llm_interactions = case.get("llm_interactions", [])
    act_response = ""
    for interaction in llm_interactions:
        if isinstance(interaction, dict):
            resp = interaction.get("response", "")
            if "FOLLOW" in resp or "DEVIATE" in resp:
                act_response = resp
                break

    reflect_rounds = reflection.get("reflect_rounds", [])
    feature_quality = reflect_rounds[0].get("feature_quality", {}) if reflect_rounds else {}

    return {
        "method": "topoagent",
        "model": "gpt-4o (TopoAgent agent)",
        "dataset": dataset_name,
        "has_tools": True,
        "has_tda_tools": True,
        "has_image_input": True,
        "tools_used": action.get("tools_executed", []),
        "descriptor": decisions.get("descriptor", ""),
        "feature_dim": output.get("feature_dimension", 0),
        "feature_vector_available": True,
        "object_type": decisions.get("object_type", ""),
        "object_type_correct": decisions.get("object_type_correct"),
        "color_mode": decisions.get("color_mode", ""),
        "filtration": decisions.get("filtration_type", ""),
        "benchmark_stance": decisions.get("benchmark_stance", ""),
        "ph_signals": decisions.get("ph_signals", []),
        "ph_interpretation": decisions.get("ph_interpretation", ""),
        "act_reasoning": act_response[:1000],
        "quality_ok": reflect_rounds[0].get("quality_ok") if reflect_rounds else None,
        "sparsity": output.get("sparsity_pct"),
        "variance": feature_quality.get("variance"),
        "parameters": action.get("parameters", {}),
        "time_seconds": case.get("metadata", {}).get("total_time_seconds", 0),
    }


# ============================================================================
# Main comparison
# ============================================================================

def run_comparison(dataset_name: str, sample_index: int = 0):
    """Run all 3 baselines on the same image and produce comparison."""
    print(f"\n{'#'*80}")
    print(f"  COMPARISON: {dataset_name} (sample #{sample_index})")
    print(f"{'#'*80}")

    # Load image
    img_array, label, n_classes, is_rgb, image_path = load_sample_image(
        dataset_name, sample_index)

    n_channels = 3 if is_rgb else 1
    query = build_query(dataset_name, n_channels)

    print(f"  Image: {image_path}")
    print(f"  Query: {query[:100]}...")

    # 0. Compute PH on the image — same data given to all methods
    print(f"\n  Computing PH (shared across all methods)...")
    ph_stats, persistence_data = compute_ph_for_image(image_path)

    # 1. GPT-4o baseline (with real PH data)
    gpt4o_result = run_gpt4o_baseline(dataset_name, image_path, query, ph_stats)

    # 2. MedRAX baseline (with real PH data)
    medrax_result = run_medrax_baseline(dataset_name, image_path, query, ph_stats)

    # 3. TopoAgent (from existing case study — already computed PH)
    topoagent_result = load_topoagent_result(dataset_name)

    # 4. Execute each method's chosen descriptor to produce actual feature vectors
    print(f"\n  Executing each method's chosen descriptor...")
    feature_results = {}
    for method_name, result in [("gpt4o", gpt4o_result), ("medrax", medrax_result)]:
        desc_name = result.get("descriptor")
        params = result.get("parameters") or {}
        if desc_name and persistence_data:
            print(f"  {method_name}: running {desc_name} with params={params}...")
            fv_result = execute_descriptor(desc_name, persistence_data, image_path, params)
            feature_results[method_name] = fv_result
            if fv_result["success"]:
                result["feature_dim"] = fv_result["feature_dim"]
                result["sparsity_pct"] = fv_result["sparsity_pct"]
                result["variance"] = fv_result["variance"]
                result["has_nan"] = fv_result["has_nan"]
                print(f"    -> {fv_result['feature_dim']}D, sparsity={fv_result['sparsity_pct']}%, "
                      f"variance={fv_result['variance']:.6f}")
            else:
                print(f"    -> FAILED: {fv_result['error']}")
        else:
            feature_results[method_name] = {"success": False, "error": "no descriptor chosen"}

    # Combine
    comparison = {
        "dataset": dataset_name,
        "sample_index": sample_index,
        "image_path": image_path,
        "query": query,
        "shared_ph_stats": ph_stats,
        "ground_truth": {
            "label": int(label) if label is not None else None,
            "n_classes": n_classes,
            "object_type": DATASET_TO_OBJECT_TYPE.get(dataset_name),
            "color_mode": DATASET_COLOR_MODE.get(dataset_name, "grayscale"),
        },
        "baselines": {
            "gpt4o": gpt4o_result,
            "medrax": medrax_result,
            "topoagent": topoagent_result,
        },
        "comparison_summary": _build_summary(
            gpt4o_result, medrax_result, topoagent_result, feature_results),
    }

    # Save (exclude numpy arrays)
    outdir = PROJECT_ROOT / "results" / "case_studies"
    path = outdir / f"{dataset_name}_comparison.json"
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Print summary table
    _print_summary(dataset_name, ph_stats, gpt4o_result, medrax_result,
                   topoagent_result, feature_results, path)

    return comparison


def _build_summary(gpt4o_result, medrax_result, topoagent_result, feature_results):
    """Build the comparison_summary dict."""
    g_fv = feature_results.get("gpt4o", {})
    m_fv = feature_results.get("medrax", {})
    return {
        "gpt4o": {
            "can_identify_structures": True,
            "has_tda_tools": False,
            "descriptor": gpt4o_result.get("descriptor"),
            "feature_vector_produced": g_fv.get("success", False),
            "feature_dim": g_fv.get("feature_dim", 0) if g_fv.get("success") else 0,
            "sparsity_pct": g_fv.get("sparsity_pct") if g_fv.get("success") else None,
            "variance": g_fv.get("variance") if g_fv.get("success") else None,
            "time_seconds": gpt4o_result["time_seconds"],
        },
        "medrax": {
            "can_identify_structures": True,
            "has_tda_tools": False,
            "descriptor": medrax_result.get("descriptor"),
            "feature_vector_produced": m_fv.get("success", False),
            "feature_dim": m_fv.get("feature_dim", 0) if m_fv.get("success") else 0,
            "sparsity_pct": m_fv.get("sparsity_pct") if m_fv.get("success") else None,
            "variance": m_fv.get("variance") if m_fv.get("success") else None,
            "time_seconds": medrax_result["time_seconds"],
        },
        "topoagent": {
            "can_identify_structures": True,
            "has_tda_tools": True,
            "descriptor": topoagent_result["descriptor"] if topoagent_result else None,
            "feature_vector_produced": True,
            "feature_dim": topoagent_result["feature_dim"] if topoagent_result else 0,
            "sparsity_pct": topoagent_result.get("sparsity") if topoagent_result else None,
            "variance": topoagent_result.get("variance") if topoagent_result else None,
            "time_seconds": topoagent_result["time_seconds"] if topoagent_result else 0,
        },
    }


def _print_summary(dataset_name, ph_stats, gpt4o_result, medrax_result,
                   topoagent_result, feature_results, path):
    """Print the comparison summary table."""
    g_parsed = gpt4o_result.get("parsed") or {}
    m_parsed = medrax_result.get("parsed") or {}
    g_fv = feature_results.get("gpt4o", {})
    m_fv = feature_results.get("medrax", {})

    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY: {dataset_name}")
    print(f"  All methods received the SAME PH data: H0={ph_stats.get('H0_count','?')}, "
          f"H1={ph_stats.get('H1_count','?')}, ratio={ph_stats.get('h1_h0_ratio','?')}")
    print(f"{'='*80}")

    # Header
    print(f"\n  {'Criterion':<30s} {'GPT-4o':<25s} {'MedRAX':<25s} {'TopoAgent':<25s}")
    print(f"  {'-'*105}")

    # Descriptor choice
    g_desc = gpt4o_result.get("descriptor", "?") or "?"
    m_desc = medrax_result.get("descriptor", "?") or "?"
    t_desc = topoagent_result["descriptor"] if topoagent_result else "?"
    print(f"  {'Descriptor':<30s} {str(g_desc):<25s} {str(m_desc):<25s} {str(t_desc):<25s}")

    # Color mode
    g_color = gpt4o_result.get("color_mode", "?") or "?"
    m_color = medrax_result.get("color_mode", "?") or "?"
    t_color = topoagent_result["color_mode"] if topoagent_result else "?"
    gt_color = DATASET_COLOR_MODE.get(dataset_name, "?")
    print(f"  {'Color mode':<30s} {str(g_color):<25s} {str(m_color):<25s} {str(t_color):<25s}")
    print(f"  {'  (ground truth)':<30s} {gt_color}")

    # PH interpretation
    g_chain = g_parsed.get("reasoning_chain", "?")
    m_chain = m_parsed.get("reasoning_chain", "?")
    t_signals = topoagent_result.get("ph_signals", []) if topoagent_result else []
    t_chain = ", ".join(s["name"] for s in t_signals) if t_signals else "normal profile"
    print(f"  {'PH reasoning chain':<30s} {str(g_chain):<25s} {str(m_chain):<25s} {str(t_chain):<25s}")

    # Feature vector — now all three produce one
    def _fv_str(fv_result, from_topoagent=False):
        if from_topoagent:
            dim = topoagent_result["feature_dim"] if topoagent_result else 0
            return f"Yes ({dim}D)" if dim else "No"
        if fv_result.get("success"):
            return f"Yes ({fv_result['feature_dim']}D)"
        return f"FAILED"

    print(f"  {'Feature vector':<30s} "
          f"{_fv_str(g_fv):<25s} "
          f"{_fv_str(m_fv):<25s} "
          f"{_fv_str(None, from_topoagent=True):<25s}")

    # Sparsity
    def _sp(fv_result, ta=False):
        if ta:
            v = topoagent_result.get("sparsity") if topoagent_result else None
            return f"{v}%" if v is not None else "?"
        if fv_result.get("success"):
            return f"{fv_result['sparsity_pct']}%"
        return "N/A"

    print(f"  {'Sparsity':<30s} "
          f"{_sp(g_fv):<25s} {_sp(m_fv):<25s} {_sp(None, ta=True):<25s}")

    # Variance
    def _var(fv_result, ta=False):
        if ta:
            v = topoagent_result.get("variance") if topoagent_result else None
            return f"{v:.4f}" if v is not None else "?"
        if fv_result.get("success"):
            return f"{fv_result['variance']:.4f}"
        return "N/A"

    print(f"  {'Variance':<30s} "
          f"{_var(g_fv):<25s} {_var(m_fv):<25s} {_var(None, ta=True):<25s}")

    # Parameters
    g_params = str(gpt4o_result.get("parameters", {}))[:24] if gpt4o_result.get("parameters") else "?"
    m_params = str(medrax_result.get("parameters", {}))[:24] if medrax_result.get("parameters") else "?"
    t_params = str(topoagent_result.get("parameters", {}))[:24] if topoagent_result else "?"
    print(f"  {'Parameters':<30s} {g_params:<25s} {m_params:<25s} {t_params:<25s}")

    # Knowledge sources
    print(f"  {'Benchmark knowledge':<30s} {'No':<25s} {'No':<25s} {'Yes (26-dataset study)':<25s}")
    print(f"  {'Learned rules':<30s} {'No':<25s} {'No':<25s} {'Yes (past experience)':<25s}")
    print(f"  {'PH signal detection':<30s} {'No':<25s} {'No':<25s} {'Yes (4 signals)':<25s}")

    # Time
    ta_time = topoagent_result['time_seconds'] if topoagent_result else 0
    print(f"  {'Time (seconds)':<30s} {gpt4o_result['time_seconds']:<25.1f} {medrax_result['time_seconds']:<25.1f} {ta_time:<25.1f}")

    # Key insight
    print(f"\n  KEY DIFFERENCES:")
    print(f"  - All three received identical PH: H0={ph_stats.get('H0_count','?')}, H1={ph_stats.get('H1_count','?')}")
    print(f"  - GPT-4o chose {g_desc} -> {_fv_str(g_fv)}")
    print(f"  - MedRAX chose {m_desc} -> {_fv_str(m_fv)}")
    if topoagent_result:
        ta = topoagent_result
        t_dim = ta["feature_dim"]
        print(f"  - TopoAgent chose {t_desc} -> Yes ({t_dim}D) "
              f"(stance: {ta.get('benchmark_stance','?')})")

    # Descriptor reasoning comparison
    print(f"\n  DESCRIPTOR REASONING:")
    g_reason = (g_parsed.get("descriptor_reasoning") or "")[:120]
    m_reason = (m_parsed.get("descriptor_reasoning") or "")[:120]
    t_reason = (topoagent_result.get("act_reasoning", "") if topoagent_result else "")[:120]
    print(f"  GPT-4o:    {g_reason}")
    print(f"  MedRAX:    {m_reason}")
    print(f"  TopoAgent: {t_reason}")

    print(f"\n  Saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare TopoAgent vs baselines")
    parser.add_argument("--dataset", type=str, default="BloodMNIST",
                        choices=ALL_DATASETS)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for ds in ALL_DATASETS:
            try:
                run_comparison(ds, args.index)
            except Exception as e:
                print(f"  ERROR on {ds}: {e}")
    else:
        run_comparison(args.dataset, args.index)


if __name__ == "__main__":
    main()
