#!/usr/bin/env python3
"""TopoBenchmark Experiment — 4-Method Descriptor Selection Comparison.

Compares TopoAgent v7 against three baselines on TDA descriptor selection
quality for medical image classification, using LIVE feature extraction
and cross-validation (no pre-computed cache).

Methods:
    topoagent  — GPT-4o + full TDA skills + PH signals (agentic v8 pipeline)
    gpt4o      — GPT-4o zero-shot (descriptor list only, no TDA knowledge)
    medrax     — GPT-4o + MedRAX medical persona + aggregated PH stats
    rule_based — Deterministic lookup from Benchmark3 rankings

Usage:
    # Single dataset (for SGE parallelism)
    python TopoBenchmark/run_experiment.py --dataset BloodMNIST

    # Smoke test
    python TopoBenchmark/run_experiment.py --dataset BloodMNIST --n-eval 50 \
        --output results/topobenchmark/experiment_test/

    # Specific methods only
    python TopoBenchmark/run_experiment.py --dataset BloodMNIST \
        --methods topoagent,rule_based
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# -- Project setup ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from topoagent.skills.rules_data import (
    DATASET_TO_OBJECT_TYPE,
    DATASET_COLOR_MODE,
    SUPPORTED_DESCRIPTORS,
    get_top_descriptors,
)
from TopoBenchmark.config import DATASET_DESCRIPTIONS, build_agent_query

# All 15 descriptors (ATOL/persistence_codebook have per-image fitting fallback)
TRAINING_FREE_DESCRIPTORS = SUPPORTED_DESCRIPTORS

# Image-based descriptors that skip PH (from benchmark_runner.py)
IMAGE_BASED_DESCRIPTORS = {
    "minkowski_functionals",
    "euler_characteristic_curve",
    "euler_characteristic_transform",
    "edge_histogram",
    "lbp_texture",
}

# Classifiers for evaluation
EVAL_CLASSIFIERS = ["TabPFN", "XGBoost", "CatBoost", "RandomForest"]


# ===========================================================================
# Utility helpers
# ===========================================================================

def _ensure_benchmark4_path():
    """Add RuleBenchmark/benchmark4 to sys.path if needed."""
    b4_dir = PROJECT_ROOT / "RuleBenchmark" / "benchmark4"
    if str(b4_dir) not in sys.path:
        sys.path.insert(0, str(b4_dir))


def _parse_json_response(text: str) -> dict:
    """Try to parse JSON from an LLM response (may have markdown fences)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def save_temp_image(image_array: np.ndarray, dataset_name: str, index: int) -> str:
    """Save image to a temp directory and return the path."""
    outdir = PROJECT_ROOT / "results" / "topobenchmark" / "experiment" / "tmp_images"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dataset_name}_sample{index}.png"
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(image_array).save(str(path))
    return str(path)


def load_dataset_unified(dataset_name: str, n_samples: int = 200, seed: int = 42):
    """Load images from any of the 26 datasets using benchmark4's loader."""
    _ensure_benchmark4_path()
    from data_loader import load_dataset

    color_mode = DATASET_COLOR_MODE.get(dataset_name, "grayscale")
    images, labels, class_names = load_dataset(
        dataset_name, n_samples=n_samples, seed=seed, color_mode=color_mode,
    )
    n_channels = 3 if images.ndim == 4 and images.shape[-1] == 3 else 1
    return images, labels, class_names, n_channels


# ===========================================================================
# Phase A: Descriptor Selection (4 methods)
# ===========================================================================

def select_topoagent(dataset_name: str, images: np.ndarray, labels: np.ndarray,
                     n_demo: int = 3, seed: int = 42, time_limit: float = 120.0) -> dict:
    """TopoAgent v8 — genuinely agentic 5-phase pipeline on demo images, majority vote."""
    from topoagent.agent import create_topoagent

    object_type = DATASET_TO_OBJECT_TYPE[dataset_name]
    n_channels = 3 if images.ndim == 4 and images.shape[-1] == 3 else 1
    desc_dict = DATASET_DESCRIPTIONS.get(dataset_name)

    agent = create_topoagent(
        model_name="gpt-4o",
        agentic_v8=True,
        time_limit_seconds=time_limit,
    )

    # Build query — intentionally generic so LLM cannot shortcut by dataset name
    parts = []
    if desc_dict:
        if "domain" in desc_dict:
            parts.append(f"Domain: {desc_dict['domain']}")
        if "description" in desc_dict:
            parts.append(desc_dict["description"])
        if "what_matters" in desc_dict:
            parts.append(f"Key features: {desc_dict['what_matters']}")
        if "n_classes" in desc_dict:
            parts.append(f"{desc_dict['n_classes']} classes")
    desc_part = "; ".join(parts)
    context_line = f"\n\nDataset context: {desc_part}" if desc_part else ""
    query = (
        f"Analyze the provided medical image, compute its persistent homology, "
        f"and determine the most suitable topology descriptor to produce a "
        f"fixed-length feature vector."
        f"{context_line}"
    )

    # Run on n_demo sample images
    rng = np.random.RandomState(seed)
    demo_indices = rng.choice(len(images), size=min(n_demo, len(images)), replace=False)

    descriptors_chosen = []
    all_results = []
    total_time = 0.0

    for idx in demo_indices:
        img_path = save_temp_image(images[idx], dataset_name, int(idx))
        t0 = time.time()
        try:
            result = agent.classify(image_path=img_path, query=query)
            elapsed = time.time() - t0
            total_time += elapsed

            # Extract descriptor from result
            report = result.get("report") or result.get("raw_answer") or {}
            if isinstance(report, dict):
                desc = report.get("descriptor")
            else:
                desc = result.get("classification")

            if desc and desc in TRAINING_FREE_DESCRIPTORS:
                descriptors_chosen.append(desc)
            else:
                # Fallback: try to find in tools_used
                for tool in reversed(result.get("tools_used", [])):
                    if tool in TRAINING_FREE_DESCRIPTORS:
                        desc = tool
                        descriptors_chosen.append(desc)
                        break

            all_results.append({
                "index": int(idx),
                "descriptor": desc,
                "observe_decisions": result.get("_observe_decisions"),
                "benchmark_stance": result.get("_benchmark_stance"),
                "ph_signals": result.get("_ph_signals"),
                "v8_analysis": result.get("_v8_analysis_context"),
                "v8_plan": result.get("_v8_plan_context"),
                "v8_experience": result.get("_v8_reflect_experience"),
                "tools_used": result.get("tools_used", []),
                "time_s": round(elapsed, 2),
            })
        except Exception as e:
            total_time += time.time() - t0
            print(f"    TopoAgent error on sample {idx}: {e}")
            all_results.append({"index": int(idx), "error": str(e)})

    # Majority vote
    if descriptors_chosen:
        descriptor = Counter(descriptors_chosen).most_common(1)[0][0]
    else:
        # Fallback to rule-based
        top = get_top_descriptors(object_type, n=1)
        descriptor = top[0]["descriptor"] if top else "persistence_image"

    return {
        "method": "topoagent",
        "descriptor": descriptor,
        "reasoning": f"Majority vote from {len(descriptors_chosen)} demo images: {descriptors_chosen}",
        "demo_results": all_results,
        "selection_time_s": round(total_time, 2),
        "n_demo": n_demo,
        "n_llm_calls": sum(
            len(r.get("tools_used", [])) for r in all_results if "tools_used" in r
        ),
    }


def select_gpt4o(dataset_name: str) -> dict:
    """GPT-4o zero-shot — no TDA knowledge, descriptor list only."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    desc_info = DATASET_DESCRIPTIONS.get(dataset_name, {})
    dataset_description = (
        f"Domain: {desc_info.get('domain', 'Medical')}\n"
        f"Description: {desc_info.get('description', dataset_name)}\n"
        f"What matters: {desc_info.get('what_matters', 'Unknown')}\n"
        f"Number of classes: {desc_info.get('n_classes', '?')}\n"
        f"Color mode: {desc_info.get('color_mode', '?')}\n"
        f"Object type: {desc_info.get('object_type', '?')}"
    )

    descriptor_list = "\n".join(f"- {d}" for d in TRAINING_FREE_DESCRIPTORS)

    # Reuse the BASELINE_ZEROSHOT_PROMPT pattern from benchmark_runner
    prompt = f"""Given a medical image dataset, select the most appropriate TDA (Topological Data Analysis) descriptor for maximizing classification accuracy.

## Dataset: {dataset_name}
{dataset_description}

## Available Descriptors
{descriptor_list}

## Instructions
Select the best descriptor for this dataset. Respond with ONLY valid JSON:

{{
  "descriptor_choice": "<descriptor name>",
  "reasoning": "<1-2 sentence justification>"
}}"""

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    t0 = time.time()
    response = model.invoke([HumanMessage(content=prompt)])
    elapsed = time.time() - t0

    parsed = _parse_json_response(response.content)
    descriptor = parsed.get("descriptor_choice") if parsed else None
    reasoning = parsed.get("reasoning", "") if parsed else response.content[:500]

    if descriptor not in TRAINING_FREE_DESCRIPTORS:
        descriptor = "persistence_image"  # safe fallback

    return {
        "method": "gpt4o",
        "descriptor": descriptor,
        "reasoning": reasoning,
        "raw_response": response.content,
        "selection_time_s": round(elapsed, 2),
        "n_llm_calls": 1,
    }


def select_medrax(dataset_name: str, images: np.ndarray, n_demo: int = 3,
                  seed: int = 42) -> dict:
    """MedRAX baseline — GPT-4o with medical persona + aggregated PH stats."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from topoagent.tools.homology.compute_ph import ComputePHTool

    desc_info = DATASET_DESCRIPTIONS.get(dataset_name, {})
    object_type = DATASET_TO_OBJECT_TYPE[dataset_name]

    # Compute PH stats on n_demo sample images and aggregate
    rng = np.random.RandomState(seed)
    demo_indices = rng.choice(len(images), size=min(n_demo, len(images)), replace=False)
    ph_tool = ComputePHTool()

    all_stats = []
    for idx in demo_indices:
        img = images[idx]
        # Convert to grayscale float
        if img.ndim == 3 and img.shape[2] == 3:
            gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        else:
            gray = img
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float32) / 255.0

        try:
            result = ph_tool._run(
                image_array=gray.tolist(),
                filtration_type="sublevel",
                max_dimension=1,
            )
            if result.get("success"):
                persistence = result["persistence"]
                h0_list = persistence.get("H0", [])
                h1_list = persistence.get("H1", [])
                h0_pers = [p["persistence"] for p in h0_list if np.isfinite(p["death"])]
                h1_pers = [p["persistence"] for p in h1_list if np.isfinite(p["death"])]
                all_stats.append({
                    "H0_count": len(h0_list),
                    "H1_count": len(h1_list),
                    "H0_avg_persistence": float(np.mean(h0_pers)) if h0_pers else 0.0,
                    "H1_avg_persistence": float(np.mean(h1_pers)) if h1_pers else 0.0,
                    "total_features": len(h0_list) + len(h1_list),
                    "h1_h0_ratio": round(len(h1_list) / max(len(h0_list), 1), 2),
                })
        except Exception:
            pass

    # Aggregate PH stats across samples
    if all_stats:
        agg = {
            "n_samples": len(all_stats),
            "avg_H0_count": round(np.mean([s["H0_count"] for s in all_stats]), 1),
            "avg_H1_count": round(np.mean([s["H1_count"] for s in all_stats]), 1),
            "avg_H0_persistence": round(np.mean([s["H0_avg_persistence"] for s in all_stats]), 6),
            "avg_H1_persistence": round(np.mean([s["H1_avg_persistence"] for s in all_stats]), 6),
            "avg_total_features": round(np.mean([s["total_features"] for s in all_stats]), 1),
            "avg_h1_h0_ratio": round(np.mean([s["h1_h0_ratio"] for s in all_stats]), 2),
        }
        ph_text = (
            f"Persistent Homology Statistics (aggregated from {agg['n_samples']} samples):\n"
            f"  H0 (connected components): avg {agg['avg_H0_count']} features\n"
            f"    avg persistence = {agg['avg_H0_persistence']:.6f}\n"
            f"  H1 (loops/holes): avg {agg['avg_H1_count']} features\n"
            f"    avg persistence = {agg['avg_H1_persistence']:.6f}\n"
            f"  Total features (avg): {agg['avg_total_features']}\n"
            f"  H1/H0 ratio (avg): {agg['avg_h1_h0_ratio']}"
        )
    else:
        ph_text = "PH computation failed on all samples."
        agg = {}

    descriptor_list = "\n".join(f"- {d}" for d in TRAINING_FREE_DESCRIPTORS)

    dataset_description = (
        f"Domain: {desc_info.get('domain', 'Medical')}\n"
        f"Description: {desc_info.get('description', dataset_name)}\n"
        f"What matters: {desc_info.get('what_matters', 'Unknown')}\n"
        f"Number of classes: {desc_info.get('n_classes', '?')}\n"
        f"Color mode: {desc_info.get('color_mode', '?')}\n"
        f"Object type: {desc_info.get('object_type', '?')}"
    )

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

    task_prompt = f"""Analyze this medical image dataset for topological feature extraction.

## Dataset: {dataset_name}
{dataset_description}

You are a medical AI agent with expertise in medical image analysis.
Make concrete decisions based on the dataset description AND the PH data below.

## Persistent Homology Data
{ph_text}

Analyze this PH profile from a medical perspective:
- What do the H0 components correspond to medically?
- What do the H1 loops correspond to medically?
- Is the persistence signal strong or noisy?

## Available Descriptors
{descriptor_list}

## Instructions
Based on your medical understanding AND the PH data, select the best TDA descriptor
for this dataset. Respond with ONLY valid JSON:

{{
  "medical_assessment": "<what medical structures are relevant>",
  "ph_analysis": "<2-3 sentences interpreting PH from medical perspective>",
  "descriptor_choice": "<chosen descriptor name>",
  "descriptor_reasoning": "<why this descriptor given PH + medical context>",
  "color_mode": "per_channel" or "grayscale"
}}"""

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    t0 = time.time()
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_prompt),
    ])
    elapsed = time.time() - t0

    parsed = _parse_json_response(response.content)
    descriptor = parsed.get("descriptor_choice") if parsed else None
    reasoning = parsed.get("descriptor_reasoning", "") if parsed else response.content[:500]

    if descriptor not in TRAINING_FREE_DESCRIPTORS:
        descriptor = "persistence_image"

    return {
        "method": "medrax",
        "descriptor": descriptor,
        "reasoning": reasoning,
        "ph_stats_aggregated": agg,
        "raw_response": response.content,
        "selection_time_s": round(elapsed, 2),
        "n_llm_calls": 1,
        "n_demo": n_demo,
    }


def select_rule_based(dataset_name: str) -> dict:
    """Rule-based — deterministic lookup from Benchmark3 rankings."""
    object_type = DATASET_TO_OBJECT_TYPE[dataset_name]
    top = get_top_descriptors(object_type, n=1, supported_only=True)
    descriptor = top[0]["descriptor"] if top else "persistence_image"
    accuracy = top[0]["accuracy"] if top else 0.0

    return {
        "method": "rule_based",
        "descriptor": descriptor,
        "reasoning": f"Top-1 for {object_type}: {descriptor} (benchmark accuracy: {accuracy:.1%})",
        "selection_time_s": 0.0,
        "n_llm_calls": 0,
    }


# ===========================================================================
# Phase B: Live Evaluation (shared across methods)
# ===========================================================================

def precompute_ph_once(images: np.ndarray, color_mode: str,
                       n_jobs: int = 4) -> dict:
    """Pre-compute PH diagrams once for all images.

    Args:
        n_jobs: Number of parallel jobs for cripser (default 4, safe for SGE).

    Returns a dict with keys:
        'gray' or 'R'/'G'/'B': list of diagram dicts
        'images_gray' or 'images_R'/'G'/'B': channel arrays
    """
    _ensure_benchmark4_path()
    from precompute_ph import compute_ph
    from data_loader import _to_grayscale_float

    cache = {}
    per_channel = (color_mode == "per_channel")

    if per_channel:
        if images.ndim == 4 and images.shape[3] == 3:
            R, G, B = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
        else:
            R = G = B = images if images.ndim == 3 else images[:, :, :, 0]

        cache["images_R"] = R
        cache["images_G"] = G
        cache["images_B"] = B

        print(f"  Computing PH for R channel ({len(R)} images)...")
        sys.stdout.flush()
        t0 = time.time()
        cache["diags_R"], _ = compute_ph(R, n_jobs=n_jobs)
        print(f"    R done in {time.time()-t0:.1f}s")
        sys.stdout.flush()

        t0 = time.time()
        cache["diags_G"], _ = compute_ph(G, n_jobs=n_jobs)
        print(f"    G done in {time.time()-t0:.1f}s")
        sys.stdout.flush()

        t0 = time.time()
        cache["diags_B"], _ = compute_ph(B, n_jobs=n_jobs)
        print(f"    B done in {time.time()-t0:.1f}s")
        sys.stdout.flush()
    else:
        if images.ndim == 4:
            gray = _to_grayscale_float(images)
        else:
            gray = images
        cache["images_gray"] = gray

        print(f"  Computing PH for grayscale ({len(gray)} images)...")
        sys.stdout.flush()
        t0 = time.time()
        cache["diags_gray"], _ = compute_ph(gray, n_jobs=n_jobs)
        print(f"    Done in {time.time()-t0:.1f}s")
        sys.stdout.flush()

    return cache


def live_evaluate(dataset_name: str, descriptor: str, labels: np.ndarray,
                  ph_cache: dict, n_eval: int = 200,
                  seed: int = 42, cv_folds: int = 5) -> tuple:
    """Extract features and evaluate via k-fold CV with 4 classifiers.

    Uses pre-computed PH diagrams from ph_cache to avoid redundant computation.

    Returns:
        (best_balanced_accuracy, details_dict)
    """
    _ensure_benchmark4_path()
    from descriptor_runner import extract_features, extract_features_per_channel
    from optimal_rules import get_rules
    from classifier_wrapper import get_classifier, get_available_classifiers
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    import torch

    object_type = DATASET_TO_OBJECT_TYPE[dataset_name]
    color_mode = DATASET_COLOR_MODE.get(dataset_name, "grayscale")

    # Get optimal params from exp4 rules
    rules = get_rules()
    config = rules.get_descriptor_config(descriptor, object_type, color_mode)
    params = config["params"]
    dim_per_channel = config["dim_per_channel"]
    total_dim = config["total_dim"]

    # Extract features using pre-computed PH
    is_ph_based = descriptor not in IMAGE_BASED_DESCRIPTORS
    per_channel = (color_mode == "per_channel")

    print(f"  Extracting {descriptor} features (dim={total_dim}, "
          f"color={color_mode}, PH={is_ph_based})...")
    t_feat = time.time()

    if per_channel:
        if is_ph_based:
            features = extract_features_per_channel(
                descriptor,
                diags_R=ph_cache["diags_R"],
                diags_G=ph_cache["diags_G"],
                diags_B=ph_cache["diags_B"],
                params=params, expected_dim_per_channel=dim_per_channel,
            )
        else:
            features = extract_features_per_channel(
                descriptor,
                images_R=ph_cache["images_R"],
                images_G=ph_cache["images_G"],
                images_B=ph_cache["images_B"],
                params=params, expected_dim_per_channel=dim_per_channel,
            )
    else:
        if is_ph_based:
            features = extract_features(
                descriptor, diagrams=ph_cache["diags_gray"],
                params=params, expected_dim=dim_per_channel,
            )
        else:
            features = extract_features(
                descriptor, images=ph_cache["images_gray"],
                params=params, expected_dim=dim_per_channel,
            )

    feat_time = time.time() - t_feat
    print(f"  Feature extraction: {feat_time:.1f}s, shape={features.shape if features is not None else 'None'}")

    if features is None or len(features) == 0:
        return 0.0, {"error": "Feature extraction returned empty", "feature_dim": 0}

    # Encode labels
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # Filter to available classifiers
    available = get_available_classifiers()
    classifiers_to_run = [c for c in EVAL_CLASSIFIERS if c in available]
    if not classifiers_to_run:
        classifiers_to_run = ["RandomForest"]

    # Skip TabPFN when no GPU — it hangs in ECOC mode for 11+ classes
    if not torch.cuda.is_available() and "TabPFN" in classifiers_to_run:
        classifiers_to_run.remove("TabPFN")
        print(f"  Skipping TabPFN (no GPU available)")
        sys.stdout.flush()

    # k-fold CV
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(features, labels_enc))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    per_classifier = {}
    for clf_name in classifiers_to_run:
        fold_ba_scores = []
        fold_acc_scores = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels_enc[train_idx], labels_enc[test_idx]

            try:
                clf = get_classifier(
                    clf_name,
                    n_features=features.shape[1],
                    n_classes=n_classes,
                    seed=seed,
                    device=device,
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                fold_ba_scores.append(balanced_accuracy_score(y_test, y_pred))
                fold_acc_scores.append(accuracy_score(y_test, y_pred))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and device == 'cuda':
                    torch.cuda.empty_cache()
                    try:
                        clf = get_classifier(
                            clf_name,
                            n_features=features.shape[1],
                            n_classes=n_classes,
                            seed=seed,
                            device='cpu',
                        )
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        fold_ba_scores.append(balanced_accuracy_score(y_test, y_pred))
                        fold_acc_scores.append(accuracy_score(y_test, y_pred))
                        device = 'cpu'
                    except Exception as e2:
                        per_classifier[clf_name] = {
                            "balanced_accuracy": 0.0, "accuracy": 0.0,
                            "std": 0.0, "error": str(e2),
                        }
                        fold_ba_scores = []
                        break
                else:
                    per_classifier[clf_name] = {
                        "balanced_accuracy": 0.0, "accuracy": 0.0,
                        "std": 0.0, "error": str(e),
                    }
                    fold_ba_scores = []
                    break
            except Exception as e:
                per_classifier[clf_name] = {
                    "balanced_accuracy": 0.0, "accuracy": 0.0,
                    "std": 0.0, "error": str(e),
                }
                fold_ba_scores = []
                break

        if fold_ba_scores:
            per_classifier[clf_name] = {
                "balanced_accuracy": float(np.mean(fold_ba_scores)),
                "accuracy": float(np.mean(fold_acc_scores)),
                "std": float(np.std(fold_ba_scores)),
                "fold_balanced_accuracy": [float(s) for s in fold_ba_scores],
                "fold_accuracy": [float(s) for s in fold_acc_scores],
            }

    # Find best classifier by balanced accuracy
    if not per_classifier:
        return 0.0, {"error": "All classifiers failed", "feature_dim": 0}

    best_clf = max(
        per_classifier,
        key=lambda c: per_classifier[c].get("balanced_accuracy", 0),
    )
    best_ba = per_classifier[best_clf]["balanced_accuracy"]
    best_acc = per_classifier[best_clf]["accuracy"]

    details = {
        "feature_dim": features.shape[1],
        "n_samples": len(features),
        "best_classifier": best_clf,
        "per_classifier": per_classifier,
        "best_balanced_accuracy": best_ba,
        "best_accuracy": best_acc,
        "feature_extraction_time_s": round(feat_time, 2),
    }

    return best_ba, details


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_experiment(dataset_name: str, methods: list, n_eval: int = 200,
                   n_demo: int = 3, seed: int = 42, output_dir: str = None):
    """Run the 4-method comparison experiment on one dataset."""
    output_dir = Path(output_dir or "results/topobenchmark/experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    object_type = DATASET_TO_OBJECT_TYPE[dataset_name]
    color_mode = DATASET_COLOR_MODE.get(dataset_name, "grayscale")

    print(f"\n{'='*80}")
    print(f"  Experiment: {dataset_name}")
    print(f"  Object type: {object_type}, Color: {color_mode}")
    print(f"  Methods: {methods}")
    print(f"  n_eval={n_eval}, n_demo={n_demo}, seed={seed}")
    print(f"{'='*80}\n")

    # Load dataset once (shared across all methods)
    print(f"Loading dataset {dataset_name} (n={n_eval})...")
    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=n_eval, seed=seed)
    print(f"  Loaded: {images.shape}, {len(class_names)} classes")

    # Pre-compute PH once (shared across all evaluations)
    print(f"\n--- Pre-computing PH (once for all methods) ---")
    t_ph = time.time()
    ph_cache = precompute_ph_once(images, color_mode)
    ph_time = time.time() - t_ph
    print(f"  PH pre-computation: {ph_time:.1f}s total")

    # Phase A: Descriptor selection per method
    eval_cache = {}  # descriptor -> (accuracy, details)
    results = {}

    for method in methods:
        print(f"\n--- Phase A: Descriptor Selection [{method}] ---")
        t0 = time.time()

        if method == "topoagent":
            selection = select_topoagent(
                dataset_name, images, labels, n_demo=n_demo,
                seed=seed, time_limit=120.0)
        elif method == "gpt4o":
            selection = select_gpt4o(dataset_name)
        elif method == "medrax":
            selection = select_medrax(
                dataset_name, images, n_demo=n_demo, seed=seed)
        elif method == "rule_based":
            selection = select_rule_based(dataset_name)
        else:
            print(f"  Unknown method: {method}, skipping")
            continue

        descriptor = selection["descriptor"]
        print(f"  Selected: {descriptor}")
        print(f"  Reasoning: {selection.get('reasoning', '')[:200]}")
        print(f"  Selection time: {selection.get('selection_time_s', 0):.1f}s")

        # Phase B: Live evaluation (cached by descriptor, uses pre-computed PH)
        print(f"\n--- Phase B: Live Evaluation [{method} -> {descriptor}] ---")
        if descriptor in eval_cache:
            print(f"  Using cached evaluation for {descriptor}")
            best_ba, details = eval_cache[descriptor]
        else:
            print(f"  Running live evaluation for {descriptor}...")
            t_eval = time.time()
            best_ba, details = live_evaluate(
                dataset_name, descriptor, labels=labels,
                ph_cache=ph_cache, n_eval=n_eval,
                seed=seed, cv_folds=5)
            eval_time = time.time() - t_eval
            details["eval_time_s"] = round(eval_time, 2)
            eval_cache[descriptor] = (best_ba, details)
            print(f"  Best balanced accuracy: {best_ba:.4f} ({details.get('best_classifier', '?')})")
            print(f"  Eval time: {eval_time:.1f}s")

        # Compose result JSON
        result = {
            "dataset": dataset_name,
            "method": method,
            "descriptor": descriptor,
            "object_type": object_type,
            "color_mode": color_mode,
            "best_balanced_accuracy": round(details.get("best_balanced_accuracy", best_ba), 4),
            "best_accuracy": round(details.get("best_accuracy", 0), 4),
            "best_classifier": details.get("best_classifier", "?"),
            "per_classifier": details.get("per_classifier", {}),
            "feature_dim": details.get("feature_dim", 0),
            "selection_reasoning": selection.get("reasoning", ""),
            "selection_time_s": selection.get("selection_time_s", 0),
            "eval_time_s": details.get("eval_time_s", 0),
            "n_eval": n_eval,
            "n_demo": selection.get("n_demo", 0),
            "n_llm_calls": selection.get("n_llm_calls", 0),
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }

        # Add method-specific details
        if method == "topoagent":
            result["demo_results"] = selection.get("demo_results", [])
        elif method == "medrax":
            result["ph_stats_aggregated"] = selection.get("ph_stats_aggregated", {})

        # Save per-method result
        out_path = output_dir / f"{dataset_name}_{method}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {out_path}")

        results[method] = result

    # Print summary
    print(f"\n{'='*80}")
    print(f"  Summary: {dataset_name}")
    print(f"{'='*80}")
    print(f"  {'Method':<15} {'Descriptor':<28} {'BalAcc':>8} {'Acc':>8} {'Classifier':<12} {'Time':>6}")
    print(f"  {'-'*85}")
    for method in methods:
        if method in results:
            r = results[method]
            print(f"  {r['method']:<15} {r['descriptor']:<28} "
                  f"{r['best_balanced_accuracy']:>7.4f} {r['best_accuracy']:>7.4f} "
                  f"{r['best_classifier']:<12} {r['selection_time_s']:>5.1f}s")

    return results


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TopoBenchmark Experiment: 4-method descriptor selection comparison")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., BloodMNIST)")
    parser.add_argument("--methods", type=str,
                        default="topoagent,gpt4o,medrax,rule_based",
                        help="Comma-separated methods to run")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of evaluation samples (default: 200)")
    parser.add_argument("--n-demo", type=int, default=3,
                        help="Number of demo images for TopoAgent/MedRAX (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str,
                        default="results/topobenchmark/experiment/",
                        help="Output directory")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]

    # Validate dataset
    if args.dataset not in DATASET_TO_OBJECT_TYPE:
        print(f"ERROR: Unknown dataset '{args.dataset}'")
        print(f"Available: {sorted(DATASET_TO_OBJECT_TYPE.keys())}")
        sys.exit(1)

    run_experiment(
        dataset_name=args.dataset,
        methods=methods,
        n_eval=args.n_eval,
        n_demo=args.n_demo,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
