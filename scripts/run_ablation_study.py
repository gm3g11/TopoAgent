#!/usr/bin/env python3
"""Ablation Study for TopoAgent v8.1 Pipeline Components.

Evaluates 5 conditions:
  C0: Full v8.1 (baseline)
  C1: w/o Skills   — remove benchmark rankings/descriptor knowledge from PLAN
  C2: w/o Memory   — remove LTM + STM summaries from LLM prompts
  C3: w/o Reflect  — skip REFLECT phase entirely
  C4: w/o Analyze  — skip ANALYZE phase (deterministic object_type/color_mode)

For each condition:
  1. Run agentic pipeline on --n-samples demo images → get descriptor choice(s)
  2. Use majority-vote descriptor from demo images
  3. Run 3-fold XGBoost CV with that descriptor on --n-eval images
  4. Save results as JSON

Usage:
    # Single condition + dataset
    python scripts/run_ablation_study.py --condition c0 --dataset BloodMNIST

    # Quick test
    python scripts/run_ablation_study.py --condition c1 --dataset BloodMNIST \
        --n-samples 3 --n-eval 100

    # All conditions for one dataset
    python scripts/run_ablation_study.py --condition all --dataset BloodMNIST
"""

import argparse
import json
import os
import sys
import time
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# -- Setup -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from topoagent.skills.rules_data import (
    DATASET_TO_OBJECT_TYPE, DATASET_COLOR_MODE, SUPPORTED_DESCRIPTORS,
)

IMAGE_BASED_DESCRIPTORS = {
    "minkowski_functionals", "euler_characteristic_curve",
    "euler_characteristic_transform", "edge_histogram", "lbp_texture",
}

CONDITION_MAP = {
    "c0": {"label": "c0_full", "flags": {}},
    "c1": {"label": "c1_no_skills", "flags": {"ablate_skills": True}},
    "c2": {"label": "c2_no_memory", "flags": {"ablate_memory": True}},
    "c3": {"label": "c3_no_reflect", "flags": {"ablate_reflect": True}},
    "c4": {"label": "c4_no_analyze", "flags": {"ablate_analyze": True}},
}

DATASET_INFO = {
    name: {
        "object_type": DATASET_TO_OBJECT_TYPE[name],
        "color_mode": DATASET_COLOR_MODE.get(name, "grayscale"),
    }
    for name in DATASET_TO_OBJECT_TYPE
}


# -- Dataset helpers -------------------------------------------------------

def load_dataset_unified(dataset_name: str, n_samples: int = 10):
    """Load images from any of the 26 datasets using benchmark4's loader."""
    sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))
    try:
        from data_loader import load_dataset
    finally:
        sys.path.pop(0)

    color_mode = DATASET_INFO[dataset_name]["color_mode"]
    images, labels, class_names = load_dataset(
        dataset_name, n_samples=n_samples, seed=42, color_mode=color_mode,
    )
    n_channels = 3 if images.ndim == 4 and images.shape[-1] == 3 else 1
    return images, labels, class_names, n_channels


def save_temp_image(image_array: np.ndarray, dataset_name: str, index: int) -> str:
    """Save image to temp dir and return the path."""
    outdir = PROJECT_ROOT / "results" / "ablation_study" / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dataset_name}_sample{index}.png"
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(image_array)
    img.save(str(path))
    return str(path)


def _get_dataset_descriptions():
    """Lazily import DATASET_DESCRIPTIONS from TopoBenchmark/config.py."""
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
        return mod.DATASET_DESCRIPTIONS
    except Exception:
        return {}


def _build_query(dataset_name: str, object_type: str,
                 n_channels: int = 3, description=None) -> str:
    """Build a generic query for the agent."""
    if description and isinstance(description, dict):
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


# -- Demo phase: run agent on sample images --------------------------------

def run_demo_phase(dataset_name: str, condition: str, n_samples: int = 5,
                   time_limit: float = 120.0, verbose: bool = False):
    """Run the agentic pipeline on n_samples images for a given condition.

    Returns:
        List of per-image result dicts with descriptor choices.
    """
    from topoagent.agent import create_topoagent

    info = DATASET_INFO[dataset_name]
    object_type = info["object_type"]
    flags = CONDITION_MAP[condition]["flags"]

    # Load images
    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=n_samples + 5  # extra buffer
    )

    # Create agent with ablation flags
    agent = create_topoagent(
        model_name="gpt-4o",
        agentic_v8=True,
        time_limit_seconds=time_limit,
        **flags,
    )
    agent.workflow.verbose = verbose

    # Clear LTM for fresh start
    if agent._long_term_memory is not None:
        agent._long_term_memory.clear()

    # Get dataset descriptions for query
    descriptions = _get_dataset_descriptions()
    desc_text = descriptions.get(dataset_name, "")

    per_image_results = []

    for img_idx in range(min(n_samples, len(images))):
        sample_img = images[img_idx]
        sample_label = labels[img_idx]
        image_path = save_temp_image(sample_img, dataset_name, img_idx)

        query = _build_query(dataset_name, object_type,
                             n_channels=n_ch, description=desc_text)

        print(f"  [Demo {img_idx+1}/{n_samples}] {dataset_name} (condition={condition})")

        t0 = time.time()
        try:
            result = agent.classify(image_path=image_path, query=query)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"    ERROR: {e}")
            per_image_results.append({
                "image_index": img_idx,
                "descriptor": "ERROR",
                "error": str(e),
                "time_seconds": time.time() - t0,
            })
            continue

        # Extract descriptor from report
        report = result.get("report") or result.get("raw_answer") or {}
        descriptor = (report.get("descriptor", "unknown")
                      if isinstance(report, dict) else "unknown")

        v8_plan = result.get("_v8_plan_context") or {}
        v8_analysis = result.get("_v8_analysis_context") or {}
        perceive_decisions = result.get("_perceive_decisions") or {}
        reflect_history = result.get("_reflect_history", [])
        llm_interactions = result.get("llm_interactions", [])

        img_result = {
            "image_index": img_idx,
            "ground_truth_label": int(sample_label),
            "descriptor": descriptor,
            "stance": v8_plan.get("stance", "?"),
            "object_type": v8_analysis.get("object_type", "?"),
            "color_mode": v8_analysis.get("color_mode", "?"),
            "filtration_type": perceive_decisions.get("filtration_type", "sublevel"),
            "n_llm_calls": len(llm_interactions),
            "retry_used": len(reflect_history) > 1,
            "time_seconds": round(elapsed, 2),
        }
        per_image_results.append(img_result)

        print(f"    descriptor={descriptor}, stance={v8_plan.get('stance', '?')}, "
              f"LLM_calls={len(llm_interactions)}, time={elapsed:.1f}s")

    return per_image_results


# -- Eval phase: classifier evaluation ------------------------------------

def _extract_features_single(tools, gray_2d, descriptor_name, params):
    """Extract features from a single grayscale 2D image array.

    Returns feature vector (numpy 1D) or None on failure.
    """
    if gray_2d.dtype != np.uint8:
        gray_uint8 = (gray_2d * 255).clip(0, 255).astype(np.uint8)
    else:
        gray_uint8 = gray_2d

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.fromarray(gray_uint8).save(f.name)
        tmp_path = f.name

    try:
        img_result = tools["image_loader"].invoke({
            "image_path": tmp_path,
            "normalize": True,
            "grayscale": True,
        })
        if not img_result.get("success"):
            return None

        image_array = img_result["image_array"]

        if descriptor_name in IMAGE_BASED_DESCRIPTORS:
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

            tool_args = {"persistence_data": ph_result["persistence"]}
            for k, v in params.items():
                if k not in ("total_dim", "classifier", "color_mode", "dim"):
                    tool_args[k] = v
            result = tools[descriptor_name].invoke(tool_args)

        if result.get("success"):
            fv = result.get("combined_vector") or result.get("feature_vector")
            if fv is not None:
                return np.asarray(fv, dtype=np.float64)
        return None
    except Exception:
        return None
    finally:
        os.unlink(tmp_path)


def run_eval_phase(dataset_name: str, descriptor_name: str,
                   object_type: str, color_mode: str, n_eval: int = 500):
    """Run 3-fold stratified CV with TabPFN on extracted features.

    Uses the benchmark4 classifier_wrapper for consistency with v7 benchmark:
    - TabPFN (default) with PCA-bagging for >2000D features
    - XGBoost fallback for >2000D when TabPFN unavailable
    - ECOC mode for >10 classes

    Per-channel mode: processes R, G, B channels separately, extracts features
    from each channel with the SAME params, concatenates → total_dim = 3 × dim.

    Returns:
        (mean_accuracy, std_accuracy, fold_accuracies, classifier_name)
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import balanced_accuracy_score
    from topoagent.agent import create_topoagent
    from topoagent.skills import SkillRegistry

    # Import benchmark4 classifier wrapper (same as v7 benchmark_runner)
    sys.path.insert(0, str(PROJECT_ROOT / "RuleBenchmark" / "benchmark4"))
    try:
        from classifier_wrapper import get_classifier, get_available_classifiers
    except ImportError:
        print("  classifier_wrapper not available, skipping classifier eval")
        return 0.0, 0.0, []
    finally:
        sys.path.pop(0)

    print(f"  [Eval] Loading {n_eval} samples from {dataset_name}...")
    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=n_eval)

    # Create agent just for tool access (no LLM calls)
    agent = create_topoagent(model_name="gpt-4o", skills_mode=True)
    tools = agent.workflow.tools

    registry = SkillRegistry()
    params = registry.configure_after_selection(
        descriptor_name, object_type, color_mode)

    is_per_channel = (color_mode == "per_channel"
                      and images.ndim == 4
                      and images.shape[-1] == 3)

    print(f"  [Eval] Extracting {descriptor_name} features for {len(images)} images...")
    print(f"         params={params}")
    print(f"         per_channel={is_per_channel}")
    t0 = time.time()

    features_list = []
    valid_labels = []

    for i, (img, lbl) in enumerate(zip(images, labels)):
        if i > 0 and i % 100 == 0:
            print(f"    ... {i}/{len(images)} ({time.time()-t0:.0f}s)")

        try:
            if is_per_channel:
                # Per-channel: extract features for R, G, B separately, concatenate
                channel_features = []
                ok = True
                for ch in range(3):
                    ch_img = img[:, :, ch]  # 2D grayscale for this channel
                    fv = _extract_features_single(
                        tools, ch_img, descriptor_name, params)
                    if fv is None:
                        ok = False
                        break
                    channel_features.append(fv)
                if ok and len(channel_features) == 3:
                    features_list.append(np.concatenate(channel_features))
                    valid_labels.append(lbl)
            else:
                # Grayscale: convert to 2D if needed, extract once
                if img.ndim == 3:
                    gray = np.mean(img, axis=-1)
                else:
                    gray = img
                fv = _extract_features_single(
                    tools, gray, descriptor_name, params)
                if fv is not None:
                    features_list.append(fv)
                    valid_labels.append(lbl)
        except Exception:
            pass

    if len(features_list) < 10:
        print(f"  Only {len(features_list)} valid features -- not enough for CV")
        return 0.0, 0.0, [], "N/A"

    # Guard against inhomogeneous feature lengths (e.g., descriptor returning
    # variable-length vectors for certain images/channels)
    expected_dim = len(features_list[0])
    filtered_features = []
    filtered_labels = []
    for fv, lbl in zip(features_list, valid_labels):
        if len(fv) == expected_dim:
            filtered_features.append(fv)
            filtered_labels.append(lbl)
    if len(filtered_features) < len(features_list):
        print(f"  WARNING: {len(features_list) - len(filtered_features)} images "
              f"had inconsistent feature dim (expected {expected_dim}), dropped")
    features_list = filtered_features
    valid_labels = filtered_labels

    if len(features_list) < 10:
        print(f"  Only {len(features_list)} valid features after filtering -- not enough for CV")
        return 0.0, 0.0, [], "N/A"

    X = np.array(features_list)
    y = np.array(valid_labels)
    print(f"  Feature matrix: {X.shape}, labels: {y.shape}")
    print(f"  Extraction took: {time.time()-t0:.1f}s")

    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    total_dim = X.shape[1]

    # Classifier selection: same logic as v7 benchmark_runner
    # TabPFN by default (requires GPU); XGBoost fallback on CPU or >2000D
    # Use XGBoost for >10 classes to avoid slow TabPFN ECOC mode
    available = get_available_classifiers()

    import torch
    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"

    if "TabPFN" in available and has_gpu and n_classes <= 10:
        clf_name = "TabPFN"
    elif "XGBoost" in available:
        clf_name = "XGBoost"
    else:
        clf_name = "RandomForest"

    print(f"  Classifier: {clf_name} (dim={total_dim}, classes={n_classes}, device={device})")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accs = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        try:
            clf = get_classifier(
                clf_name,
                n_features=total_dim,
                n_classes=n_classes,
                seed=42,
                device=device,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred) * 100
            fold_accs.append(acc)
            print(f"    Fold {fold_i+1}: {acc:.1f}%")
        except Exception as e:
            # If TabPFN fails (e.g., CUDA error), fall back to XGBoost
            if clf_name == "TabPFN" and "XGBoost" in available:
                print(f"    Fold {fold_i+1}: TabPFN failed ({e}), falling back to XGBoost")
                try:
                    clf_name = "XGBoost"
                    clf = get_classifier(
                        "XGBoost",
                        n_features=total_dim,
                        n_classes=n_classes,
                        seed=42,
                        device="cpu",
                    )
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = balanced_accuracy_score(y_test, y_pred) * 100
                    fold_accs.append(acc)
                    print(f"    Fold {fold_i+1}: {acc:.1f}% (XGBoost fallback)")
                except Exception as e2:
                    print(f"    Fold {fold_i+1}: FAILED (XGBoost fallback also failed: {e2})")
            else:
                print(f"    Fold {fold_i+1}: FAILED ({e})")

    if not fold_accs:
        return 0.0, 0.0, [], clf_name

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"  Mean accuracy: {mean_acc:.1f}% (+/- {std_acc:.1f}%)")
    return round(mean_acc, 2), round(std_acc, 2), [round(a, 2) for a in fold_accs], clf_name


# -- Main ------------------------------------------------------------------

def run_ablation(condition: str, dataset_name: str,
                 n_samples: int = 5, n_eval: int = 500,
                 time_limit: float = 120.0, verbose: bool = False,
                 skip_eval: bool = False):
    """Run full ablation for one condition + dataset.

    Returns:
        Result dict saved as JSON.
    """
    cond_info = CONDITION_MAP[condition]
    info = DATASET_INFO[dataset_name]
    object_type = info["object_type"]
    color_mode = info["color_mode"]

    print("=" * 80)
    print(f"  ABLATION: {cond_info['label']} | {dataset_name}")
    print(f"  object_type={object_type}, color_mode={color_mode}")
    print(f"  n_samples={n_samples}, n_eval={n_eval}")
    print("=" * 80)

    # Phase 1: Demo — get descriptor choices
    print(f"\n--- DEMO PHASE ({n_samples} images) ---")
    t0 = time.time()
    per_image_details = run_demo_phase(
        dataset_name, condition, n_samples=n_samples,
        time_limit=time_limit, verbose=verbose,
    )
    demo_time = time.time() - t0

    # Determine majority-vote descriptor
    descriptor_choices = [r["descriptor"] for r in per_image_details
                          if r.get("descriptor") not in ("ERROR", "unknown")]
    if not descriptor_choices:
        print("  ERROR: No valid descriptor choices from demo phase")
        descriptor_choices = ["persistence_statistics"]  # fallback

    counter = Counter(descriptor_choices)
    majority_descriptor = counter.most_common(1)[0][0]
    unique_descriptors = len(set(descriptor_choices))

    print(f"\n  Descriptor choices: {descriptor_choices}")
    print(f"  Majority descriptor: {majority_descriptor} "
          f"({counter[majority_descriptor]}/{len(descriptor_choices)} votes)")

    # Phase 2: Classifier evaluation
    balanced_accuracy = 0.0
    std_accuracy = 0.0
    fold_accuracies = []
    classifier_name = "N/A"

    if not skip_eval:
        print(f"\n--- EVAL PHASE ({n_eval} images, descriptor={majority_descriptor}) ---")
        t1 = time.time()
        balanced_accuracy, std_accuracy, fold_accuracies, classifier_name = run_eval_phase(
            dataset_name, majority_descriptor,
            object_type, color_mode, n_eval=n_eval,
        )
        eval_time = time.time() - t1
    else:
        eval_time = 0.0

    # Build result
    result = {
        "condition": cond_info["label"],
        "condition_id": condition,
        "dataset": dataset_name,
        "object_type": object_type,
        "color_mode": color_mode,
        "n_demo_images": n_samples,
        "descriptor_choices": descriptor_choices,
        "majority_descriptor": majority_descriptor,
        "unique_descriptors": unique_descriptors,
        "n_eval_images": n_eval if not skip_eval else 0,
        "classifier": classifier_name,
        "balanced_accuracy": balanced_accuracy,
        "std_accuracy": std_accuracy,
        "fold_accuracies": fold_accuracies,
        "per_image_details": per_image_details,
        "demo_time_seconds": round(demo_time, 2),
        "eval_time_seconds": round(eval_time, 2),
        "total_time_seconds": round(demo_time + eval_time, 2),
    }

    # Save result
    outdir = PROJECT_ROOT / "results" / "ablation_study" / cond_info["label"]
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{dataset_name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved to: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="TopoAgent v8.1 Ablation Study")
    parser.add_argument("--condition", type=str, required=True,
                        choices=list(CONDITION_MAP.keys()) + ["all"],
                        help="Ablation condition (c0-c4 or 'all')")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_INFO.keys()),
                        help="Dataset name")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of demo images for descriptor selection (default: 5)")
    parser.add_argument("--n-eval", type=int, default=500,
                        help="Number of images for classifier evaluation (default: 500)")
    parser.add_argument("--time-limit", type=float, default=120.0,
                        help="Time limit per image in seconds (default: 120)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full LLM prompts/responses")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip classifier evaluation (demo phase only)")
    args = parser.parse_args()

    conditions = list(CONDITION_MAP.keys()) if args.condition == "all" else [args.condition]

    summaries = []
    for cond in conditions:
        try:
            result = run_ablation(
                condition=cond,
                dataset_name=args.dataset,
                n_samples=args.n_samples,
                n_eval=args.n_eval,
                time_limit=args.time_limit,
                verbose=args.verbose,
                skip_eval=args.no_eval,
            )
            summaries.append(result)
        except Exception as e:
            print(f"\n  ERROR on {cond}: {e}")
            import traceback
            traceback.print_exc()
            summaries.append({
                "condition": CONDITION_MAP[cond]["label"],
                "dataset": args.dataset,
                "error": str(e),
            })

    # Print comparison table
    if len(summaries) > 1:
        print("\n" + "=" * 90)
        print(f"  ABLATION COMPARISON — {args.dataset}")
        print("=" * 90)
        header = f"  {'Condition':<20} {'Descriptor':<25} {'Unique':>6} {'Accuracy':>10} {'Time':>8}"
        print(header)
        print("  " + "-" * 88)
        for s in summaries:
            if "error" in s:
                print(f"  {s['condition']:<20} ERROR: {s['error'][:50]}")
            else:
                print(f"  {s['condition']:<20} {s['majority_descriptor']:<25} "
                      f"{s['unique_descriptors']:>6} "
                      f"{s['balanced_accuracy']:>9.1f}% "
                      f"{s['total_time_seconds']:>7.0f}s")


if __name__ == "__main__":
    main()
