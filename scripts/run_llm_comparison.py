#!/usr/bin/env python3
"""LLM Comparison: TopoAgent with different LLM backends vs fixed baselines.

Evaluates 6 methods across 26 medical image datasets:
  1. fixed_pi:     Always use persistence_image (no LLM)
  2. fixed_ps:     Always use persistence_statistics (no LLM)
  3. gpt4o:        TopoAgent v8 with gpt-4o
  4. claude:       TopoAgent v8 with claude-sonnet-4-6
  5. gemini_flash: TopoAgent v8 with gemini-2.5-flash
  6. gemini_pro:   TopoAgent v8 with gemini-2.5-pro

Each evaluation: n_eval images (default 200), 3-fold stratified CV with XGBoost,
balanced accuracy. Results grouped by object type.

Usage:
    # Test on one dataset with all methods
    python scripts/run_llm_comparison.py --datasets BloodMNIST --n-eval 50

    # Run all 26 datasets with all 4 methods
    python scripts/run_llm_comparison.py --n-eval 200

    # Only fixed baselines (no LLM API keys needed)
    python scripts/run_llm_comparison.py --methods fixed_pi,fixed_ps

    # Only agent methods on specific datasets
    python scripts/run_llm_comparison.py --methods claude,gemini --datasets BloodMNIST,PathMNIST

    # Single dataset for SGE parallelization
    python scripts/run_llm_comparison.py --datasets BloodMNIST --output results/llm_comparison/BloodMNIST.json
"""

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from collections import defaultdict

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
from topoagent.skills import SkillRegistry

# -- Constants -------------------------------------------------------------

DATASET_INFO = {
    name: {
        "object_type": DATASET_TO_OBJECT_TYPE[name],
        "color": DATASET_COLOR_MODE.get(name, "grayscale") == "per_channel",
        "color_mode_str": DATASET_COLOR_MODE.get(name, "grayscale"),
    }
    for name in DATASET_TO_OBJECT_TYPE
}

IMAGE_BASED_DESCRIPTORS = {
    "minkowski_functionals", "euler_characteristic_curve",
    "euler_characteristic_transform", "edge_histogram", "lbp_texture",
}

# Method definitions: type, descriptor/model, display label
METHODS = {
    "fixed_pi": {
        "type": "fixed",
        "descriptor": "persistence_image",
        "label": "Fixed-PI",
    },
    "fixed_ps": {
        "type": "fixed",
        "descriptor": "persistence_statistics",
        "label": "Fixed-PS",
    },
    "gpt4o": {
        "type": "agent",
        "model": "gpt-4o",
        "label": "GPT-4o",
    },
    "claude": {
        "type": "agent",
        "model": "claude-sonnet-4-6",
        "label": "Claude",
    },
    "gemini_flash": {
        "type": "agent",
        "model": "gemini-2.5-flash",
        "label": "Gemini-Flash",
    },
    "gemini_pro": {
        "type": "agent",
        "model": "gemini-2.5-pro",
        "label": "Gemini-Pro",
    },
}


# -- Dataset loading -------------------------------------------------------

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


# -- Tool initialization ---------------------------------------------------

_TOOLS_CACHE = None


def _init_tools():
    """Initialize feature extraction tools (no LLM needed). Cached."""
    global _TOOLS_CACHE
    if _TOOLS_CACHE is not None:
        return _TOOLS_CACHE

    from topoagent.tools.preprocessing import ImageLoaderTool
    from topoagent.tools.homology import ComputePHTool
    from topoagent.tools.descriptors import get_all_descriptors

    tools = {
        "image_loader": ImageLoaderTool(),
        "compute_ph": ComputePHTool(),
    }
    descriptor_tools = get_all_descriptors()
    for desc_name in SUPPORTED_DESCRIPTORS:
        if desc_name in descriptor_tools:
            tools[desc_name] = descriptor_tools[desc_name]

    _TOOLS_CACHE = tools
    return tools


# -- Dataset description loading -------------------------------------------

_DATASET_DESCRIPTIONS = None


def _get_dataset_descriptions():
    """Lazily load dataset descriptions for agent queries."""
    global _DATASET_DESCRIPTIONS
    if _DATASET_DESCRIPTIONS is not None:
        return _DATASET_DESCRIPTIONS
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


def _build_agent_query(dataset_name: str, n_channels: int = 3) -> str:
    """Build a generic query for the agent (same as demo_topoagent_e2e.py)."""
    descriptions = _get_dataset_descriptions()
    desc_text = descriptions.get(dataset_name, "")

    if desc_text and isinstance(desc_text, dict):
        parts = []
        if "domain" in desc_text:
            parts.append(f"Domain: {desc_text['domain']}")
        if "description" in desc_text:
            parts.append(desc_text["description"])
        if "what_matters" in desc_text:
            parts.append(f"Key features: {desc_text['what_matters']}")
        if "n_classes" in desc_text:
            parts.append(f"{desc_text['n_classes']} classes")
        context = "; ".join(parts)
    elif desc_text and isinstance(desc_text, str):
        context = desc_text
    else:
        context = ""

    context_line = f"\n\nDataset context: {context}" if context else ""
    return (
        f"Analyze the provided medical image, compute its persistent homology, "
        f"and determine the most suitable topology descriptor to produce a "
        f"fixed-length feature vector."
        f"{context_line}"
    )


# -- Feature extraction ----------------------------------------------------

def extract_features(images, labels, descriptor_name, tools, params):
    """Extract features for all images with a given descriptor.

    Args:
        images: numpy array of images
        labels: numpy array of labels
        descriptor_name: name of descriptor tool
        tools: dict of tool_name -> tool instance
        params: optimal params from SkillRegistry

    Returns:
        (features_list, valid_labels) — lists of numpy arrays and int labels
    """
    features_list = []
    valid_labels = []

    for i, (img, lbl) in enumerate(zip(images, labels)):
        if i > 0 and i % 100 == 0:
            print(f"    ... {i}/{len(images)}")

        # Handle float32 images
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

            # Build tool args from params (exclude meta keys)
            tool_params = {
                k: v for k, v in params.items()
                if k not in ("total_dim", "classifier", "color_mode", "dim")
            }

            if descriptor_name in IMAGE_BASED_DESCRIPTORS:
                tool_args = {"image_array": image_array, **tool_params}
                result = tools[descriptor_name].invoke(tool_args)
            else:
                ph_result = tools["compute_ph"].invoke({
                    "image_array": image_array,
                    "filtration_type": "sublevel",
                    "max_dimension": 1,
                })
                if not ph_result.get("success"):
                    continue

                tool_args = {
                    "persistence_data": ph_result["persistence"],
                    **tool_params,
                }
                result = tools[descriptor_name].invoke(tool_args)

            if result.get("success"):
                fv = result.get("combined_vector") or result.get("feature_vector")
                if fv is not None:
                    features_list.append(np.asarray(fv, dtype=np.float64))
                    valid_labels.append(lbl)
        except Exception:
            pass
        finally:
            os.unlink(tmp_path)

    return features_list, valid_labels


# -- XGBoost CV evaluation -------------------------------------------------

def run_xgboost_cv(features_list, labels, n_splits=3):
    """Run stratified k-fold CV with XGBoost.

    Returns:
        mean balanced accuracy (%) or 0.0 if too few samples
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import balanced_accuracy_score
    from xgboost import XGBClassifier

    if len(features_list) < 10:
        return 0.0

    X = np.array(features_list)
    y = np.array(labels)

    # Clean features
    X = np.clip(X, -1e10, 1e10)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
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

    return round(float(np.mean(fold_accs)), 2)


# -- Fixed baseline evaluation ---------------------------------------------

def run_fixed_baseline(dataset_name, descriptor_name, n_eval=200):
    """Run a fixed-descriptor baseline (no LLM needed).

    Returns:
        dict with accuracy, descriptor, feature_dim, time
    """
    info = DATASET_INFO[dataset_name]
    object_type = info["object_type"]
    color_mode = info["color_mode_str"]

    registry = SkillRegistry()
    params = registry.configure_after_selection(descriptor_name, object_type, color_mode)

    print(f"  [{dataset_name}] Fixed: {descriptor_name} (ot={object_type})")

    t0 = time.time()

    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=n_eval)
    print(f"    Loaded {len(images)} images")

    tools = _init_tools()
    features_list, valid_labels = extract_features(
        images, labels, descriptor_name, tools, params)

    extract_time = time.time() - t0

    if len(features_list) < 10:
        print(f"    Only {len(features_list)} valid features -- skipping")
        return {
            "accuracy": 0.0, "descriptor": descriptor_name,
            "n_valid": len(features_list), "feature_dim": 0,
            "time": round(time.time() - t0, 1), "error": "too_few_features",
        }

    dim = features_list[0].shape[0]
    print(f"    {len(features_list)} features ({dim}D) in {extract_time:.1f}s")

    accuracy = run_xgboost_cv(features_list, valid_labels)
    total_time = time.time() - t0

    print(f"    Accuracy: {accuracy:.1f}% ({total_time:.1f}s)")

    return {
        "accuracy": accuracy,
        "descriptor": descriptor_name,
        "n_valid": len(features_list),
        "feature_dim": dim,
        "time": round(total_time, 1),
    }


# -- Agent-based evaluation ------------------------------------------------

def run_agent_eval(dataset_name, model_name, n_eval=200, time_limit=90.0):
    """Run TopoAgent v8 with a specific LLM to select descriptor, then evaluate.

    Steps:
      1. Run agent on one sample image -> get descriptor selection
      2. Extract features for all n_eval images with that descriptor
      3. Run 3-fold XGBoost CV

    Returns:
        dict with accuracy, descriptor, feature_dim, agent_time, time
    """
    from topoagent.agent import create_topoagent_auto

    info = DATASET_INFO[dataset_name]
    object_type = info["object_type"]
    color_mode = info["color_mode_str"]

    print(f"  [{dataset_name}] Agent: {model_name} (ot={object_type})")

    t0 = time.time()

    # Load dataset
    images, labels, class_names, n_ch = load_dataset_unified(
        dataset_name, n_samples=n_eval)
    print(f"    Loaded {len(images)} images")

    # Step 1: Run agent on first image to get descriptor selection
    img0 = images[0]
    if img0.dtype != np.uint8:
        img0_save = (img0 * 255).clip(0, 255).astype(np.uint8)
    else:
        img0_save = img0

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.fromarray(img0_save).save(f.name)
        img_path = f.name

    query = _build_agent_query(dataset_name, n_channels=n_ch)

    agent_t0 = time.time()
    descriptor_name = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            agent = create_topoagent_auto(
                model_name=model_name,
                agentic_v8=True,
                time_limit_seconds=time_limit,
            )
            agent.workflow.verbose = False
            result = agent.classify(image_path=img_path, query=query)
            agent_time = time.time() - agent_t0

            # Extract descriptor from result
            report = result.get("report") or result.get("raw_answer") or {}
            descriptor_name = (
                report.get("descriptor", None)
                if isinstance(report, dict) else None
            )

            # Validate
            if descriptor_name not in set(SUPPORTED_DESCRIPTORS):
                print(f"    WARNING: Agent chose '{descriptor_name}', "
                      f"falling back to persistence_statistics")
                descriptor_name = "persistence_statistics"

            print(f"    Agent chose: {descriptor_name} ({agent_time:.1f}s)")
            break  # success

        except Exception as e:
            agent_time = time.time() - agent_t0
            if attempt < max_retries - 1 and ("overloaded" in str(e).lower()
                                               or "529" in str(e)
                                               or "rate" in str(e).lower()
                                               or "429" in str(e)):
                wait = 30 * (attempt + 1)
                print(f"    Agent attempt {attempt+1} failed ({agent_time:.1f}s): {e}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
                agent_t0 = time.time()
            else:
                print(f"    Agent FAILED ({agent_time:.1f}s): {e}")
                descriptor_name = "persistence_statistics"
                print(f"    Falling back to: {descriptor_name}")
                break

    # Clean up temp image
    try:
        os.unlink(img_path)
    except OSError:
        pass

    # Step 2: Extract features with agent-chosen descriptor
    registry = SkillRegistry()
    params = registry.configure_after_selection(
        descriptor_name, object_type, color_mode)

    tools = _init_tools()
    features_list, valid_labels = extract_features(
        images, labels, descriptor_name, tools, params)

    if len(features_list) < 10:
        print(f"    Only {len(features_list)} valid features -- skipping")
        return {
            "accuracy": 0.0, "descriptor": descriptor_name,
            "n_valid": len(features_list), "feature_dim": 0,
            "agent_time": round(agent_time, 1),
            "time": round(time.time() - t0, 1), "error": "too_few_features",
        }

    dim = features_list[0].shape[0]
    extract_time = time.time() - t0 - agent_time
    print(f"    {len(features_list)} features ({dim}D) in {extract_time:.1f}s")

    # Step 3: Run CV
    accuracy = run_xgboost_cv(features_list, valid_labels)
    total_time = time.time() - t0

    print(f"    Accuracy: {accuracy:.1f}% ({total_time:.1f}s, agent={agent_time:.1f}s)")

    return {
        "accuracy": accuracy,
        "descriptor": descriptor_name,
        "n_valid": len(features_list),
        "feature_dim": dim,
        "agent_time": round(agent_time, 1),
        "time": round(total_time, 1),
    }


# -- Orchestration ---------------------------------------------------------

def run_all(method_names, datasets, n_eval=200, time_limit=90.0,
            output_path=None):
    """Run all method x dataset combinations and collect results."""
    results = {}

    n_total = len(datasets) * len(method_names)
    print(f"\n{'='*90}")
    print(f"  LLM Comparison: {len(datasets)} datasets x {len(method_names)} methods = {n_total} evaluations")
    print(f"  Methods: {method_names}")
    print(f"  n_eval={n_eval}, time_limit={time_limit}s")
    print(f"{'='*90}")

    done = 0
    for ds_name in datasets:
        results[ds_name] = {}
        for method_name in method_names:
            method = METHODS[method_name]
            done += 1
            print(f"\n[{done}/{n_total}] {ds_name} / {method_name}")

            try:
                if method["type"] == "fixed":
                    r = run_fixed_baseline(
                        ds_name, method["descriptor"], n_eval)
                elif method["type"] == "agent":
                    r = run_agent_eval(
                        ds_name, method["model"], n_eval, time_limit)
                else:
                    raise ValueError(f"Unknown method type: {method['type']}")
            except Exception as e:
                print(f"    ERROR: {e}")
                r = {"accuracy": 0.0, "error": str(e), "time": 0}

            results[ds_name][method_name] = r

            # Save intermediate results after each evaluation
            if output_path:
                _save_results(results, method_names, output_path)

    return results


# -- Results aggregation and display ---------------------------------------

def _aggregate_by_object_type(results, method_names):
    """Aggregate per-dataset results by object type."""
    by_type = defaultdict(lambda: {"datasets": [], "methods": defaultdict(list)})

    for ds_name, method_results in results.items():
        ot = DATASET_TO_OBJECT_TYPE.get(ds_name, "unknown")
        by_type[ot]["datasets"].append(ds_name)
        for method_name in method_names:
            r = method_results.get(method_name, {})
            acc = r.get("accuracy", 0.0)
            if acc > 0:
                by_type[ot]["methods"][method_name].append(acc)

    agg = {}
    for ot, data in by_type.items():
        agg[ot] = {"n_datasets": len(data["datasets"])}
        for method_name in method_names:
            accs = data["methods"].get(method_name, [])
            agg[ot][method_name] = round(float(np.mean(accs)), 2) if accs else 0.0
    return agg


def print_results_table(results, method_names):
    """Print formatted results table."""
    # Build column labels
    labels = {m: METHODS[m]["label"] for m in method_names}
    col_width = max(10, max(len(v) for v in labels.values()) + 1)

    # Per-dataset table
    print(f"\n{'='*120}")
    print(f"  RESULTS: Per-Dataset Balanced Accuracy (%)")
    print(f"{'='*120}")

    header = f"  {'Dataset':<22} {'ObjType':<16}"
    for m in method_names:
        header += f" {labels[m]:>{col_width}}"
    header += f"  {'Agent Descriptor':>22}"
    print(header)
    print("  " + "-" * (22 + 16 + col_width * len(method_names) + 24))

    for ds_name in sorted(results.keys()):
        ot = DATASET_TO_OBJECT_TYPE.get(ds_name, "?")
        row = f"  {ds_name:<22} {ot:<16}"

        agent_descs = []
        for m in method_names:
            r = results[ds_name].get(m, {})
            acc = r.get("accuracy", 0.0)
            if r.get("error"):
                row += f" {'ERR':>{col_width}}"
            else:
                row += f" {acc:>{col_width - 1}.1f}%"

            if METHODS[m]["type"] == "agent" and r.get("descriptor"):
                agent_descs.append(f"{labels[m]}={r['descriptor']}")

        desc_str = ", ".join(agent_descs) if agent_descs else ""
        row += f"  {desc_str}"
        print(row)

    # Per-object-type aggregation
    agg = _aggregate_by_object_type(results, method_names)

    print(f"\n{'='*100}")
    print(f"  RESULTS: Per Object Type (Mean Balanced Accuracy %)")
    print(f"{'='*100}")

    header2 = f"  {'Object Type':<22} {'#DS':>4}"
    for m in method_names:
        header2 += f" {labels[m]:>{col_width}}"
    print(header2)
    print("  " + "-" * (22 + 4 + col_width * len(method_names) + 2))

    overall = defaultdict(list)
    for ot in sorted(agg.keys()):
        n = agg[ot].get("n_datasets", 0)
        row = f"  {ot:<22} {n:>4}"
        for m in method_names:
            acc = agg[ot].get(m, 0.0)
            row += f" {acc:>{col_width - 1}.1f}%"
            if acc > 0:
                overall[m].append(acc)
        print(row)

    # Overall mean
    print("  " + "-" * (22 + 4 + col_width * len(method_names) + 2))
    row = f"  {'OVERALL MBA':<22} {'':>4}"
    for m in method_names:
        accs = overall.get(m, [])
        mean_acc = float(np.mean(accs)) if accs else 0.0
        row += f" {mean_acc:>{col_width - 1}.1f}%"
    print(row)
    print()


def _save_results(results, method_names, output_path):
    """Save results dict to JSON."""
    agg = _aggregate_by_object_type(results, method_names)

    # Compute overall means
    overall = {}
    for m in method_names:
        accs = []
        for ds_results in results.values():
            r = ds_results.get(m, {})
            acc = r.get("accuracy", 0.0)
            if acc > 0:
                accs.append(acc)
        overall[m] = round(float(np.mean(accs)), 2) if accs else 0.0

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_datasets": len(results),
            "methods": method_names,
            "labels": {m: METHODS[m]["label"] for m in method_names},
        },
        "per_dataset": results,
        "per_object_type": agg,
        "overall": overall,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


# -- CLI entry point -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Comparison: TopoAgent with different backends vs fixed baselines")
    parser.add_argument(
        "--methods", type=str, default=None,
        help="Comma-separated methods (default: all 6). "
             "Options: fixed_pi, fixed_ps, gpt4o, claude, gemini_flash, gemini_pro")
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="Comma-separated dataset names or 'all' (default: all)")
    parser.add_argument(
        "--n-eval", type=int, default=200,
        help="Number of samples per dataset for CV (default: 200)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON output path (default: results/llm_comparison/results.json)")
    parser.add_argument(
        "--time-limit", type=float, default=90.0,
        help="Time limit in seconds for agent pipeline (default: 90)")
    args = parser.parse_args()

    # Parse methods
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(",")]
        for m in method_names:
            if m not in METHODS:
                print(f"ERROR: Unknown method '{m}'")
                print(f"Available: {list(METHODS.keys())}")
                sys.exit(1)
    else:
        method_names = list(METHODS.keys())

    # Parse datasets
    if args.datasets == "all":
        datasets = list(DATASET_TO_OBJECT_TYPE.keys())
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]
        for d in datasets:
            if d not in DATASET_TO_OBJECT_TYPE:
                print(f"ERROR: Unknown dataset '{d}'")
                print(f"Available: {list(DATASET_TO_OBJECT_TYPE.keys())}")
                sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        outdir = PROJECT_ROOT / "results" / "llm_comparison"
        outdir.mkdir(parents=True, exist_ok=True)
        output_path = str(outdir / "results.json")

    # Run all evaluations
    results = run_all(
        method_names, datasets,
        n_eval=args.n_eval,
        time_limit=args.time_limit,
        output_path=output_path,
    )

    # Print formatted results
    print_results_table(results, method_names)

    # Save final results
    _save_results(results, method_names, output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
