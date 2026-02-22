#!/usr/bin/env python
"""
Consolidate all Exp4 raw results into a single JSON file.

Reads all exp4_fine_*, exp4_param_*, exp4_sigma_*, exp4_boundary_*, exp4_color_*
files and combines them into one unified structure:

{
  "metadata": {...},
  "results": {
    "BloodMNIST": {
      "persistence_image": {
        "dimension_search": [
          {"params": {...}, "dim": 200, "accuracy": 0.42, "std": 0.02, "time": 10.5}
        ],
        "parameter_search": [...],
        "sigma_extension_v1": [...],
        "sigma_extension_v2": [...],
        "boundary_extension": [...],
        "color_experiment": {"grayscale": {...}, "per_channel": {...}}
      },
      ...
    },
    ...
  }
}

Usage:
    python benchmarks/benchmark3/consolidate_exp4_raw.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results/benchmark3/exp4")
DATASETS = ["BloodMNIST", "PathMNIST", "RetinaMNIST", "DermaMNIST", "OrganAMNIST"]


def deep_dict():
    """Nested defaultdict for results[dataset][descriptor]."""
    return defaultdict(lambda: defaultdict(list))


def load_json(path):
    """Load JSON file, return None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def process_fine_search_multi(results, filepath):
    """Process exp4_fine_{descriptor}_n2000.json (multi-dataset, dict-keyed all_results)."""
    data = load_json(filepath)
    if not data:
        return
    for dataset, entry in data.items():
        if dataset not in DATASETS:
            continue
        descriptor = entry["descriptor"]
        control_param = entry["control_param"]
        fixed_params = entry.get("fixed_params", {})
        for _step, r in entry.get("all_results", {}).items():
            params = dict(fixed_params)
            params[control_param] = r["value"]
            results[dataset][descriptor].append({
                "experiment": "dimension_search",
                "params": params,
                "dim": r["dim"],
                "accuracy": r["accuracy"],
                "std": r.get("std"),
                "time": r.get("time"),
            })


def process_fine_search_single(results, filepath):
    """Process exp4_fine_{descriptor}_{dataset}_n2000.json (single-dataset)."""
    # Same format as multi, just with one dataset key
    process_fine_search_multi(results, filepath)


def process_parameter_search(results, filepath, experiment_name="parameter_search"):
    """Process exp4_parameter_search_n2000.json or exp4_param_*_n2000.json."""
    data = load_json(filepath)
    if not data:
        return
    for dataset, descriptors in data.get("results", {}).items():
        if dataset not in DATASETS:
            continue
        for descriptor, entry in descriptors.items():
            dim_param = entry.get("dimension_param", "")
            dim_value = entry.get("dimension_value")
            dim_size = entry.get("dimension_size")
            classifier = entry.get("classifier", "TabPFN")
            for r in entry.get("all_results", []):
                params = dict(r.get("params", {}))
                # Add dimension param if not in params
                if dim_param and dim_value and dim_param not in params:
                    params[dim_param] = dim_value
                results[dataset][descriptor].append({
                    "experiment": experiment_name,
                    "params": params,
                    "dim": dim_size,
                    "accuracy": r["accuracy"],
                    "std": r.get("std"),
                    "time": r.get("time"),
                    "classifier": classifier,
                })


def process_sigma_extension(results, filepath, version="v1"):
    """Process exp4_sigma_extension_n2000.json or exp4_sigma_extension_v2_*_n2000.json."""
    data = load_json(filepath)
    if not data:
        return
    experiment_name = f"sigma_extension_{version}"
    for dataset, entry in data.get("results", {}).items():
        if dataset not in DATASETS:
            continue
        config = entry.get("config", {})
        resolution = config.get("resolution")
        dim_size = config.get("dim_size")
        for r in entry.get("extension_results", []):
            params = {
                "sigma": r["sigma"],
                "weight_function": r["weight_function"],
            }
            if resolution:
                params["resolution"] = resolution
            results[dataset]["persistence_image"].append({
                "experiment": experiment_name,
                "params": params,
                "dim": dim_size,
                "accuracy": r.get("accuracy", 0),
                "std": r.get("std"),
                "time": r.get("time"),
                "error": r.get("error"),
            })


def process_boundary_extension(results, filepath):
    """Process exp4_boundary_extension_{dataset}_{descriptor}_n2000.json."""
    data = load_json(filepath)
    if not data:
        return
    dataset = data.get("dataset")
    if dataset not in DATASETS:
        return
    for descriptor, entry in data.get("results", {}).items():
        config = entry.get("config", {})
        control_param = config.get("control_param", "")
        fixed_params = config.get("fixed_params", {})
        classifier = entry.get("classifier", "TabPFN")
        for r in entry.get("extension_results", []):
            params = dict(fixed_params)
            params[control_param] = r["value"]
            results[dataset][descriptor].append({
                "experiment": "boundary_extension",
                "params": params,
                "dim": r["dim"],
                "accuracy": r.get("accuracy", 0),
                "std": r.get("std"),
                "time": r.get("time"),
                "classifier": classifier,
                "oom": r.get("oom"),
                "error": r.get("error"),
            })


def process_color_experiment(results, filepath):
    """Process exp4_color_experiment_consolidated_n2000.json.

    Structure: {obj_type: {dataset: str, descriptors: {desc: {grayscale: {accuracy, std}, per_channel: {...}}}}}
    """
    data = load_json(filepath)
    if not data:
        return
    for obj_type, obj_data in data.items():
        if not isinstance(obj_data, dict) or "dataset" not in obj_data:
            continue
        dataset = obj_data["dataset"]
        if dataset not in DATASETS:
            continue
        descriptors = obj_data.get("descriptors", {})
        if not isinstance(descriptors, dict):
            continue
        for descriptor, desc_data in descriptors.items():
            if not isinstance(desc_data, dict):
                continue
            color_entry = {}
            for mode in ("grayscale", "per_channel"):
                if mode in desc_data:
                    color_entry[mode] = {
                        "accuracy": desc_data[mode].get("accuracy"),
                        "std": desc_data[mode].get("std"),
                    }
            if color_entry:
                results[dataset][descriptor].append({
                    "experiment": "color_experiment",
                    "color_results": color_entry,
                })


def process_color_experiment_raw(results, filepath):
    """Process exp4_color_experiment_n2000.json (raw per-object-type format)."""
    data = load_json(filepath)
    if not data:
        return
    for obj_type, obj_data in data.items():
        if not isinstance(obj_data, dict):
            continue
        obj_results = obj_data.get("results", {})
        for dataset, ds_data in obj_results.items():
            if dataset not in DATASETS:
                continue
            descriptors = ds_data.get("descriptors", {})
            for descriptor, desc_data in descriptors.items():
                color_entry = {}
                for mode in ("grayscale", "per_channel"):
                    if mode in desc_data:
                        color_entry[mode] = {
                            "accuracy": desc_data[mode].get("accuracy"),
                            "std": desc_data[mode].get("std"),
                            "dim": desc_data[mode].get("dim"),
                            "params": desc_data[mode].get("params", {}),
                            "time": desc_data[mode].get("time"),
                        }
                if color_entry:
                    # Check if we already have a color entry from consolidated
                    existing = [e for e in results[dataset][descriptor]
                                if e.get("experiment") == "color_experiment"]
                    if not existing:
                        results[dataset][descriptor].append({
                            "experiment": "color_experiment",
                            "color_results": color_entry,
                        })


def clean_results(results):
    """Remove None values and convert defaultdict to regular dict."""
    cleaned = {}
    for dataset in DATASETS:
        if dataset not in results:
            continue
        cleaned[dataset] = {}
        for descriptor in sorted(results[dataset].keys()):
            entries = results[dataset][descriptor]
            clean_entries = []
            for e in entries:
                clean_e = {k: v for k, v in e.items() if v is not None}
                clean_entries.append(clean_e)
            cleaned[dataset][descriptor] = clean_entries
    return cleaned


def count_entries(results):
    """Count total entries."""
    total = 0
    for dataset in results:
        for descriptor in results[dataset]:
            total += len(results[dataset][descriptor])
    return total


def main():
    results = deep_dict()

    print("Loading dimension search (fine) files...")

    # Multi-dataset fine search files
    for f in sorted(RESULTS_DIR.glob("exp4_fine_*_n2000.json")):
        if "OrganAMNIST" in f.name:
            continue  # Handle separately
        print(f"  {f.name}")
        process_fine_search_multi(results, f)

    # OrganAMNIST-specific fine search files
    for f in sorted(RESULTS_DIR.glob("exp4_fine_*_OrganAMNIST_n2000.json")):
        print(f"  {f.name}")
        process_fine_search_single(results, f)

    print(f"  -> {count_entries(results)} entries so far")

    print("\nLoading parameter search files...")

    # Main parameter search
    f = RESULTS_DIR / "exp4_parameter_search_n2000.json"
    if f.exists():
        print(f"  {f.name}")
        process_parameter_search(results, f, "parameter_search")

    # Additional parameter search files
    for f in sorted(RESULTS_DIR.glob("exp4_param_*_n2000.json")):
        print(f"  {f.name}")
        process_parameter_search(results, f, f"parameter_search_{f.stem.split('param_')[1].split('_n2000')[0]}")

    print(f"  -> {count_entries(results)} entries so far")

    print("\nLoading sigma extension files...")

    # Sigma extension v1
    f = RESULTS_DIR / "exp4_sigma_extension_n2000.json"
    if f.exists():
        print(f"  {f.name}")
        process_sigma_extension(results, f, "v1")

    # Sigma extension v2
    for f in sorted(RESULTS_DIR.glob("exp4_sigma_extension_v2_*_n2000.json")):
        print(f"  {f.name}")
        process_sigma_extension(results, f, "v2")

    print(f"  -> {count_entries(results)} entries so far")

    print("\nLoading boundary extension files...")
    for f in sorted(RESULTS_DIR.glob("exp4_boundary_extension_*_n2000.json")):
        print(f"  {f.name}")
        process_boundary_extension(results, f)

    print(f"  -> {count_entries(results)} entries so far")

    print("\nLoading color experiment files...")

    # Consolidated color (preferred, has all 5 datasets)
    f = RESULTS_DIR / "exp4_color_experiment_consolidated_n2000.json"
    if f.exists():
        print(f"  {f.name}")
        process_color_experiment(results, f)

    # Raw color (fallback for any datasets not in consolidated)
    f = RESULTS_DIR / "exp4_color_experiment_n2000.json"
    if f.exists():
        print(f"  {f.name}")
        process_color_experiment_raw(results, f)

    print(f"  -> {count_entries(results)} entries so far")

    # Fill gaps from exp4_optimal_dimensions.json
    print("\nFilling gaps from optimal dimensions...")
    opt_dims = load_json(RESULTS_DIR / "exp4_optimal_dimensions.json")
    if opt_dims:
        filled = 0
        for dataset in DATASETS:
            ds_opt = opt_dims.get(dataset, {})
            for descriptor, opt in ds_opt.items():
                if descriptor == "metadata":
                    continue
                if not results[dataset][descriptor]:
                    # No raw data at all — add the optimal point
                    results[dataset][descriptor].append({
                        "experiment": "optimal_dimension_only",
                        "params": {"best_value": opt.get("best_value")},
                        "dim": opt.get("best_dim"),
                        "accuracy": opt.get("accuracy"),
                        "note": "Only optimal value available; full search curve not saved as JSON",
                    })
                    filled += 1
        print(f"  Filled {filled} descriptor×dataset gaps with optimal-only entries")

    # Clean and output
    cleaned = clean_results(results)

    # Stats
    print(f"\n{'='*60}")
    print("  CONSOLIDATION SUMMARY")
    print(f"{'='*60}")
    total = 0
    for dataset in DATASETS:
        ds_data = cleaned.get(dataset, {})
        n_descriptors = len(ds_data)
        n_entries = sum(len(v) for v in ds_data.values())
        total += n_entries
        print(f"  {dataset}: {n_descriptors} descriptors, {n_entries} entries")
        for desc in sorted(ds_data.keys()):
            experiments = set()
            for e in ds_data[desc]:
                experiments.add(e.get("experiment", "unknown"))
            print(f"    {desc}: {len(ds_data[desc])} entries [{', '.join(sorted(experiments))}]")
    print(f"\n  TOTAL: {total} entries across {len(cleaned)} datasets")

    output = {
        "metadata": {
            "description": "Consolidated raw accuracy results from all Exp4 experiments",
            "generated": "2026-02-05",
            "n_samples": 2000,
            "n_folds": 5,
            "classifier_default": "TabPFN (XGBoost for >2000D)",
            "experiment_types": [
                "dimension_search: accuracy at different D values (golden section search)",
                "parameter_search: accuracy at different param values (grid search)",
                "sigma_extension_v1: PI sigma 0.3-0.4 extension",
                "sigma_extension_v2: PI sigma 0.45-0.6 extension",
                "boundary_extension: extended search range for boundary descriptors",
                "color_experiment: grayscale vs per_channel comparison",
            ],
            "entry_format": "Each entry has: experiment, params, dim, accuracy, std, time",
        },
        "results": cleaned,
    }

    output_path = RESULTS_DIR / "exp4_all_raw_results_consolidated.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
