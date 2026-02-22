#!/usr/bin/env python
"""
Rebuild exp4_parameter_search_n2000.json with all 10 descriptors (including ECT).

The original file was overwritten by the last job (persistence_image only).
This script recovers data from:
- .o log files for: persistence_landscapes, persistence_silhouette, betti_curves, persistence_entropy, template_functions, ECT (Retina+Derma only)
- Individual JSON files for: edge_histogram, lbp_texture, minkowski_functionals
- Per-dataset JSON files for: ECT Blood/Path/Organ (from rerun jobs)
- Current JSON file for: persistence_image (already there)
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np

RESULTS_DIR = Path("results/benchmark3/exp4")
LOGS_DIR = Path("benchmarks/benchmark3")


def parse_log_file(log_path: str, descriptor: str) -> dict:
    """Parse a .o log file to extract parameter search results for a descriptor."""
    with open(log_path) as f:
        content = f.read()

    datasets = ["BloodMNIST", "PathMNIST", "RetinaMNIST", "DermaMNIST", "OrganAMNIST"]
    results = {}

    for dataset in datasets:
        # Find the section for this dataset
        ds_pattern = rf"Dataset: {dataset}\n.*?(?=Dataset:|PARAMETER RULES)"
        ds_match = re.search(ds_pattern, content, re.DOTALL)
        if not ds_match:
            continue

        ds_text = ds_match.group(0)

        # Extract dimension info: "at n_layers=35 (3500D) [XGBoost]"
        dim_match = re.search(
            rf"{descriptor}: Testing \d+ parameter configs at (\w+)=(\S+) \((\d+)D\) \[(\w+)\]",
            ds_text
        )
        if not dim_match:
            continue

        dim_param = dim_match.group(1)
        dim_value_str = dim_match.group(2)
        dim_size = int(dim_match.group(3))
        classifier = dim_match.group(4)

        # Parse dimension value
        try:
            dim_value = int(dim_value_str)
        except ValueError:
            try:
                dim_value = float(dim_value_str)
            except ValueError:
                dim_value = dim_value_str

        # Extract individual results: "[1/8] {'combine_dims': True, 'n_bins': 50}... 0.882 (107.6s)"
        result_pattern = r"\[\d+/\d+\] (\{[^}]+\})\.\.\.\s*([\d.]+)\s*\(([\d.]+)s\)"
        all_results = []
        for match in re.finditer(result_pattern, ds_text):
            params_str = match.group(1)
            accuracy = float(match.group(2))
            time_s = float(match.group(3))

            # Parse params dict
            params = {}
            # Handle Python-style booleans and strings
            params_str_clean = params_str.replace("True", "true").replace("False", "false")
            params_str_clean = params_str_clean.replace("'", '"')
            try:
                params = json.loads(params_str_clean)
            except json.JSONDecodeError:
                # Manual parse
                for item in re.findall(r"'(\w+)'\s*:\s*([^,}]+)", params_str):
                    key = item[0]
                    val = item[1].strip().strip("'\"")
                    if val == "True":
                        params[key] = True
                    elif val == "False":
                        params[key] = False
                    else:
                        try:
                            params[key] = float(val) if '.' in val else int(val)
                        except ValueError:
                            params[key] = val

            all_results.append({
                "params": params,
                "accuracy": accuracy,
                "time": time_s,
            })

        # Extract BEST line
        best_match = re.search(
            r"BEST: (\{[^}]+\}) = ([\d.]+)",
            ds_text
        )
        best_params = {}
        best_accuracy = 0.0
        if best_match:
            best_str = best_match.group(1)
            best_accuracy = float(best_match.group(2))
            best_str_clean = best_str.replace("True", "true").replace("False", "false").replace("'", '"')
            try:
                best_params = json.loads(best_str_clean)
            except json.JSONDecodeError:
                for item in re.findall(r"'(\w+)'\s*:\s*([^,}]+)", best_str):
                    key = item[0]
                    val = item[1].strip().strip("'\"")
                    if val == "True":
                        best_params[key] = True
                    elif val == "False":
                        best_params[key] = False
                    else:
                        try:
                            best_params[key] = float(val) if '.' in val else int(val)
                        except ValueError:
                            best_params[key] = val

        # Skip datasets where all configs errored (best_accuracy=0, no valid results)
        if best_accuracy <= 0 and not all_results:
            continue

        results[dataset] = {
            "descriptor": descriptor,
            "dataset": dataset,
            "dimension_param": dim_param,
            "dimension_value": dim_value,
            "dimension_size": dim_size,
            "classifier": classifier,
            "best_params": best_params,
            "best_accuracy": best_accuracy,
            "all_results": all_results,
            "n_configs_tested": len(all_results),
        }

    return results


def analyze_all_results(consolidated: dict) -> dict:
    """Generate analysis section with per_descriptor and rules."""
    analysis = {
        "per_descriptor": {},
        "rules": {},
    }

    # Aggregate by descriptor
    for dataset, ds_results in consolidated["results"].items():
        for descriptor, desc_result in ds_results.items():
            if descriptor not in analysis["per_descriptor"]:
                analysis["per_descriptor"][descriptor] = []

            if desc_result.get("best_params"):
                analysis["per_descriptor"][descriptor].append({
                    "dataset": dataset,
                    "best_params": desc_result["best_params"],
                    "best_accuracy": desc_result.get("best_accuracy", 0),
                })

    # Find most common best params per descriptor
    for descriptor, results_list in analysis["per_descriptor"].items():
        if not results_list:
            continue

        param_counts = {}
        param_accs = {}

        for r in results_list:
            params = r["best_params"]
            param_key = tuple(sorted((k, str(v)) for k, v in params.items()))

            if param_key not in param_counts:
                param_counts[param_key] = 0
                param_accs[param_key] = []

            param_counts[param_key] += 1
            param_accs[param_key].append(r["best_accuracy"])

        best_key = max(param_counts.keys(),
                       key=lambda k: (param_counts[k], np.mean(param_accs[k])))
        best_params = dict(best_key)

        analysis["rules"][descriptor] = {
            "recommended_params": best_params,
            "frequency": param_counts[best_key],
            "mean_accuracy": float(np.mean(param_accs[best_key])),
            "n_datasets": len(results_list),
        }

    return analysis


def main():
    # 1. Load current JSON (has persistence_image data)
    pi_json_path = RESULTS_DIR / "exp4_parameter_search_n2000.json"
    with open(pi_json_path) as f:
        pi_data = json.load(f)

    # Initialize consolidated results
    consolidated = {
        "n_samples": 2000,
        "n_folds": 5,
        "seed": 42,
        "results": {},
    }

    datasets = ["BloodMNIST", "PathMNIST", "RetinaMNIST", "DermaMNIST", "OrganAMNIST"]
    for ds in datasets:
        consolidated["results"][ds] = {}

    # 2. Copy persistence_image from current JSON
    print("Loading persistence_image from current JSON...")
    for ds in datasets:
        if ds in pi_data["results"] and "persistence_image" in pi_data["results"][ds]:
            consolidated["results"][ds]["persistence_image"] = pi_data["results"][ds]["persistence_image"]
            print(f"  {ds}: persistence_image = {pi_data['results'][ds]['persistence_image']['best_accuracy']:.3f}")

    # 3. Parse log files for 6 descriptors (including template_functions and ECT)
    log_sources = {
        "persistence_landscapes": LOGS_DIR / "param_pl.o87192",
        "persistence_silhouette": LOGS_DIR / "param_sil.o87207",
        "betti_curves": LOGS_DIR / "param_bc.o87208",
        "persistence_entropy": LOGS_DIR / "param_ent.o87209",
        "template_functions": LOGS_DIR / "param_tf.o87211",
        "euler_characteristic_transform": LOGS_DIR / "param_ect.o87213",  # Retina+Derma only (Blood/Path/Organ errored)
    }

    for descriptor, log_path in log_sources.items():
        print(f"\nParsing {descriptor} from {log_path.name}...")
        parsed = parse_log_file(str(log_path), descriptor)
        for ds, result in parsed.items():
            consolidated["results"][ds][descriptor] = result
            print(f"  {ds}: {descriptor} = {result['best_accuracy']:.3f}")

    # 4. Load individual JSON files for 3 descriptors
    json_sources = {
        "edge_histogram": RESULTS_DIR / "exp4_param_edge_histogram_n2000.json",
        "lbp_texture": RESULTS_DIR / "exp4_param_lbp_texture_n2000.json",
        "minkowski_functionals": RESULTS_DIR / "exp4_param_minkowski_functionals_n2000.json",
    }

    for descriptor, json_path in json_sources.items():
        print(f"\nLoading {descriptor} from {json_path.name}...")
        with open(json_path) as f:
            data = json.load(f)
        for ds in datasets:
            if ds in data["results"] and descriptor in data["results"][ds]:
                consolidated["results"][ds][descriptor] = data["results"][ds][descriptor]
                print(f"  {ds}: {descriptor} = {data['results'][ds][descriptor]['best_accuracy']:.3f}")

    # 5. Load per-dataset JSON files (from single-dataset rerun jobs)
    #    These have filenames like: exp4_param_{descriptor}_{dataset}_n2000.json
    import glob
    per_dataset_pattern = str(RESULTS_DIR / "exp4_param_*_*MNIST_n2000.json")
    per_dataset_files = glob.glob(per_dataset_pattern)
    if per_dataset_files:
        print(f"\nLoading {len(per_dataset_files)} per-dataset JSON files...")
        for json_path in sorted(per_dataset_files):
            json_path = Path(json_path)
            print(f"  Reading {json_path.name}...")
            with open(json_path) as f:
                data = json.load(f)
            for ds in datasets:
                if ds in data.get("results", {}):
                    for descriptor, desc_result in data["results"][ds].items():
                        if desc_result.get("best_accuracy", 0) > 0:
                            existing = consolidated["results"][ds].get(descriptor, {})
                            existing_acc = existing.get("best_accuracy", 0)
                            new_acc = desc_result["best_accuracy"]
                            # Only replace if new result is better or no existing result
                            if new_acc > existing_acc or not existing:
                                consolidated["results"][ds][descriptor] = desc_result
                                print(f"    {ds}: {descriptor} = {new_acc:.3f}")

    # 6. Verify completeness
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    total = 0
    for ds in datasets:
        n_desc = len(consolidated["results"][ds])
        total += n_desc
        descs = list(consolidated["results"][ds].keys())
        print(f"  {ds}: {n_desc} descriptors: {', '.join(descs)}")
    print(f"\n  Total: {total} (expected: 50 = 10 descriptors × 5 datasets, or 47 if ECT Blood/Path/Organ not yet done)")

    # 6. Generate analysis
    consolidated["analysis"] = analyze_all_results(consolidated)

    # Print rules summary
    print(f"\n{'='*60}")
    print("PARAMETER RULES")
    print(f"{'='*60}")
    for descriptor, rule in consolidated["analysis"]["rules"].items():
        print(f"  {descriptor}: {rule['recommended_params']} (freq={rule['frequency']}, mean_acc={rule['mean_accuracy']:.3f})")

    # 7. Save (with backup)
    output_path = RESULTS_DIR / "exp4_parameter_search_n2000.json"
    if output_path.exists():
        import shutil
        import time
        backup_path = output_path.with_suffix(f'.backup_{int(time.time())}.json')
        shutil.copy2(output_path, backup_path)
        print(f"\n  Backed up existing file to: {backup_path}")

    with open(output_path, "w") as f:
        json.dump(consolidated, f, indent=2, default=str)

    print(f"\nSaved consolidated results to: {output_path}")


if __name__ == "__main__":
    main()
