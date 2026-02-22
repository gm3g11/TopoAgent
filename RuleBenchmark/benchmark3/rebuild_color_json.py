#!/usr/bin/env python
"""
Rebuild exp4_color_experiment_n2000.json from log files.

The original 4 completed jobs all wrote to the same file, so only
glands_lumens survives. This script recovers all 5 object types:
- discrete_cells (BloodMNIST) from color_discrete_cells.o85844
- glands_lumens (PathMNIST) from color_glands_lumens.o85845
- vessel_trees (RetinaMNIST) from color_vessel_trees.o85846
- surface_lesions (DermaMNIST) from color_surface_lesions.o86079
- organ_shape (OrganAMNIST) - grayscale dataset, trivial result
"""

import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path("results/benchmark3/exp4")
LOGS_DIR = Path("benchmarks/benchmark3")

# Log files for each object type
LOG_SOURCES = {
    "discrete_cells": LOGS_DIR / "color_discrete_cells.o85844",
    "glands_lumens": LOGS_DIR / "color_glands_lumens.o85845",
    "vessel_trees": LOGS_DIR / "color_vessel_trees.o85846",
    "surface_lesions": LOGS_DIR / "color_surface_lesions.o86079",
}

# Also check for per-object-type JSON files (from fixed script)
JSON_SOURCES = {
    "discrete_cells": RESULTS_DIR / "exp4_color_discrete_cells_n2000.json",
    "glands_lumens": RESULTS_DIR / "exp4_color_glands_lumens_n2000.json",
    "vessel_trees": RESULTS_DIR / "exp4_color_vessel_trees_n2000.json",
    "surface_lesions": RESULTS_DIR / "exp4_color_surface_lesions_n2000.json",
    "organ_shape": RESULTS_DIR / "exp4_color_organ_shape_n2000.json",
}

OBJECT_TYPE_DATASETS = {
    "discrete_cells": "BloodMNIST",
    "glands_lumens": "PathMNIST",
    "vessel_trees": "RetinaMNIST",
    "surface_lesions": "DermaMNIST",
    "organ_shape": "OrganAMNIST",
}

DESCRIPTORS = ["persistence_image", "betti_curves", "persistence_statistics", "ATOL"]


def parse_color_log(log_path: str, obj_type: str) -> dict:
    """Parse a color experiment log file."""
    with open(log_path) as f:
        content = f.read()

    result = {
        "object_type": obj_type,
        "dataset": OBJECT_TYPE_DATASETS[obj_type],
        "descriptors": {},
        "recommendation": None,
        "improvement": None,
    }

    # Extract per-descriptor results for grayscale and per_channel
    for descriptor in DESCRIPTORS:
        desc_result = {}

        # Pattern: "  Descriptor: persistence_image" followed by grayscale/per_channel results
        # Look for "grayscale: 0.943 ± 0.008 (367.8s)" and "per_channel: 0.969 ± 0.006 (1727.7s)"
        gray_pattern = rf"{descriptor}.*?grayscale:\s*([\d.]+)\s*±\s*([\d.]+)"
        gray_match = re.search(gray_pattern, content, re.DOTALL)
        if gray_match:
            desc_result["grayscale"] = {
                "accuracy": float(gray_match.group(1)),
                "std": float(gray_match.group(2)),
            }

        pc_pattern = rf"{descriptor}.*?per_channel:\s*([\d.]+)\s*±\s*([\d.]+)"
        pc_match = re.search(pc_pattern, content, re.DOTALL)
        if pc_match:
            desc_result["per_channel"] = {
                "accuracy": float(pc_match.group(1)),
                "std": float(pc_match.group(2)),
            }

        if desc_result:
            result["descriptors"][descriptor] = desc_result

    # Extract recommendation
    rec_match = re.search(r"Best method:\s*(\w+)", content)
    if rec_match:
        result["recommendation"] = rec_match.group(1)

    imp_match = re.search(r"vs grayscale:\s*([+-][\d.]+)%", content)
    if imp_match:
        result["improvement"] = float(imp_match.group(1))

    return result


def main():
    consolidated = {}

    # 1. Try loading from per-object-type JSON files first (from fixed script)
    for obj_type, json_path in JSON_SOURCES.items():
        if json_path.exists():
            print(f"Loading {obj_type} from JSON: {json_path.name}")
            with open(json_path) as f:
                data = json.load(f)
            if obj_type in data:
                consolidated[obj_type] = data[obj_type]
            else:
                # Might be the only key
                for key, val in data.items():
                    if isinstance(val, dict) and "recommendation" in str(val):
                        consolidated[obj_type] = val
                        break

    # 2. Parse log files for anything not yet loaded
    for obj_type, log_path in LOG_SOURCES.items():
        if obj_type in consolidated:
            print(f"  {obj_type}: already loaded from JSON, skipping log")
            continue

        if not log_path.exists():
            print(f"  WARNING: {log_path} not found, skipping {obj_type}")
            continue

        print(f"Parsing {obj_type} from log: {log_path.name}...")
        parsed = parse_color_log(str(log_path), obj_type)

        if parsed["recommendation"]:
            consolidated[obj_type] = parsed
            dataset = OBJECT_TYPE_DATASETS[obj_type]
            print(f"  {dataset}: {parsed['recommendation']} ({parsed['improvement']:+.1f}%)")
        else:
            print(f"  WARNING: No recommendation found for {obj_type}")

    # 3. Add organ_shape (OrganAMNIST is grayscale - trivial result)
    if "organ_shape" not in consolidated:
        print(f"\nAdding organ_shape (OrganAMNIST is grayscale, trivial)...")
        consolidated["organ_shape"] = {
            "object_type": "organ_shape",
            "dataset": "OrganAMNIST",
            "recommendation": "grayscale",
            "improvement": 0.0,
            "note": "OrganAMNIST is a grayscale dataset, per_channel not applicable",
            "descriptors": {},
        }

    # 4. Also load the existing glands_lumens JSON if available (may have more detail)
    existing_json = RESULTS_DIR / "exp4_color_experiment_n2000.json"
    if existing_json.exists():
        print(f"\nChecking existing JSON: {existing_json.name}...")
        with open(existing_json) as f:
            existing = json.load(f)
        for obj_type, data in existing.items():
            if obj_type not in consolidated and isinstance(data, dict):
                consolidated[obj_type] = data
                print(f"  Added {obj_type} from existing JSON")

    # 5. Summary
    print(f"\n{'='*60}")
    print("COLOR RULES SUMMARY")
    print(f"{'='*60}")
    for obj_type in ["discrete_cells", "glands_lumens", "vessel_trees", "surface_lesions", "organ_shape"]:
        data = consolidated.get(obj_type, {})
        dataset = OBJECT_TYPE_DATASETS.get(obj_type, "?")
        rec = data.get("recommendation", "MISSING")
        imp = data.get("improvement", 0)
        status = "DONE" if rec != "MISSING" else "MISSING"
        print(f"  {obj_type:<20} ({dataset:<12}): {rec:<12} {imp:+.1f}%  [{status}]")

    done = sum(1 for d in consolidated.values() if d.get("recommendation"))
    print(f"\n  {done}/5 object types complete")

    # 6. Save
    output_path = RESULTS_DIR / "exp4_color_experiment_consolidated_n2000.json"
    if output_path.exists():
        import shutil
        import time
        backup_path = output_path.with_suffix(f'.backup_{int(time.time())}.json')
        shutil.copy2(output_path, backup_path)
        print(f"\n  Backed up existing file to: {backup_path}")

    with open(output_path, "w") as f:
        json.dump(consolidated, f, indent=2, default=str)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
