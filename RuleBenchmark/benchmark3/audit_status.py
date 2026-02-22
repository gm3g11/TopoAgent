#!/usr/bin/env python
"""Print comprehensive status of Exp4.1, 4.2, 4.3."""

import json
import glob
from pathlib import Path

R = Path("results/benchmark3/exp4")
datasets = ["BloodMNIST", "PathMNIST", "RetinaMNIST", "DermaMNIST", "OrganAMNIST"]
all_15 = [
    "persistence_image", "persistence_landscapes", "betti_curves",
    "persistence_silhouette", "persistence_entropy", "persistence_statistics",
    "tropical_coordinates", "persistence_codebook", "ATOL", "template_functions",
    "minkowski_functionals", "euler_characteristic_curve",
    "euler_characteristic_transform", "edge_histogram", "lbp_texture",
]
param_10 = [
    "persistence_image", "persistence_landscapes", "persistence_silhouette",
    "betti_curves", "persistence_entropy", "template_functions",
    "minkowski_functionals", "euler_characteristic_transform",
    "lbp_texture", "edge_histogram",
]

# --- EXP 4.1: DIMENSION ---
with open(R / "exp4_optimal_dimensions.json") as f:
    dims = json.load(f)

# Check for per-dataset OrganAMNIST files from completed jobs
org_files = glob.glob(str(R / "exp4_fine_*_OrganAMNIST_n2000.json"))
org_done_from_jobs = set()
for fp in org_files:
    with open(fp) as f:
        data = json.load(f)
    for ds, v in data.items():
        if isinstance(v, dict) and v.get("best_accuracy", 0) > 0:
            name = Path(fp).stem
            desc = name.replace("exp4_fine_", "").replace("_OrganAMNIST_n2000", "")
            org_done_from_jobs.add(desc)

print("=" * 80)
print("  EXP 4.1: DIMENSION STUDY (D*) -- 15 descriptors x 5 datasets = 75 cells")
print("=" * 80)
print()
header = f"{'':35}" + "".join(f"{ds[:6]:>8}" for ds in datasets)
print(header)
print("-" * len(header))

dim_done = 0
dim_submitted = 0
for desc in all_15:
    row = f"{desc:35}"
    for ds in datasets:
        entry = dims.get(ds, {}).get(desc, {})
        if isinstance(entry, dict):
            acc = entry.get("accuracy")
            note = entry.get("note", "")
            if acc is not None:
                row += f"{acc:8.3f}"
                dim_done += 1
            elif desc in org_done_from_jobs and ds == "OrganAMNIST":
                row += "    DONE"
                dim_done += 1
            elif note == "estimated" and ds == "OrganAMNIST":
                row += "     RUN"
                dim_submitted += 1
            else:
                row += "     ???"
        else:
            row += "     ???"
    print(row)

print()
print(f"  DONE: {dim_done}  |  RUNNING: {dim_submitted}  |  Total: {dim_done + dim_submitted}/75")

# --- EXP 4.2: PARAMETER SEARCH ---
with open(R / "exp4_parameter_search_n2000.json") as f:
    params = json.load(f)

print()
print("=" * 80)
print("  EXP 4.2: PARAMETER SEARCH (P*) -- 10 descriptors x 5 datasets = 50 cells")
print("=" * 80)
print("  (5 dim-only descriptors excluded: codebook, tropical, ATOL, ECC, stats)")
print()
header = f"{'':35}" + "".join(f"{ds[:6]:>8}" for ds in datasets)
print(header)
print("-" * len(header))

param_done = 0
param_run = 0
for desc in param_10:
    row = f"{desc:35}"
    for ds in datasets:
        entry = params.get("results", {}).get(ds, {}).get(desc, {})
        acc = entry.get("best_accuracy", None)
        if acc is not None and acc > 0:
            row += f"{acc:8.3f}"
            param_done += 1
        else:
            row += "     RUN"
            param_run += 1
    print(row)

print()
print(f"  DONE: {param_done}  |  RUNNING: {param_run}  |  Total: {param_done + param_run}/50")

# --- EXP 4.3: COLOR ---
with open(R / "exp4_color_experiment_consolidated_n2000.json") as f:
    color = json.load(f)

print()
print("=" * 80)
print("  EXP 4.3: COLOR RULES -- 5 object types")
print("=" * 80)
print()
obj_types = ["discrete_cells", "glands_lumens", "vessel_trees", "surface_lesions", "organ_shape"]
ds_map = {
    "discrete_cells": "BloodMNIST", "glands_lumens": "PathMNIST",
    "vessel_trees": "RetinaMNIST", "surface_lesions": "DermaMNIST",
    "organ_shape": "OrganAMNIST",
}
color_done = 0
for ot in obj_types:
    data = color.get(ot, {})
    rec = data.get("recommendation", "MISSING")
    imp = data.get("improvement", 0)
    ds = ds_map[ot]
    status = "DONE" if rec != "MISSING" else "MISSING"
    if status == "DONE":
        color_done += 1
    print(f"  {ot:20} ({ds:12}): {rec:12} {imp:+.1f}%  [{status}]")
print()
print(f"  DONE: {color_done}/5")

# --- GRAND SUMMARY ---
print()
print("=" * 80)
print("  GRAND SUMMARY")
print("=" * 80)
print(f"  Exp4.1 Dimension:  {dim_done}/75 done, {dim_submitted} running")
print(f"  Exp4.2 Parameter:  {param_done}/50 done, {param_run} running")
print(f"  Exp4.3 Color:      {color_done}/5 done")
print()
total = dim_done + param_done + color_done
total_target = 75 + 50 + 5
running = dim_submitted + param_run
print(f"  Overall: {total}/{total_target} done, {running} running")
print(f"  After all jobs complete: {total + running}/{total_target}")

if total + running < total_target:
    print()
    print("  NOTE: OrganAMNIST param search used estimated D*.")
    print("        After dim study completes, may want to re-run param search")
    print("        at the real D* for full accuracy.")
