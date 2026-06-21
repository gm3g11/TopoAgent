#!/usr/bin/env python3
"""
Compute execution cost statistics and failure analysis for TopoAgent paper.
Covers Prompt 1 (efficiency/cost), Prompt 2 (failure analysis), Prompt 3 (sanity check).

Data sources:
- results/ablation_study/c0_full/*.json  (26 datasets × 5 demo images = 130 traces)
- results/case_studies/*_multi_image_case_study.json  (detailed v9 traces)
- topoagent/skills/rules_data.py  (ground truth rankings)
"""

import json
import os
import glob
import sys
from collections import defaultdict, Counter
import statistics

# ============================================================
# CONFIG
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C0_DIR = os.path.join(BASE, "results", "ablation_study", "c0_full")
MULTI_IMG_DIR = os.path.join(BASE, "results", "case_studies")

# Ground truth object types
DATASET_TO_OBJECT_TYPE = {
    "BloodMNIST": "discrete_cells",
    "TissueMNIST": "discrete_cells",
    "PathMNIST": "glands_lumens",
    "OCTMNIST": "organ_shape",
    "OrganAMNIST": "organ_shape",
    "RetinaMNIST": "vessel_trees",
    "PneumoniaMNIST": "organ_shape",
    "BreastMNIST": "organ_shape",
    "DermaMNIST": "surface_lesions",
    "OrganCMNIST": "organ_shape",
    "OrganSMNIST": "organ_shape",
    "ISIC2019": "surface_lesions",
    "Kvasir": "glands_lumens",
    "BrainTumorMRI": "organ_shape",
    "MURA": "organ_shape",
    "BreakHis": "glands_lumens",
    "NCT_CRC_HE": "glands_lumens",
    "MalariaCell": "discrete_cells",
    "IDRiD": "vessel_trees",
    "PCam": "discrete_cells",
    "LC25000": "glands_lumens",
    "SIPaKMeD": "discrete_cells",
    "AML_Cytomorphology": "discrete_cells",
    "APTOS2019": "vessel_trees",
    "GasHisSDB": "surface_lesions",
    "Chaoyang": "glands_lumens",
}

# Benchmark ground truth: top-3 per object type (from SUPPORTED_TOP_PERFORMERS)
BENCHMARK_TOP3 = {
    "discrete_cells": ["minkowski_functionals", "template_functions", "persistence_statistics"],
    "glands_lumens": ["persistence_statistics", "minkowski_functionals", "euler_characteristic_curve"],
    "vessel_trees": ["lbp_texture", "persistence_statistics", "template_functions"],
    "surface_lesions": ["persistence_statistics", "minkowski_functionals", "euler_characteristic_curve"],
    "organ_shape": ["minkowski_functionals", "persistence_statistics", "lbp_texture"],
}

BENCHMARK_TOP1 = {k: v[0] for k, v in BENCHMARK_TOP3.items()}

# ============================================================
# DATA LOADING
# ============================================================

def load_c0_data():
    """Load all ablation c0_full data (per-image details)."""
    all_images = []
    all_datasets = {}
    for fpath in sorted(glob.glob(os.path.join(C0_DIR, "*.json"))):
        with open(fpath) as f:
            data = json.load(f)
        dataset = data["dataset"]
        all_datasets[dataset] = data
        gt_ot = DATASET_TO_OBJECT_TYPE.get(dataset, "unknown")
        for img in data.get("per_image_details", []):
            img["_dataset"] = dataset
            img["_gt_object_type"] = gt_ot
            img["_balanced_accuracy"] = data.get("balanced_accuracy")
            img["_majority_descriptor"] = data.get("majority_descriptor")
            all_images.append(img)
    return all_images, all_datasets


def load_multi_image_data():
    """Load multi-image case study data (with richer v9 phase info)."""
    all_images = []
    for fpath in sorted(glob.glob(os.path.join(MULTI_IMG_DIR, "*_multi_image_case_study.json"))):
        with open(fpath) as f:
            data = json.load(f)
        dataset = data["dataset"]
        gt_ot = DATASET_TO_OBJECT_TYPE.get(dataset, "unknown")
        for img in data.get("per_image_results", []):
            img["_dataset"] = dataset
            img["_gt_object_type"] = gt_ot
            all_images.append(img)
    return all_images


# ============================================================
# PROMPT 3: SANITY CHECK
# ============================================================

def sanity_check(images, datasets, multi_images):
    print("=" * 70)
    print("PROMPT 3: SANITY CHECK")
    print("=" * 70)

    print(f"\n1. Total ablation c0_full files: {len(datasets)}")
    print(f"   Total per-image traces (c0): {len(images)}")
    print(f"   Total multi-image case study traces: {len(multi_images)}")

    print(f"\n2. Unique datasets (c0): {len(set(im['_dataset'] for im in images))}")
    ds_list = sorted(set(im['_dataset'] for im in images))
    for d in ds_list:
        print(f"      {d}")

    print(f"\n3. Unique object types detected by agent:")
    ot_counter = Counter(im.get("object_type", "N/A") for im in images)
    for ot, cnt in ot_counter.most_common():
        print(f"      {ot}: {cnt} images")

    print(f"\n4. Incomplete / error traces:")
    incomplete = [im for im in images if im.get("n_llm_calls", 0) == 0 or im.get("time_seconds", 0) == 0]
    if incomplete:
        for im in incomplete:
            print(f"      {im['_dataset']} img={im['image_index']}: n_llm_calls={im.get('n_llm_calls')}, time={im.get('time_seconds')}")
    else:
        print("      None — all traces complete")

    print(f"\n5. Unique agent stances:")
    stance_counter = Counter(im.get("stance", "N/A") for im in images)
    for s, cnt in stance_counter.most_common():
        print(f"      {s}: {cnt}")

    print(f"\n6. Unique final descriptors:")
    desc_counter = Counter(im.get("descriptor", "N/A") for im in images)
    for d, cnt in desc_counter.most_common():
        print(f"      {d}: {cnt}")

    # Show examples
    print(f"\n7. Example traces:")
    # FOLLOW
    follow = [im for im in images if im.get("stance") == "FOLLOW"]
    if follow:
        ex = follow[0]
        print(f"\n   FOLLOW example: {ex['_dataset']} img={ex['image_index']}")
        print(f"      descriptor={ex['descriptor']}, n_llm_calls={ex['n_llm_calls']}, time={ex['time_seconds']:.1f}s, retry={ex.get('retry_used', False)}")

    # DEVIATE
    deviate = [im for im in images if im.get("stance") == "DEVIATE"]
    if deviate:
        ex = deviate[0]
        print(f"\n   DEVIATE example: {ex['_dataset']} img={ex['image_index']}")
        print(f"      descriptor={ex['descriptor']}, n_llm_calls={ex['n_llm_calls']}, time={ex['time_seconds']:.1f}s, retry={ex.get('retry_used', False)}")

    # Retry
    retry = [im for im in images if im.get("retry_used", False)]
    if retry:
        ex = retry[0]
        print(f"\n   RETRY example: {ex['_dataset']} img={ex['image_index']}")
        print(f"      descriptor={ex['descriptor']}, n_llm_calls={ex['n_llm_calls']}, time={ex['time_seconds']:.1f}s, retry={ex.get('retry_used', True)}")

    print()


# ============================================================
# PROMPT 1: EFFICIENCY / COST TABLE
# ============================================================

def efficiency_analysis(images, datasets):
    print("=" * 70)
    print("PROMPT 1: EFFICIENCY / COST TABLE")
    print("=" * 70)

    # --- 1. Per-image averages across ALL datasets ---
    n_total = len(images)
    llm_calls = [im["n_llm_calls"] for im in images]
    times = [im["time_seconds"] for im in images]
    retries = [im for im in images if im.get("retry_used", False)]
    no_retries = [im for im in images if not im.get("retry_used", False)]

    print(f"\n### 1. Per-Image Averages (N={n_total} images, {len(datasets)} datasets)")
    print(f"{'Metric':<45} {'Value':>10}")
    print("-" * 57)
    print(f"{'Total images':<45} {n_total:>10}")
    print(f"{'Avg LLM calls (all)':<45} {statistics.mean(llm_calls):>10.2f}")
    print(f"{'Median LLM calls':<45} {statistics.median(llm_calls):>10.1f}")
    if no_retries:
        nr_calls = [im["n_llm_calls"] for im in no_retries]
        print(f"{'Avg LLM calls (no-retry only)':<45} {statistics.mean(nr_calls):>10.2f}")
    if retries:
        r_calls = [im["n_llm_calls"] for im in retries]
        print(f"{'Avg LLM calls (retry cases only)':<45} {statistics.mean(r_calls):>10.2f}")
    print(f"{'Retry rate':<45} {100*len(retries)/n_total:>9.1f}%")
    print(f"{'Avg wall-clock time (all)':<45} {statistics.mean(times):>9.1f}s")
    print(f"{'Median wall-clock time':<45} {statistics.median(times):>9.1f}s")
    print(f"{'Std wall-clock time':<45} {statistics.stdev(times):>9.1f}s")
    print(f"{'Min wall-clock time':<45} {min(times):>9.1f}s")
    print(f"{'Max wall-clock time':<45} {max(times):>9.1f}s")
    if no_retries:
        nr_times = [im["time_seconds"] for im in no_retries]
        print(f"{'Avg wall-clock time (no-retry)':<45} {statistics.mean(nr_times):>9.1f}s")
    if retries:
        r_times = [im["time_seconds"] for im in retries]
        print(f"{'Avg wall-clock time (retry cases)':<45} {statistics.mean(r_times):>9.1f}s")

    # LLM calls distribution
    print(f"\n### LLM Calls Distribution")
    call_dist = Counter(llm_calls)
    for k in sorted(call_dist.keys()):
        pct = 100 * call_dist[k] / n_total
        print(f"   {k} calls: {call_dist[k]:>3} images ({pct:5.1f}%)")

    # --- 2. Per-dataset breakdown ---
    print(f"\n### 2. Per-Dataset Breakdown")
    print(f"{'Dataset':<25} {'N':>3} {'Calls':>6} {'Retry':>6} {'Time(s)':>8} {'Descriptor':<25} {'Acc':>6}")
    print("-" * 85)
    for ds_name in sorted(datasets.keys()):
        ds = datasets[ds_name]
        imgs = [im for im in images if im["_dataset"] == ds_name]
        n = len(imgs)
        avg_calls = statistics.mean([im["n_llm_calls"] for im in imgs])
        n_retry = sum(1 for im in imgs if im.get("retry_used", False))
        avg_time = statistics.mean([im["time_seconds"] for im in imgs])
        maj_desc = ds.get("majority_descriptor", "N/A")
        bal_acc = ds.get("balanced_accuracy", 0)
        print(f"{ds_name:<25} {n:>3} {avg_calls:>6.1f} {n_retry:>6} {avg_time:>8.1f} {maj_desc:<25} {bal_acc:>5.1f}%")

    # --- 3. Per-object-type breakdown ---
    print(f"\n### 3. Per-Object-Type Breakdown")
    ot_groups = defaultdict(list)
    for im in images:
        ot_groups[im["_gt_object_type"]].append(im)

    print(f"{'Object Type':<20} {'N':>4} {'Calls':>6} {'Retry%':>7} {'Time(s)':>8}")
    print("-" * 50)
    for ot in sorted(ot_groups.keys()):
        imgs = ot_groups[ot]
        n = len(imgs)
        avg_calls = statistics.mean([im["n_llm_calls"] for im in imgs])
        retry_pct = 100 * sum(1 for im in imgs if im.get("retry_used", False)) / n
        avg_time = statistics.mean([im["time_seconds"] for im in imgs])
        print(f"{ot:<20} {n:>4} {avg_calls:>6.2f} {retry_pct:>6.1f}% {avg_time:>8.1f}")

    # --- 4. Demo + Eval time breakdown ---
    print(f"\n### 4. Total Pipeline Time (Demo + Eval)")
    demo_times = []
    eval_times = []
    total_times = []
    for ds in datasets.values():
        demo_times.append(ds.get("demo_time_seconds", 0))
        eval_times.append(ds.get("eval_time_seconds", 0))
        total_times.append(ds.get("total_time_seconds", 0))

    print(f"{'Metric':<40} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 75)
    print(f"{'Demo time (5 images, s)':<40} {statistics.mean(demo_times):>8.1f} {statistics.median(demo_times):>8.1f} {min(demo_times):>8.1f} {max(demo_times):>8.1f}")
    print(f"{'Eval time (200 images, s)':<40} {statistics.mean(eval_times):>8.1f} {statistics.median(eval_times):>8.1f} {min(eval_times):>8.1f} {max(eval_times):>8.1f}")
    print(f"{'Total time (s)':<40} {statistics.mean(total_times):>8.1f} {statistics.median(total_times):>8.1f} {min(total_times):>8.1f} {max(total_times):>8.1f}")

    # --- 5. LaTeX table ---
    print(f"\n### 5. LaTeX Table: Efficiency Statistics")
    print(r"""
\begin{table}[t]
\centering
\caption{TopoAgent execution cost per image (v9 pipeline, GPT-4o, 26 datasets).}
\label{tab:execution-cost}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule""")
    print(f"Total demo images & {n_total} \\\\")
    print(f"LLM calls (no retry) & {statistics.mean(nr_calls):.1f} \\\\")
    print(f"LLM calls (with retry) & {statistics.mean(r_calls):.1f} \\\\")
    print(f"LLM calls (overall avg) & {statistics.mean(llm_calls):.2f} \\\\")
    print(f"Retry rate & {100*len(retries)/n_total:.1f}\\% \\\\")
    print(f"Wall-clock time (no retry) & {statistics.mean(nr_times):.1f}s \\\\")
    print(f"Wall-clock time (with retry) & {statistics.mean(r_times):.1f}s \\\\")
    print(f"Wall-clock time (overall avg) & {statistics.mean(times):.1f}s \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    return {
        "n_total": n_total,
        "avg_llm_calls": statistics.mean(llm_calls),
        "avg_llm_calls_no_retry": statistics.mean(nr_calls) if no_retries else None,
        "avg_llm_calls_retry": statistics.mean(r_calls) if retries else None,
        "retry_rate": len(retries) / n_total,
        "avg_time": statistics.mean(times),
        "avg_time_no_retry": statistics.mean(nr_times) if no_retries else None,
        "avg_time_retry": statistics.mean(r_times) if retries else None,
    }


# ============================================================
# PROMPT 2: FAILURE ANALYSIS
# ============================================================

def failure_analysis(images, datasets):
    print("\n" + "=" * 70)
    print("PROMPT 2: FAILURE ANALYSIS")
    print("=" * 70)

    n_total = len(images)

    # --- A. Overall descriptor selection accuracy ---
    print(f"\n### A. Overall Descriptor Selection Accuracy (N={n_total})")

    # Per-image accuracy (descriptor vs benchmark top-k for the agent's perceived OT)
    # But more importantly: vs ground-truth OT
    top1_match = 0
    top3_match = 0
    wrong = 0
    wrong_cases = []

    # Also check against GT object type's benchmark
    gt_top1_match = 0
    gt_top3_match = 0
    gt_wrong = 0
    gt_wrong_cases = []

    for im in images:
        desc = im.get("descriptor", "")
        perceived_ot = im.get("object_type", "")
        gt_ot = im["_gt_object_type"]
        dataset = im["_dataset"]

        # vs perceived OT rankings
        p_top3 = BENCHMARK_TOP3.get(perceived_ot, [])
        p_top1 = BENCHMARK_TOP1.get(perceived_ot, "")
        if desc == p_top1:
            top1_match += 1
        if desc in p_top3:
            top3_match += 1
        else:
            wrong += 1

        # vs ground truth OT rankings
        gt_top3_list = BENCHMARK_TOP3.get(gt_ot, [])
        gt_top1 = BENCHMARK_TOP1.get(gt_ot, "")
        if desc == gt_top1:
            gt_top1_match += 1
        if desc in gt_top3_list:
            gt_top3_match += 1
        else:
            gt_wrong += 1
            gt_wrong_cases.append({
                "dataset": dataset,
                "image": im["image_index"],
                "perceived_ot": perceived_ot,
                "gt_ot": gt_ot,
                "descriptor": desc,
                "gt_top3": gt_top3_list,
                "stance": im.get("stance", ""),
            })

    print(f"\n  Against PERCEIVED object type rankings:")
    print(f"    descriptor == top-1:  {top1_match:>3}/{n_total} ({100*top1_match/n_total:.1f}%)")
    print(f"    descriptor in top-3:  {top3_match:>3}/{n_total} ({100*top3_match/n_total:.1f}%)")
    print(f"    descriptor NOT top-3: {wrong:>3}/{n_total} ({100*wrong/n_total:.1f}%)")

    print(f"\n  Against GROUND TRUTH object type rankings:")
    print(f"    descriptor == top-1:  {gt_top1_match:>3}/{n_total} ({100*gt_top1_match/n_total:.1f}%)")
    print(f"    descriptor in top-3:  {gt_top3_match:>3}/{n_total} ({100*gt_top3_match/n_total:.1f}%)")
    print(f"    descriptor NOT top-3: {gt_wrong:>3}/{n_total} ({100*gt_wrong/n_total:.1f}%)")

    # --- B. Error attribution by phase ---
    print(f"\n### B. Error Attribution (against GT object type, {gt_wrong} errors)")

    perception_errors = []
    reasoning_errors = []
    for case in gt_wrong_cases:
        if case["perceived_ot"] != case["gt_ot"]:
            perception_errors.append(case)
        else:
            reasoning_errors.append(case)

    print(f"\n  B1. Perception errors (wrong object type): {len(perception_errors)}/{gt_wrong}")
    if perception_errors:
        ot_confusions = Counter((c["gt_ot"], c["perceived_ot"]) for c in perception_errors)
        print(f"      Object type confusions:")
        for (gt, pred), cnt in ot_confusions.most_common():
            print(f"        {gt} → {pred}: {cnt} images")
        print(f"\n      Detailed perception errors:")
        for c in perception_errors[:10]:
            print(f"        {c['dataset']} img={c['image']}: GT={c['gt_ot']}, perceived={c['perceived_ot']}, desc={c['descriptor']}")

    print(f"\n  B2. Reasoning errors (correct OT but wrong descriptor): {len(reasoning_errors)}/{gt_wrong}")
    if reasoning_errors:
        for c in reasoning_errors[:10]:
            print(f"        {c['dataset']} img={c['image']}: OT={c['gt_ot']}, desc={c['descriptor']}, gt_top3={c['gt_top3']}, stance={c['stance']}")

    # --- C. Object type perception accuracy ---
    print(f"\n### C. Object Type Perception Accuracy")
    ot_correct = sum(1 for im in images if im.get("object_type") == im["_gt_object_type"])
    print(f"    Correct: {ot_correct}/{n_total} ({100*ot_correct/n_total:.1f}%)")
    print(f"    Wrong:   {n_total-ot_correct}/{n_total} ({100*(n_total-ot_correct)/n_total:.1f}%)")

    ot_confusion = defaultdict(lambda: Counter())
    for im in images:
        gt = im["_gt_object_type"]
        pred = im.get("object_type", "N/A")
        ot_confusion[gt][pred] += 1

    print(f"\n    Confusion matrix:")
    all_ots = sorted(set(list(ot_confusion.keys()) + [p for c in ot_confusion.values() for p in c.keys()]))
    print(f"    {'GT \\ Predicted':<20}", end="")
    for ot in all_ots:
        print(f" {ot[:12]:>12}", end="")
    print(f" {'Total':>6} {'Acc%':>6}")
    for gt in sorted(ot_confusion.keys()):
        print(f"    {gt:<20}", end="")
        total = sum(ot_confusion[gt].values())
        for ot in all_ots:
            cnt = ot_confusion[gt].get(ot, 0)
            print(f" {cnt:>12}", end="")
        correct = ot_confusion[gt].get(gt, 0)
        print(f" {total:>6} {100*correct/total:>5.1f}%")

    # --- D. Retry analysis ---
    print(f"\n### D. Retry Analysis")
    retries = [im for im in images if im.get("retry_used", False)]
    no_retries = [im for im in images if not im.get("retry_used", False)]
    print(f"    Images with retry:    {len(retries)}/{n_total} ({100*len(retries)/n_total:.1f}%)")
    print(f"    Images without retry: {len(no_retries)}/{n_total} ({100*len(no_retries)/n_total:.1f}%)")

    if retries:
        print(f"\n    Retry cases by dataset:")
        retry_by_ds = Counter(im["_dataset"] for im in retries)
        for ds, cnt in retry_by_ds.most_common():
            print(f"      {ds}: {cnt}/{sum(1 for im in images if im['_dataset']==ds)} images")

        # Check if retry descriptor is in top-3
        retry_in_top3 = sum(1 for im in retries
                           if im.get("descriptor") in BENCHMARK_TOP3.get(im["_gt_object_type"], []))
        print(f"\n    Retry images where final descriptor in GT top-3: {retry_in_top3}/{len(retries)} ({100*retry_in_top3/len(retries):.1f}%)")

    # --- E. Behavioral patterns ---
    print(f"\n### E. Behavioral Patterns (Stance Distribution)")
    stance_counter = Counter(im.get("stance", "N/A") for im in images)
    for s, cnt in stance_counter.most_common():
        # Accuracy of this stance
        stance_imgs = [im for im in images if im.get("stance") == s]
        in_top3 = sum(1 for im in stance_imgs
                     if im.get("descriptor") in BENCHMARK_TOP3.get(im["_gt_object_type"], []))
        in_top1 = sum(1 for im in stance_imgs
                     if im.get("descriptor") == BENCHMARK_TOP1.get(im["_gt_object_type"]))
        pct = 100 * cnt / n_total
        acc3 = 100 * in_top3 / cnt if cnt > 0 else 0
        acc1 = 100 * in_top1 / cnt if cnt > 0 else 0
        print(f"    {s:<25} {cnt:>3} ({pct:5.1f}%)  top-1: {acc1:5.1f}%  top-3: {acc3:5.1f}%")

    # --- F. Per-object-type error rates ---
    print(f"\n### F. Per-Object-Type Descriptor Selection Accuracy")
    print(f"{'Object Type':<20} {'N':>4} {'Top-1':>6} {'Top-3':>6} {'Wrong':>6} {'OT Acc':>7}")
    print("-" * 55)
    for ot in sorted(set(im["_gt_object_type"] for im in images)):
        ot_imgs = [im for im in images if im["_gt_object_type"] == ot]
        n = len(ot_imgs)
        t1 = sum(1 for im in ot_imgs if im.get("descriptor") == BENCHMARK_TOP1.get(ot))
        t3 = sum(1 for im in ot_imgs if im.get("descriptor") in BENCHMARK_TOP3.get(ot, []))
        w = n - t3
        ot_acc = sum(1 for im in ot_imgs if im.get("object_type") == ot)
        print(f"{ot:<20} {n:>4} {100*t1/n:>5.1f}% {100*t3/n:>5.1f}% {100*w/n:>5.1f}% {100*ot_acc/n:>6.1f}%")

    # --- G. Dataset-level accuracy (majority vote) ---
    print(f"\n### G. Dataset-Level Descriptor Selection (Majority Vote)")
    print(f"{'Dataset':<25} {'GT OT':<18} {'GT Top-1':<22} {'Majority Desc':<22} {'Match?':>6} {'Acc':>6}")
    print("-" * 105)
    n_ds_match_top1 = 0
    n_ds_match_top3 = 0
    for ds_name in sorted(datasets.keys()):
        ds = datasets[ds_name]
        gt_ot = DATASET_TO_OBJECT_TYPE.get(ds_name, "unknown")
        gt_top1 = BENCHMARK_TOP1.get(gt_ot, "?")
        gt_top3_list = BENCHMARK_TOP3.get(gt_ot, [])
        maj = ds.get("majority_descriptor", "N/A")
        match = "TOP-1" if maj == gt_top1 else ("TOP-3" if maj in gt_top3_list else "MISS")
        bal_acc = ds.get("balanced_accuracy", 0)
        if maj == gt_top1:
            n_ds_match_top1 += 1
        if maj in gt_top3_list:
            n_ds_match_top3 += 1
        print(f"{ds_name:<25} {gt_ot:<18} {gt_top1:<22} {maj:<22} {match:>6} {bal_acc:>5.1f}%")

    n_ds = len(datasets)
    print(f"\n    Dataset-level top-1 match: {n_ds_match_top1}/{n_ds} ({100*n_ds_match_top1/n_ds:.1f}%)")
    print(f"    Dataset-level top-3 match: {n_ds_match_top3}/{n_ds} ({100*n_ds_match_top3/n_ds:.1f}%)")

    # --- H. LaTeX failure analysis table ---
    print(f"\n### H. LaTeX Table: Failure Analysis")
    print(r"""
\begin{table}[t]
\centering
\caption{Descriptor selection accuracy and error attribution (v9 pipeline, 26 datasets).}
\label{tab:failure-analysis}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Per-image accuracy (vs.\ ground-truth rankings)}} \\""")
    print(f"~~Top-1 match & {gt_top1_match}/{n_total} ({100*gt_top1_match/n_total:.1f}\\%) \\\\")
    print(f"~~Top-3 match & {gt_top3_match}/{n_total} ({100*gt_top3_match/n_total:.1f}\\%) \\\\")
    print(f"~~Wrong (not in top-3) & {gt_wrong}/{n_total} ({100*gt_wrong/n_total:.1f}\\%) \\\\")
    print(r"\midrule")
    print(r"\multicolumn{2}{l}{\textit{Error attribution}} \\")
    print(f"~~Perception error (wrong OT) & {len(perception_errors)}/{gt_wrong} \\\\")
    print(f"~~Reasoning error (right OT, wrong desc) & {len(reasoning_errors)}/{gt_wrong} \\\\")
    print(r"\midrule")
    print(r"\multicolumn{2}{l}{\textit{Object type perception}} \\")
    print(f"~~OT correct & {ot_correct}/{n_total} ({100*ot_correct/n_total:.1f}\\%) \\\\")
    print(r"\midrule")
    print(r"\multicolumn{2}{l}{\textit{Dataset-level (majority vote)}} \\")
    print(f"~~Top-1 match & {n_ds_match_top1}/{n_ds} ({100*n_ds_match_top1/n_ds:.1f}\\%) \\\\")
    print(f"~~Top-3 match & {n_ds_match_top3}/{n_ds} ({100*n_ds_match_top3/n_ds:.1f}\\%) \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    return {
        "n_total": n_total,
        "gt_top1_match": gt_top1_match,
        "gt_top3_match": gt_top3_match,
        "gt_wrong": gt_wrong,
        "perception_errors": len(perception_errors),
        "reasoning_errors": len(reasoning_errors),
        "ot_correct": ot_correct,
        "ds_top1_match": n_ds_match_top1,
        "ds_top3_match": n_ds_match_top3,
    }


# ============================================================
# MULTI-IMAGE ANALYSIS (richer v9 fields)
# ============================================================

def multi_image_analysis(multi_images):
    """Analyze multi-image case studies for richer v9 phase details."""
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY: MULTI-IMAGE CASE STUDY ANALYSIS (v9 phase details)")
    print("=" * 70)

    n = len(multi_images)
    print(f"\nTotal images: {n}")

    # Reflect stats
    reflect_rounds = [im["reflect"]["n_rounds"] for im in multi_images if "reflect" in im]
    quality_ok = sum(1 for im in multi_images if im.get("reflect", {}).get("quality_ok", False))
    quality_fail = sum(1 for im in multi_images if not im.get("reflect", {}).get("quality_ok", True))

    print(f"\nReflection phase:")
    print(f"  Avg rounds: {statistics.mean(reflect_rounds):.2f}")
    print(f"  Quality OK: {quality_ok}/{n} ({100*quality_ok/n:.1f}%)")
    print(f"  Quality FAIL: {quality_fail}/{n} ({100*quality_fail/n:.1f}%)")
    rounds_dist = Counter(reflect_rounds)
    for r in sorted(rounds_dist.keys()):
        print(f"  {r} round(s): {rounds_dist[r]}")

    # LTM usage
    ltm_avail = [im.get("ltm_entries_available", 0) for im in multi_images]
    ltm_after = [im.get("ltm_entries_after", 0) for im in multi_images]
    print(f"\nLTM entries available: mean={statistics.mean(ltm_avail):.1f}, max={max(ltm_avail)}")
    print(f"LTM entries after:    mean={statistics.mean(ltm_after):.1f}, max={max(ltm_after)}")

    # Descriptor intuition vs final
    n_match_intuition = 0
    for im in multi_images:
        intuition = im.get("analysis", {}).get("descriptor_intuition", "")
        final = im.get("plan", {}).get("descriptor", "")
        if final.lower() in intuition.lower():
            n_match_intuition += 1
    print(f"\nDescriptor intuition matches final: {n_match_intuition}/{n} ({100*n_match_intuition/n:.1f}%)")

    # Plan stance distribution from multi-image
    stances = Counter(im.get("plan", {}).get("stance", "N/A") for im in multi_images)
    print(f"\nPlan stance distribution (multi-image):")
    for s, cnt in stances.most_common():
        print(f"  {s}: {cnt} ({100*cnt/n:.1f}%)")

    # Filtration types
    filt = Counter(im.get("perceive_decisions", {}).get("filtration_type", "N/A") for im in multi_images)
    print(f"\nFiltration type distribution:")
    for f, cnt in filt.most_common():
        print(f"  {f}: {cnt} ({100*cnt/n:.1f}%)")

    # Denoising
    denoise = sum(1 for im in multi_images
                  if im.get("perceive_decisions", {}).get("apply_denoising", False))
    print(f"\nDenoising applied: {denoise}/{n} ({100*denoise/n:.1f}%)")

    # Tools used frequency
    tool_counter = Counter()
    for im in multi_images:
        for t in im.get("tools_used", []):
            tool_counter[t] += 1
    print(f"\nMost common tools used:")
    for t, cnt in tool_counter.most_common(15):
        print(f"  {t}: {cnt} ({100*cnt/n:.1f}%)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")
    images, datasets = load_c0_data()
    multi_images = load_multi_image_data()

    # Prompt 3: Sanity check
    sanity_check(images, datasets, multi_images)

    # Prompt 1: Efficiency
    eff_stats = efficiency_analysis(images, datasets)

    # Prompt 2: Failure analysis
    fail_stats = failure_analysis(images, datasets)

    # Multi-image supplementary
    multi_image_analysis(multi_images)

    # Save raw numbers
    output = {
        "efficiency": eff_stats,
        "failure_analysis": fail_stats,
    }
    out_path = os.path.join(BASE, "results", "paper_efficiency_failure.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nRaw numbers saved to: {out_path}")


if __name__ == "__main__":
    main()
