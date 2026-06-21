#!/usr/bin/env python3
"""Aggregate EXP1 Stage 1 calibration cells into a comparison report.

Reads results/exp1_calibration/c0_full/{dataset}_seed{S}_T{T}.json,
groups by (dataset, T), computes mean/std of balanced_accuracy across seeds,
and compares against published Tab. 2 c0_full numbers.

Outputs:
- rebuttal/exp1_calibration_per_cell.csv       — flat (dataset, seed, T, ba)
- rebuttal/exp1_calibration_aggregated.csv     — (dataset, T, mean, std, n_seeds)
- rebuttal/exp1_calibration_report.md          — narrative with the recommendation
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Published c0_full numbers from results/ablation_study/c0_full/{dataset}.json
# (single-seed=42, T=0.7 implicit, paper-faithful baseline values).
PUBLISHED_C0_FULL = {
    "BloodMNIST":    85.08,
    "DermaMNIST":    15.08,
    "ISIC2019":      23.61,
    "OrganAMNIST":   50.52,
    "TissueMNIST":   24.07,
    "PathMNIST":     82.19,
    "OCTMNIST":      43.56,
    "RetinaMNIST":   34.67,
    "PneumoniaMNIST": 84.01,
    "BreastMNIST":   70.90,
    "OrganCMNIST":   46.40,
    "OrganSMNIST":   43.66,
    "Kvasir":        56.66,
    "BrainTumorMRI": 80.03,
    "MURA":          28.93,
    "BreakHis":      30.22,
    "NCT_CRC_HE":    81.76,
    "MalariaCell":   94.53,
    "IDRiD":         35.18,
    "PCam":          75.97,
    "LC25000":       87.00,
    "SIPaKMeD":      84.03,
    "AML_Cytomorphology": 67.40,
    "APTOS2019":     37.16,
    "GasHisSDB":     83.39,
    "Chaoyang":      59.78,
}

PUBLISHED_DESCRIPTOR = {
    "BloodMNIST":    "template_functions",
    "DermaMNIST":    "ATOL",
    "ISIC2019":      "lbp_texture",
    "OrganAMNIST":   "minkowski_functionals",
}


def load_cells(indir: Path):
    """Yield (dataset, seed, T, top_p, ba, descriptor) tuples."""
    rows = []
    for p in sorted((indir / "c0_full").glob("*_seed*_T*.json")):
        with open(p) as f:
            r = json.load(f)
        rows.append({
            "dataset": r["dataset"],
            "seed": int(r["seed"]),
            "temperature": float(r["temperature"]),
            "top_p": float(r["top_p"]),
            "ba": float(r["balanced_accuracy"]),
            "descriptor": r["majority_descriptor"],
            "fold_accuracies": r["fold_accuracies"],
            "classifier": r.get("classifier", "?"),
        })
    return rows


def aggregate(rows):
    """Group by (dataset, T) and compute mean/std/n."""
    groups = {}
    for r in rows:
        key = (r["dataset"], r["temperature"])
        groups.setdefault(key, []).append(r)

    agg = []
    for (dataset, T), cells in sorted(groups.items()):
        bas = [c["ba"] for c in cells]
        descriptors = [c["descriptor"] for c in cells]
        n = len(bas)
        mean = statistics.mean(bas)
        std = statistics.stdev(bas) if n >= 2 else 0.0
        # Mode descriptor (which one came up most)
        descriptor_counts = {}
        for d in descriptors:
            descriptor_counts[d] = descriptor_counts.get(d, 0) + 1
        mode_descriptor = max(descriptor_counts, key=descriptor_counts.get)
        n_unique_descriptors = len(set(descriptors))

        agg.append({
            "dataset": dataset,
            "temperature": T,
            "n_seeds": n,
            "mean_ba": round(mean, 2),
            "std_ba": round(std, 2),
            "min_ba": round(min(bas), 2),
            "max_ba": round(max(bas), 2),
            "mode_descriptor": mode_descriptor,
            "n_unique_descriptors": n_unique_descriptors,
            "all_descriptors": descriptor_counts,
            "published_ba": PUBLISHED_C0_FULL.get(dataset),
            "published_descriptor": PUBLISHED_DESCRIPTOR.get(dataset),
            "delta_vs_published": round(mean - PUBLISHED_C0_FULL.get(dataset, 0.0), 2)
                if dataset in PUBLISHED_C0_FULL else None,
        })
    return agg


def write_per_cell_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "temperature", "top_p", "balanced_accuracy",
                    "descriptor", "classifier", "fold1", "fold2", "fold3"])
        for r in rows:
            folds = r["fold_accuracies"] + [None] * (3 - len(r["fold_accuracies"]))
            w.writerow([r["dataset"], r["seed"], r["temperature"], r["top_p"],
                        r["ba"], r["descriptor"], r["classifier"]] + folds[:3])


def write_aggregated_csv(agg, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "temperature", "n_seeds",
                    "mean_ba", "std_ba", "min_ba", "max_ba",
                    "mode_descriptor", "n_unique_descriptors",
                    "published_ba", "published_descriptor",
                    "delta_vs_published"])
        for r in agg:
            w.writerow([r["dataset"], r["temperature"], r["n_seeds"],
                        r["mean_ba"], r["std_ba"], r["min_ba"], r["max_ba"],
                        r["mode_descriptor"], r["n_unique_descriptors"],
                        r["published_ba"], r["published_descriptor"],
                        r["delta_vs_published"]])


def write_report(agg, rows, out_path: Path):
    """Generate narrative MD report with the T=0.7 vs T=0 verdict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pivot: (dataset) x (T=0.7, T=0) -> mean ± std
    by_dataset = {}
    for r in agg:
        by_dataset.setdefault(r["dataset"], {})[r["temperature"]] = r

    def fmt(r):
        if r is None:
            return "—"
        return f"{r['mean_ba']:.2f} ± {r['std_ba']:.2f}"

    lines = []
    lines.append("# EXP1 Stage 1 — Calibration Report (T=0.7 vs T=0)")
    lines.append("")
    n_cells = len(rows)
    lines.append(f"**Cells completed:** {n_cells} / 40 expected")
    lines.append("**Setup:** c0_full only; n_demo=5, n_eval=200; seeds = 42, 123, 456, 789, 999.")
    lines.append("**Goal:** which T setting reproduces the published Tab. 2 numbers within ~1 pp?")
    lines.append("")
    lines.append("## Per-cell summary")
    lines.append("")
    lines.append("| Dataset | Published (T=0.7, seed=42) | Re-run T=0.7, seeded | Re-run T=0, seeded | Δ(T=0.7) vs published | Δ(T=0) vs published |")
    lines.append("|---|---|---|---|---|---|")
    for dataset in sorted(by_dataset.keys()):
        cells = by_dataset[dataset]
        r07 = cells.get(0.7)
        r0  = cells.get(0.0)
        published = PUBLISHED_C0_FULL.get(dataset, None)
        d07 = r07["delta_vs_published"] if r07 else None
        d0  = r0["delta_vs_published"]  if r0 else None
        lines.append(f"| {dataset} | "
                     f"{published if published is not None else '—'} ({PUBLISHED_DESCRIPTOR.get(dataset,'?')}) | "
                     f"{fmt(r07)} ({r07['mode_descriptor'] if r07 else '?'}, "
                     f"{r07['n_unique_descriptors'] if r07 else 0} unique) | "
                     f"{fmt(r0)} ({r0['mode_descriptor'] if r0 else '?'}, "
                     f"{r0['n_unique_descriptors'] if r0 else 0} unique) | "
                     f"{d07 if d07 is not None else '—'} | "
                     f"{d0 if d0 is not None else '—'} |")
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    deltas_07 = [abs(r["delta_vs_published"]) for r in agg
                 if r["temperature"] == 0.7 and r["delta_vs_published"] is not None]
    deltas_0  = [abs(r["delta_vs_published"]) for r in agg
                 if r["temperature"] == 0.0 and r["delta_vs_published"] is not None]
    stds_07 = [r["std_ba"] for r in agg if r["temperature"] == 0.7]
    stds_0  = [r["std_ba"] for r in agg if r["temperature"] == 0.0]

    if deltas_07:
        lines.append(f"- **T=0.7**: mean |Δ vs published| = {statistics.mean(deltas_07):.2f} pp; "
                     f"max |Δ| = {max(deltas_07):.2f} pp.")
        lines.append(f"  - Per-cell std across 5 seeds: median = "
                     f"{statistics.median(stds_07):.2f} pp; max = {max(stds_07):.2f} pp.")
    if deltas_0:
        lines.append(f"- **T=0**: mean |Δ vs published| = {statistics.mean(deltas_0):.2f} pp; "
                     f"max |Δ| = {max(deltas_0):.2f} pp.")
        lines.append(f"  - Per-cell std across 5 seeds: median = "
                     f"{statistics.median(stds_0):.2f} pp; max = {max(stds_0):.2f} pp.")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if deltas_07 and deltas_0:
        if statistics.mean(deltas_07) <= statistics.mean(deltas_0):
            lines.append("**Use T=0.7 for full Stage 2** — it tracks the published numbers more closely. "
                         "Per-cell std at T=0.7 is the variance the published number is itself subject to, "
                         "which is exactly what we want to report.")
        else:
            lines.append("**Use T=0 for full Stage 2** — re-running at T=0.7 drifts further from the "
                         "published numbers than greedy decoding does. The published values are likely closer "
                         "to a greedy mode of the LLM distribution that happened at higher T by chance.")
    else:
        lines.append("(insufficient data — re-run aggregation after all 40 cells finish)")
    lines.append("")

    lines.append("## Open questions / flags")
    lines.append("")
    lines.append("- If `n_unique_descriptors > 1` per (dataset, T) in the table above, the LLM's descriptor "
                 "choice itself drifts across seeds — std_ba conflates LLM trajectory variance with downstream "
                 "classifier variance. We may want to report both std components separately in Stage 2.")
    lines.append("- If `n_unique_descriptors == 1` at T=0 (deterministic descriptor) but `> 1` at T=0.7, that "
                 "confirms the LLM is the dominant variance source in published numbers.")
    lines.append("- DermaMNIST in the published table chose ATOL with 4/5 votes (1 dissent). Watch for whether "
                 "the dissent rate changes here.")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str,
                        default=str(PROJECT_ROOT / "results" / "exp1_calibration"))
    parser.add_argument("--outdir", type=str,
                        default=str(PROJECT_ROOT / "rebuttal"))
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    rows = load_cells(indir)
    if not rows:
        print(f"No cells found under {indir / 'c0_full'}")
        return
    print(f"Loaded {len(rows)} cells from {indir / 'c0_full'}")

    agg = aggregate(rows)

    write_per_cell_csv(rows, outdir / "exp1_calibration_per_cell.csv")
    write_aggregated_csv(agg, outdir / "exp1_calibration_aggregated.csv")
    write_report(agg, rows, outdir / "exp1_calibration_report.md")

    print(f"Wrote: {outdir / 'exp1_calibration_per_cell.csv'}")
    print(f"Wrote: {outdir / 'exp1_calibration_aggregated.csv'}")
    print(f"Wrote: {outdir / 'exp1_calibration_report.md'}")


if __name__ == "__main__":
    main()
