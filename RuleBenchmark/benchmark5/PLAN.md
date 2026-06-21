# Plan: TopoBenchmark — Lookup-Based Evaluation with LODO

## Context & Motivation

TopoBenchmark evaluates TopoAgent's descriptor selection: given a medical image dataset, select the best TDA descriptor from 15 options.

**Key realization**: Benchmark4 is 97.7% complete (381/390 pairs, 26 datasets × 15 descriptors × 6 classifiers). This means we already have ground truth accuracy for nearly every (dataset, descriptor) pair. The benchmark runner should be a **lookup table**, not a live pipeline.

**Two research questions**:
- **Experiment A** (zero-shot): Can LLMs reason about descriptor selection from static knowledge alone?
- **Experiment B** (skills + LODO): Does empirical data from prior datasets improve selection?

## Ground Truth Data (Benchmark4)

**Source**: `results/benchmark4/summary/benchmark4_all_results.csv` (381 rows)
- Columns: dataset, descriptor, object_type, color_mode, {6 classifiers}\_mean/std, best_classifier, best_accuracy
- User chose: **best classifier per pair** for ground truth accuracy

**Missing benchmark4 entries (9 total)**:
- ATOL missing for: APTOS2019, Chaoyang, ISIC2019, Kvasir
- persistence_codebook missing for: APTOS2019, BreakHis, ISIC2019, Kvasir
- minkowski_functionals missing for: ISIC2019
- **Recommendation**: Submit 9 eval jobs to complete the matrix (PH cache likely exists already)

## Descriptor Taxonomy

**Training-free (12)**: persistence_image, persistence_landscapes, betti_curves, persistence_silhouette, persistence_entropy, persistence_statistics, tropical_coordinates, template_functions, minkowski_functionals, euler_characteristic_curve, euler_characteristic_transform, edge_histogram, lbp_texture

**Training-based (3)**: ATOL (k-means on diagrams), persistence_codebook (dictionary learning), ~~template_functions~~ (fixed tent/Gaussian basis — training-free)

Note: template_functions evaluates diagram points against a **fixed grid** of tent/Gaussian basis functions. Grid centers are determined by the diagram's birth-death range, not learned from data. This is normalization, not learning. It is training-free.

For paper reporting: primary table with all 15 descriptors, secondary analysis restricting to the 12 training-free descriptors only (removes ATOL, persistence_codebook concern about inflated scores with single-image user mode).

## Architecture: Pre-compute + Lookup

```
Phase 1 (DONE): Benchmark4 → 381 accuracy values
Phase 2 (DONE): Exp4 → optimal parameters (fixed)
Phase 3 (DONE): precompute_benchmark_assets.py → all derived artifacts
Phase 4 (DONE): benchmark_runner.py (lookup mode) → selection + lookup
```

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `scripts/precompute_benchmark_assets.py` | **NEW** | Generate all pre-computed artifacts (9 outputs) |
| `topoagent/benchmark_runner.py` | **Rewritten** | Lookup mode, 6 baselines, LODO, gap metrics |
| `topoagent/prompts.py` | Modified | Added `{cheap_features}` to 2 prompts |
| `scripts/test_protocol2_small.py` | Modified | Added --experiment, --lookup flags |

## All Methods (10 total)

| Method | Type | LLM? | Description |
|--------|------|------|-------------|
| `oracle` | Upper bound | No | Best descriptor per dataset (from benchmark4) |
| `topoagent` | Our method | Yes | Full skill knowledge + learned rules + cheap features |
| `topoagent_no_skills` | Ablation | Yes | LLM + sample stats + cheap features, no expert rules |
| `object_type_default` | Baseline | No | Best descriptor for the object type (between SBA and oracle) |
| `meta_learner` | Baseline | No | GBR trained on 25 cheap features, LODO predictions |
| `meta_learner_ridge` | Baseline | No | RidgeCV sanity check (simpler model, same features) |
| `rule_based` | Baseline | No | 4 hand-crafted rules from exp7 decision tree |
| `sba` | Baseline | No | Single best algorithm (persistence_statistics, full coverage) |
| `fewshot` | LLM baseline | Yes | GPT-4o with 3 examples from other datasets |
| `zeroshot` | LLM baseline | Yes | GPT-4o zero-shot (no skill knowledge) |
| `random` | Lower bound | No | Random descriptor selection |

### Baseline hierarchy (expected ordering):
```
oracle > topoagent > object_type_default ≈ meta_learner > rule_based > sba > zeroshot > random
```

`object_type_default` fills the gap between SBA (one descriptor for all) and oracle (best per dataset) — it knows the object type and picks the best descriptor for that type.

`meta_learner_ridge` is a sanity check: if Ridge agrees with GBR on most datasets, GBR isn't overfitting. If they diverge, that's a red flag worth investigating.

## Pre-computed Assets (9 files)

`results/topobenchmark/assets/`:

1. **`accuracy_lookup.json`**: `{dataset: {descriptor: best_accuracy}}` from benchmark4 CSV
2. **`oracle.json`**: `{dataset: {descriptor, accuracy}}` — best descriptor per dataset
3. **`sba.json`**: `{descriptor, mean_accuracy, n_datasets, per_dataset}`
4. **`top_performers.json`**: Rankings per object type from benchmark4
5. **`top_performers_lodo/`**: 26 JSON files, one per held-out dataset
6. **`cheap_features.json`**: `{dataset: {25 feature values}}`
7. **`meta_learner_lodo.json`**: GBR LODO predictions per dataset
8. **`meta_learner_ridge_lodo.json`**: RidgeCV LODO predictions (sanity check)
9. **`rule_based_predictions.json`**: Rule-based predictions per dataset

## Key Implementation Details

### Gap Closed Metric

```
gap_closed = (method_mean - sba_mean) / (oracle_mean - sba_mean) * 100
```

**Edge case guard**: Only computed when `(oracle_mean - sba_mean) > 0.005` (>0.5 percentage points). Returns `None` otherwise — avoids division by near-zero when oracle and SBA happen to be close.

### object_type_default Baseline

```python
def _select_object_type_default(self, context):
    object_type = context["object_type"]
    active_tp = self._get_active_top_performers()
    return active_tp[object_type][0]  # top-ranked descriptor for this object type
```

Uses LODO-masked rankings when LODO is active (same as topoagent).

### Meta-learner: GBR + RidgeCV

Both models trained on same data: `(dataset_features[25], descriptor_id) → accuracy`.

- **GBR**: `GradientBoostingRegressor(n_estimators=100, max_depth=3, lr=0.05)` — ~375 training samples per LODO fold
- **RidgeCV**: `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])` — linear, much simpler, less overfit-prone

During precomputation, GBR vs RidgeCV agreement rate is printed. If agreement is high (>80%), both models find similar signal; if low, GBR may be fitting noise.

### Rule-based Baseline Note

The 4 rules were derived from exp7's 13-dataset decision tree. We now evaluate on 26 datasets — the rules may not generalize to the 13 new datasets. This is fine for fairness (rules weren't tuned on test data) but worth noting in the paper.

## LODO Protocol

| Method | What's masked in LODO? | How? |
|--------|----------------------|------|
| topoagent | TOP_PERFORMERS rankings | Load `top_performers_lodo/{dataset}.json` |
| object_type_default | TOP_PERFORMERS rankings | Same LODO rankings |
| meta_learner | GBR training data | Pre-computed LODO prediction in `meta_learner_lodo.json` |
| meta_learner_ridge | Ridge training data | Pre-computed LODO prediction in `meta_learner_ridge_lodo.json` |
| rule_based | Nothing (rules are hand-crafted) | N/A |
| sba | Nothing (fixed strategy) | N/A |
| zeroshot | Nothing (no training data) | N/A |
| fewshot | Examples exclude test dataset | Already built-in |
| topoagent_no_skills | Nothing (no rankings) | N/A |

## Expected Output (Lookup Mode)

```
================================================================================
BENCHMARK MODE — Experiment A (26 datasets, lookup mode)
================================================================================
Method                     | Blood | Path  | Brain | ...  |   Avg | Regret | % Gap
---------------------------|-------|-------|-------|------|-------|--------|------
Oracle (upper bound)       | 98.0% | 97.5% | 96.1% | ...  | 80.2% |  0.000 |   --
TopoAgent (ours)           | 97.6% | 97.5% | 96.1% | ...  | 78.8% |  0.014 | 64.3%
ObjType default            | 97.6% | 97.5% | 96.1% | ...  | 78.0% |  0.022 | 50.0%
Meta-learner (GBR)         | 97.6% | 97.5% | 96.1% | ...  | 77.5% |  0.027 | 28.6%
Meta-learner (Ridge)       | ...   | ...   | ...   | ...  |  ...  |  ...   |  ...
Rule-based (4 rules)       | 94.9% | 74.8% | 96.1% | ...  | 76.1% |  0.041 |  0.0%
SBA (persist_stats)        | 92.2% | 97.5% | 93.4% | ...  | 74.8% |  0.054 |  0.0%
GPT-4o (zero-shot)         | 97.6% | 62.4% | 96.1% | ...  | 73.2% |  0.070 | -21.4%
Random (lower bound)       | 64.3% | 71.2% | 78.4% | ...  | 66.8% |  0.134 |   --

Evaluation time: ~30 seconds (lookup) vs ~hours (live)
```

## 3-Tier Evaluation Plan

### Tier 1: Selection Evaluation (main results)

**Question**: Which method selects the best descriptor?

- **Scope**: 26 datasets, all 10+ methods, lookup mode
- **Metrics**: mean regret, % gap closed, top-1 match rate
- **LODO**: Applied for methods using benchmark4 data (topoagent, object_type_default, meta_learner, fewshot)
- **Cost**: ~5 min (lookup mode)
- **Paper output**:
  - Table 1: Main results matrix (methods × datasets)
  - Table 2: Per-object-type breakdown (important: method rankings may flip across types, e.g., vessel_trees is much harder than glands_lumens)
  - Appendix: Feature importance analysis (top 10 GBR features + category breakdown)

### Tier 2: Full Agent Evaluation (agent justification)

**Question**: Does the full agent loop (select → verify → reflect) improve over single-call selection?

- **Scope**: 8 datasets (cover all 5 object types; include 2-3 high-regret datasets from Tier 1), 3 methods
- **Methods**:
  - `topoagent_full`: Full v5 workflow — select → extract features → verify quality → optionally switch descriptor → reflect
  - `topoagent_single`: Single-call selection (same as Tier 1's topoagent method)
  - `zeroshot`: Baseline LLM
- **Mode**: Hybrid — live context (loads images, computes PH, extracts features so agent can assess quality) but final accuracy from benchmark4 lookup
- **Repetitions**: 3 per method (captures LLM variance at temp=0.3)
- **Cost**: ~2-3 hours (8 datasets × 3 methods × 3 reps, each needs PH + feature extraction for context)
- **Key metrics**:
  - Accuracy (looked up from benchmark4 based on final descriptor choice)
  - **Correction rate**: How often does `topoagent_full` change its descriptor after verification?
  - Accuracy delta from corrections (did changes help or hurt?)
- **Paper output**:
  - Table 3: Does reflection help? (initial pick vs final pick, correction rate, accuracy delta)

  ```
  Dataset      | Initial pick  | Final pick    | Changed? | Acc delta
  BloodMNIST   | ATOL          | ATOL          | No       | 0
  RetinaMNIST  | ATOL          | persist_stats | Yes      | +0.12
  ```

- **Dataset selection criteria**: Choose 8 from Tier 1 results:
  - 2 "easy" (high oracle accuracy, most methods agree)
  - 2-3 "hard" (low oracle or high regret for topoagent — room for reflection to help)
  - Cover all 5 object types

### Tier 3: Case Studies (qualitative)

**Question**: What does the agent's reasoning look like?

- **Scope**: 2 datasets from Tier 2 (1 where correction helped, 1 where first pick was correct)
- **Content**: Full agent trace with reasoning chain — shows the selection rationale, feature quality assessment, and (if applicable) the correction decision
- **Cost**: Free (logs from Tier 2 runs)
- **Paper output**:
  - Figure: Agent trace diagram showing the reasoning flow
  - Shows: successful correction case + confident first-pick case

### Tier Dependencies

```
Tier 1 (lookup, ~5 min)
  ↓ identifies high-regret datasets
Tier 2 (hybrid live+lookup, ~2-3 hours)
  ↓ selects interesting cases
Tier 3 (free, from Tier 2 logs)
```

## Verification

1. **Pre-compute**: `python scripts/precompute_benchmark_assets.py` — generates all 9+ assets
2. **Asset check**: Verify `results/topobenchmark/assets/` has all expected files (including `meta_learner_ridge_lodo.json`, `feature_importance.json`)
3. **Import test**: `python -c "from topoagent.benchmark_runner import TopoBenchmarkRunner, NO_LLM_METHODS; print(sorted(NO_LLM_METHODS))"`
4. **Smoke test (lookup, Exp A)**: `python scripts/test_protocol2_small.py --mode benchmark --lookup --experiment A --dataset BloodMNIST,PathMNIST --methods topoagent,object_type_default,meta_learner,meta_learner_ridge,rule_based,sba,oracle`
5. **Smoke test (lookup, Exp B)**: Same with `--experiment B` — verify LODO applies
6. **Verify metrics**: % Gap Closed column present; SBA shows 0.0%; near-zero guard works
7. **Live mode still works**: `python scripts/test_protocol2_small.py --mode benchmark --dataset BloodMNIST --methods oracle --n-eval 100 --cv-folds 3`
8. **GBR vs Ridge**: Check agreement rate in precompute output — expect >70% agreement
