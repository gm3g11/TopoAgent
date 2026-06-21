# TopoAgent Descriptor Selection Rules

Rules for selecting the optimal TDA descriptor based on observable image characteristics.
Derived from Benchmark 3 results across 13 medical imaging datasets and 13 TDA descriptors.

**Adjustment applied:** ATOL and persistence_codebook scores penalized by 2% to account for data leakage (global KMeans fitting).

---

## Quantitative Rule Summary (Cascading)

Rules are evaluated in order. The first matching rule determines the descriptor.

| # | Rule | Condition | Descriptor | Train Acc |
|---|------|-----------|------------|-----------|
| 1 | Discrete objects | `fg_bg_contrast >= 0.327 AND intensity_skewness < -0.628` | **betti_curves** | 100% |
| 2 | Diffuse/scattered | `largest_component_ratio >= 0.978` | **persistence_statistics** | 100% |
| 3 | Multi-scale/dense | `intensity_mean < 0.196` | **template_functions** | 90% |
| 4 | Default | (none) | **ATOL** | — |

---

## Rule 1: Many Discrete Objects -> **betti_curves**

**Quantitative thresholds:**
- `fg_bg_contrast >= 0.327` — Strong separation between foreground and background
- `intensity_skewness < -0.628` — Negative skewness (bright objects on dark bg)
- Both conditions must hold (AND)

**Feature values for validated datasets:**

| Dataset | fg_bg_contrast | intensity_skewness | Winner gap |
|---------|----------------|--------------------|------------|
| BloodMNIST | 0.456 | -0.978 | 0.0002 |
| MalariaCell | 0.629 | -0.844 | 0.0002 |

**Boundary cases (excluded by thresholds):**

| Dataset | fg_bg_contrast | intensity_skewness | Why excluded |
|---------|----------------|--------------------|--------------|
| ISIC2019 | 0.324 | -1.394 | fg_bg_contrast < 0.327 |
| BreakHis | 0.199 | -0.412 | fg_bg_contrast < 0.327 |

**Observable signals:**
- After Otsu thresholding: many separate connected components
- Clear foreground/background separation (high contrast)
- Objects have similar sizes and don't touch each other
- Binary image shows "dots" or "blobs" pattern

**Reasoning:**
Betti curves count connected components (beta_0) and holes (beta_1) at each threshold. When images contain many discrete objects (like blood cells), the beta_0 curve shape directly encodes count and size distribution.

**Example images:** Blood cells, parasites, cell nuclei

---

## Rule 2: Diffuse/Scattered Patterns -> **persistence_statistics**

**Quantitative threshold:**
- `largest_component_ratio >= 0.978` — One dominant connected component filling nearly the entire image

**Feature values for validated datasets:**

| Dataset | largest_component_ratio | gradient_mean | Winner gap |
|---------|------------------------|---------------|------------|
| RetinaMNIST | 0.990 | 0.011 | 0.0230 |

**Boundary cases:**

| Dataset | largest_component_ratio | Why excluded |
|---------|------------------------|--------------|
| BloodMNIST | 0.975 | Captured by Rule 1 first |
| MalariaCell | 1.000 | Captured by Rule 1 first |
| Kvasir | 0.966 | Below threshold |

**Observable signals:**
- Many small, scattered features throughout the image (e.g., microaneurysms, hemorrhages)
- No dominant structures -- features are spread diffusely
- Low gradient mean (diffuse, not sharp boundaries)
- Classification depends on overall density/severity rather than specific feature shapes

**Reasoning:**
When images contain many small scattered features, individual persistence points are noisy and unreliable. Persistence statistics capture global summaries: mean lifetime reflects average feature size, sum reflects total topological activity, entropy reflects diversity.

**Example images:** Retinal fundus with vascular damage, images requiring severity grading

---

## Rule 3: Multi-Scale Structures -> **template_functions**

**Quantitative threshold:**
- `intensity_mean < 0.196` — Dark images (low mean intensity, normalized 0-1)

**Feature values for validated datasets:**

| Dataset | intensity_mean | intensity_skewness | Winner gap |
|---------|----------------|--------------------|------------|
| TissueMNIST | 0.108 | 2.188 | 0.0024 |
| BrainTumorMRI | 0.185 | 1.283 | 0.0112 |
| OCTMNIST | 0.195 | 2.354 | 0.0342 |

**Near misses (above threshold, template_functions still wins):**

| Dataset | intensity_mean | Actual winner |
|---------|----------------|---------------|
| Kvasir | 0.388 | template_functions |
| NCT_CRC_HE | 0.622 | persistence_codebook (adj.) |

**Observable signals:**
- Image shows gradual intensity transitions (not binary)
- Positive intensity skewness (dark background with bright features)
- Multiple structural scales visible (fine texture + larger patterns)
- Layered or organized structures at different scales

**Reasoning:**
Template functions place fixed "tent" kernels across the birth-death plane, capturing features at multiple scales simultaneously. Dark images with positive skewness (common in microscopy, OCT, MRI) tend to have rich multi-scale topology that template functions handle well.

---

## Rule 4: Default -> **ATOL**

**When to apply:**
- No specific rule triggered (rules 1-3 don't match)
- Typically: moderate-to-high intensity, moderate contrast, no extreme skewness

**Datasets falling to default:**

| Dataset | intensity_mean | fg_bg_contrast | Adj. winner | ATOL regret |
|---------|----------------|----------------|-------------|-------------|
| BreakHis | 0.706 | 0.199 | ATOL | 0.000 |
| ISIC2019 | 0.556 | 0.324 | ATOL | 0.000 |
| MURA | 0.196 | 0.286 | ATOL | 0.000 |
| OrganAMNIST | 0.489 | 0.451 | ATOL | 0.000 |
| PathMNIST | 0.617 | 0.270 | ATOL | 0.000 |
| Kvasir | 0.388 | 0.456 | template_functions | 0.0445 |
| NCT_CRC_HE | 0.622 | 0.273 | persistence_codebook | 0.0040 |

**Reasoning:**
ATOL learns cluster centers where persistence features concentrate in the data. For images without a clear structural signal (neither discrete objects, nor diffuse patterns, nor dark multi-scale), ATOL's adaptive learning finds dataset-specific topological signatures.

---

## LODO Evaluation Results

Leave-One-Dataset-Out cross-validation: for each dataset, thresholds are re-derived from the remaining 12 datasets, then the held-out dataset is classified.

### Per-Dataset Results

| Dataset | Predicted | Oracle | Regret | ATOL Regret | TF Regret |
|---------|-----------|--------|--------|-------------|-----------|
| BloodMNIST | ATOL | betti_curves | 0.0194 | 0.0194 | 0.0002 |
| BrainTumorMRI | ATOL | template_functions | 0.0112 | 0.0112 | 0.0000 |
| BreakHis | ATOL | ATOL | **0.0000** | 0.0000 | 0.0667 |
| ISIC2019 | betti_curves | ATOL | 0.0859 | 0.0000 | 0.0354 |
| Kvasir | persistence_statistics | template_functions | 0.0295 | 0.0445 | 0.0000 |
| MURA | template_functions | ATOL | 0.0190 | 0.0000 | 0.0190 |
| MalariaCell | betti_curves | betti_curves | **0.0000** | 0.0148 | 0.0006 |
| NCT_CRC_HE | ATOL | persistence_codebook | 0.0040 | 0.0040 | 0.0038 |
| OCTMNIST | persistence_statistics | template_functions | 0.0834 | 0.0406 | 0.0000 |
| OrganAMNIST | ATOL | ATOL | **0.0000** | 0.0000 | 0.0518 |
| PathMNIST | template_functions | ATOL | 0.0036 | 0.0000 | 0.0036 |
| RetinaMNIST | betti_curves | persistence_statistics | 0.1022 | 0.0732 | 0.0230 |
| TissueMNIST | persistence_statistics | template_functions | 0.0436 | 0.0036 | 0.0000 |

### Aggregate Metrics

| Strategy | Mean Regret | Top-1 | Soft Acc (regret < 1%) |
|----------|-------------|-------|------------------------|
| **rule_based (LODO)** | 0.0309 | 3/13 | 5/13 |
| always_template_functions | **0.0157** | 4/13 | **8/13** |
| always_ATOL | 0.0163 | **5/13** | 7/13 |
| always_persistence_codebook | 0.0275 | 1/13 | 3/13 |
| always_persistence_statistics | 0.0340 | 1/13 | 2/13 |
| random (uniform) | 0.1173 | 0/13 | 0/13 |

### Key Finding

**The rule-based selector does not beat the best fixed strategies** under LODO evaluation. With only 13 datasets:
- Rule thresholds are unstable across folds (removing 1 of 13 datasets shifts thresholds)
- Groups with 1-2 members (persistence_statistics, betti_curves) lose their signal when the member is held out
- The gap between top descriptors is often < 1%, making correct selection nearly impossible

**Practical recommendation:** Use **template_functions** as the default descriptor. It achieves the lowest mean regret (0.0157) and highest soft accuracy (8/13 datasets within 1% of oracle). Use betti_curves only when discrete objects are clearly visible.

---

## Adjusted Winners (with 2% leakage penalty)

| Dataset | Adjusted Winner | Gap to Runner-Up | Previous Winner (raw) |
|---------|-----------------|-------------------|-----------------------|
| BloodMNIST | betti_curves | 0.0002 | persistence_codebook |
| MalariaCell | betti_curves | 0.0002 | ATOL |
| RetinaMNIST | persistence_statistics | 0.0230 | persistence_statistics |
| BrainTumorMRI | template_functions | 0.0112 | ATOL |
| OCTMNIST | template_functions | 0.0342 | template_functions |
| Kvasir | template_functions | 0.0295 | template_functions |
| TissueMNIST | template_functions | 0.0024 | persistence_codebook |
| ISIC2019 | ATOL | 0.0036 | ATOL |
| MURA | ATOL | 0.0190 | ATOL |
| BreakHis | ATOL | 0.0184 | ATOL |
| OrganAMNIST | ATOL | 0.0156 | ATOL |
| PathMNIST | ATOL | 0.0010 | ATOL |
| NCT_CRC_HE | persistence_codebook | 0.0038 | persistence_codebook |

**Distribution:** ATOL (5), template_functions (4), betti_curves (2), persistence_codebook (1), persistence_statistics (1)

**Note:** 6 of 13 datasets have winner gap < 1% -- the "winner" label is fragile.

---

## Feature Discriminability

Top discriminative features per group (based on single-feature classification accuracy):

### betti_curves group (BloodMNIST, MalariaCell)

| Feature | Accuracy | Threshold | Direction | Group Mean | Other Mean |
|---------|----------|-----------|-----------|------------|------------|
| intensity_skewness | 0.923 | -0.628 | < | -0.911 | 0.520 |
| intensity_kurtosis | 0.923 | -1.070 | < | -0.676 | 1.261 |
| fg_bg_contrast | 0.923 | 0.327* | >= | 0.543 | 0.321 |

### persistence_statistics group (RetinaMNIST)

| Feature | Accuracy | Threshold | Direction | Group Mean | Other Mean |
|---------|----------|-----------|-----------|------------|------------|
| gradient_mean | 0.923 | 0.011 | < | 0.011 | 0.029 |
| largest_component_ratio | 0.923 | 0.978 | >= | 0.990 | 0.799 |
| otsu_stability | 0.923 | 0.541 | >= | 0.553 | 0.498 |

### template_functions group (BrainTumorMRI, Kvasir, OCTMNIST, TissueMNIST)

| Feature | Accuracy | Threshold | Direction | Group Mean | Other Mean |
|---------|----------|-----------|-----------|------------|------------|
| intensity_mean | 0.923 | 0.196 | < | 0.219 | 0.512 |
| intensity_skewness | 0.846 | 0.846 | >= | 1.436 | -0.205 |
| intensity_kurtosis | 0.846 | 1.571 | >= | 3.032 | 0.043 |

---

## Quick Reference Table

| Observable Signal | Descriptor | Quantitative Threshold |
|-------------------|------------|----------------------|
| Discrete blobs, high contrast, negative skewness | **betti_curves** | fg_bg >= 0.327 AND skew < -0.628 |
| Diffuse features, dominant component | **persistence_statistics** | largest_comp_ratio >= 0.978 |
| Dark images, positive skewness | **template_functions** | intensity_mean < 0.196 |
| None of the above | **ATOL** | Default |

---

## Usage Example

**User input:** "Analyze this blood smear image for malaria detection"

**Agent reasoning:**
> Observing this blood smear image:
> - fg_bg_contrast = 0.63 (>= 0.327)
> - intensity_skewness = -0.84 (< -0.628)
> - Rule 1 triggered: Discrete objects
>
> **Recommendation: betti_curves**
>
> Reason: The image contains many discrete cell objects with high foreground-background
> contrast and negative intensity skewness. Betti curves will capture the count and
> size distribution of connected components.

---

## Notes

1. **Rules are derived from 13 datasets** -- may need refinement for novel modalities
2. **ATOL and persistence_codebook have 2% leakage penalty** applied -- uses global KMeans fitting
3. **Rules are based on computable image features** -- no prior knowledge of dataset metadata required
4. **6 of 13 datasets have < 1% gap** between top descriptors -- winner selection is inherently noisy
5. **template_functions is the safest single choice** (mean regret 0.0157, lowest of all fixed strategies)
6. **LODO evaluation shows rules don't beat fixed strategies** on 13 datasets -- too few datasets for robust threshold learning
7. **Code:** `scripts/run_benchmark3/exp7_meta_learning/rule_based_selector.py`

---

*Generated from Benchmark 3 results (13 datasets, 13 TDA descriptors, TabPFN classifier)*
*Leakage-adjusted scores (2% penalty on ATOL, persistence_codebook)*
*LODO evaluation: 13-fold leave-one-dataset-out*
*Last updated: 2026-02-02*
