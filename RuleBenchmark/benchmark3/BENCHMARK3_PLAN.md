# Benchmark3 Plan

**Goal:** Refine benchmark2 with (1) updated/increased dimensions for existing descriptors, (2) 7 new descriptors (19 total), (3) 5 new datasets (13 total).

---

## Changes from Benchmark2

### Descriptor Changes Summary

| Change | Descriptors |
|--------|-------------|
| **Removed** | carlsson_coordinates (8D, low performance in benchmark2) |
| **Added (7)** | template_functions, complex_polynomial, heat_kernel, topological_vector, persistence_lengths, euler_characteristic_transform, edge_histogram |
| **Dim increased** | persistence_image (98→800), persistence_landscapes (96→800), persistence_silhouette (100→200), betti_curves (100→200), euler_characteristic_curve (50→100), minkowski_functionals (6→30) |
| **Dim decreased** | persistence_statistics (42→24), tropical_coordinates (40→14), persistence_codebook (100→64), lbp_texture (59→54) |
| **Unchanged** | persistence_entropy (2D), ATOL (32D) |

**Rationale for removing 100D cap:** TabPFN 2.5 supports up to 1000D features. All descriptors now use their natural/recommended dimensions.

**Note on persistence_entropy:** giotto-tda's `PersistenceEntropy` returns a single scalar per homology dimension (total 2D). This is NOT a curve. A custom "entropy curve" (entropy at each filtration threshold) would give 200D but requires custom implementation — not included in this benchmark.

### Why carlsson_coordinates was removed
- Only 8D, limited expressiveness
- Never won any dataset in benchmark2 Exp1
- Bottom-tier accuracy across all classifiers

---

## Full Descriptor List (19 total)

### PH Vectorizations (15)

| # | Descriptor | Dim | Source | Benchmark2 Dim | Change |
|---|-----------|-----|--------|----------------|--------|
| 1 | persistence_statistics | 24D | Custom | 42D | Reduced (12 stats × 2 hom) |
| 2 | persistence_image | 800D | gtda PersistenceImage | 98D | **8x increase** (2×20×20) |
| 3 | persistence_landscapes | 800D | gtda PersistenceLandscape | 96D | **8x increase** (2×4×100) |
| 4 | persistence_silhouette | 200D | gtda Silhouette | 100D | **2x increase** (2×100) |
| 5 | betti_curves | 200D | gtda BettiCurve | 100D | **2x increase** (2×100) |
| 6 | persistence_entropy | 2D | gtda PersistenceEntropy | 2D | Same (scalar per dim) |
| 7 | ATOL | 32D | GUDHI Atol | 32D | Same (16 clusters × 2) |
| 8 | template_functions | 50D | pervect/TDAvec | NEW | 25 tents × 2 hom |
| 9 | tropical_coordinates | 14D | TDAvec/custom | 40D | Corrected (7 functions × 2) |
| 10 | complex_polynomial | 40D | gtda ComplexPolynomial | NEW | 10 coeffs × 2 (real+imag) × 2 hom |
| 11 | heat_kernel | 800D | gtda HeatKernel | NEW | 2×20×20 (n_bins=20!) |
| 12 | euler_characteristic_curve | 100D | Custom | 50D | **2x increase** |
| 13 | topological_vector | 50D | GUDHI TopologicalVector | NEW | 25 distances × 2 hom |
| 14 | persistence_lengths | 20D | GUDHI PersistenceLengths | NEW | 10 lengths × 2 hom |
| 15 | persistence_codebook | 64D | Custom (KMeans BoW) | 100D | Adjusted (32 words × 2) |

### Beyond-PH Topology (2)

| # | Descriptor | Dim | Source | Benchmark2 Dim | Change |
|---|-----------|-----|--------|----------------|--------|
| 16 | minkowski_functionals | 30D | quantimpy | 6D | **5x increase** (3 funcs × 10 thresholds) |
| 17 | euler_characteristic_transform | 640D | demeter/dect | NEW | 32 directions × 20 heights |

### Non-TDA Baselines (2)

| # | Descriptor | Dim | Source | Benchmark2 Dim | Change |
|---|-----------|-----|--------|----------------|--------|
| 18 | lbp_texture | 54D | skimage | 59D | Multi-scale uniform LBP (radii 1,2,3) |
| 19 | edge_histogram | 80D | Custom/OpenCV | NEW | 8 orientations × 10 cells |

---

## Parameter Configs (Benchmark3) — CORRECTED

See `config.py` for the authoritative implementation. Key corrections from initial plan:

| Descriptor | Initial Plan | Corrected | Issue |
|------------|-------------|-----------|-------|
| persistence_landscapes | 5×40×2 = 400D | 4×100×2 = **800D** | n_layers=4, n_bins=100 |
| persistence_entropy | 100×2 = 200D (curve) | **2D** (scalar) | giotto-tda returns 1 value per dim |
| tropical_coordinates | 10×2 = 20D | 7×2 = **14D** | Standard 7 tropical functions |
| heat_kernel | n_bins=100 (20000D!) | n_bins=20 (**800D**) | Must match PI resolution |
| complex_polynomial | polynomial_type='T' | polynomial_type='R' | R transformation standard |
| topological_vector | threshold=10 (20D) | threshold=25 (**50D**) | 25 per dim |

```python
# Authoritative configs (from config.py):
DESCRIPTOR_CONFIGS = {
    'persistence_statistics': {'stats': 12_per_dim},            # 24D
    'persistence_image': {'sigma': 0.1, 'n_bins': 20},          # 800D (2×20×20)
    'persistence_landscapes': {'n_layers': 4, 'n_bins': 100},   # 800D (2×4×100)
    'persistence_silhouette': {'n_bins': 100, 'power': 1.0},    # 200D (2×100)
    'betti_curves': {'n_bins': 100},                             # 200D (2×100)
    'persistence_entropy': {'normalize': True},                  # 2D (scalar!)
    'atol': {'n_clusters': 16},                                  # 32D (16×2)
    'template_functions': {'n_templates': 25, 'type': 'tent'},   # 50D (25×2)
    'tropical_coordinates': {'n_functions': 7},                  # 14D (7×2)
    'complex_polynomial': {'n_coefficients': 10, 'type': 'R'},   # 40D
    'heat_kernel': {'sigma': 0.1, 'n_bins': 20},                 # 800D (2×20×20)
    'euler_characteristic_curve': {'n_bins': 100},               # 100D
    'topological_vector': {'threshold': 25},                     # 50D (25×2)
    'persistence_lengths': {'num_lengths': 10},                  # 20D (10×2)
    'persistence_codebook': {'n_codewords': 32},                 # 64D (32×2)
    'minkowski_functionals': {'n_thresholds': 10},               # 30D (3×10)
    'euler_characteristic_transform': {'n_dir': 32, 'n_h': 20}, # 640D (32×20)
    'lbp_texture': {'radii': [1,2,3], 'method': 'uniform'},     # 54D
    'edge_histogram': {'n_orient': 8, 'n_cells': 10},           # 80D
}
```

---

## Dimension Distribution

```
<50D:    persistence_entropy(2), tropical_coordinates(14), persistence_lengths(20),
         persistence_statistics(24), minkowski_functionals(30), atol(32),
         complex_polynomial(40)

50-100D: template_functions(50), topological_vector(50), lbp_texture(54),
         persistence_codebook(64), edge_histogram(80), euler_characteristic_curve(100)

100-500D: persistence_silhouette(200), betti_curves(200)

500-1000D: euler_characteristic_transform(640),
           persistence_image(800), heat_kernel(800), persistence_landscapes(800)
```

Total across all 19 descriptors: 4,000D
All individual descriptors within TabPFN 2.5's 1000D limit.

---

## Datasets (13 total)

### From Benchmark2 (8)
| # | Dataset | Modality | Classes | Type | Notes |
|---|---------|----------|---------|------|-------|
| 1 | BloodMNIST | Microscopy | 8 | MedMNIST | Blood cell types |
| 2 | TissueMNIST | Microscopy | 8 | MedMNIST | Kidney cortex cells |
| 3 | PathMNIST | Histopathology | 9 | MedMNIST | Colorectal cancer tissue |
| 4 | OCTMNIST | OCT | 4 | MedMNIST | Retinal OCT layers |
| 5 | OrganAMNIST | CT (axial) | 11 | MedMNIST | Abdominal organs (needs ECOC) |
| 6 | RetinaMNIST | Fundus | 5 | MedMNIST | Diabetic retinopathy |
| 7 | ISIC2019 | Dermoscopy | 8 | External | Skin lesion classification |
| 8 | Kvasir | Endoscopy | 8 | External | GI tract findings |

### New Datasets (5)
| # | Dataset | Modality | Classes | Source | Notes |
|---|---------|----------|---------|--------|-------|
| 9 | BrainTumorMRI | MRI | 4 | Kaggle | Glioma/meningioma/pituitary/no tumor |
| 10 | MURA | X-ray | 2 | Stanford | Musculoskeletal abnormality detection |
| 11 | BreakHis | Histopathology | 2 (or 8) | External | Breast cancer histology, multi-magnification |
| 12 | NCT_CRC_HE | Histopathology | 9 | External | Colorectal cancer H&E patches (100K) |
| 13 | MalariaCell | Microscopy | 2 | NIH | Parasitized vs uninfected cells |

### Dataset Diversity Coverage
| Modality | Datasets | Count |
|----------|----------|-------|
| Microscopy | BloodMNIST, TissueMNIST, MalariaCell | 3 |
| Histopathology | PathMNIST, BreakHis, NCT_CRC_HE | 3 |
| CT | OrganAMNIST | 1 |
| OCT | OCTMNIST | 1 |
| MRI | BrainTumorMRI | 1 |
| X-ray | MURA | 1 |
| Fundus | RetinaMNIST | 1 |
| Dermoscopy | ISIC2019 | 1 |
| Endoscopy | Kvasir | 1 |

### New Dataset Details

**BrainTumorMRI** (Kaggle)
- ~7000 images, 4 classes (glioma, meningioma, pituitary tumor, no tumor)
- T1-weighted contrast-enhanced MRI
- Expected: H1 features important (tumor boundaries, ring enhancement)

**MURA** (Stanford ML Group)
- ~40,000 X-ray images, binary (normal/abnormal)
- Upper extremity: elbow, finger, forearm, hand, humerus, shoulder, wrist
- Expected: Subtle structural changes, low TDA signal, challenging baseline

**BreakHis** (Spanhol et al. 2016)
- 7,909 images at 40x/100x/200x/400x magnification
- Binary: benign (4 subtypes) vs malignant (4 subtypes)
- Expected: Multi-scale topology important, H0+H1 synergy

**NCT_CRC_HE** (Kather et al. 2018)
- 100,000 non-overlapping patches (224x224), 9 tissue classes
- H&E stained colorectal cancer tissue
- Expected: Similar to PathMNIST but larger and more standardized

**MalariaCell** (NIH NLM)
- 27,558 cell images, binary (parasitized/uninfected)
- Thin blood smear, Giemsa-stained
- Expected: H0 dominant (parasite inclusions as connected components)

---

## Classifiers (same as Benchmark2)

| Classifier | Notes |
|------------|-------|
| KNN | k=5, distance-weighted |
| LogisticRegression | C=1.0, max_iter=1000 |
| XGBoost | GPU-first, 100 trees |
| TabPFN / TabPFN_ECOC | TabPFN 2.5, ECOC for >10 classes |

---

## Experiments Plan

### Exp1: Main Benchmark (19 descriptors x 13 datasets x 4 classifiers)
- N_SAMPLES = 5000
- 5-fold CV
- Report: accuracy, balanced accuracy, per-class F1
- Compare to benchmark2 baselines (8 overlapping datasets)

### Exp2: Dimension Ablation
- For descriptors with increased dims: compare benchmark2 dims vs benchmark3 dims
- Answer: does higher dimensionality actually help?

### Exp3: New Descriptor Analysis
- Focus on the 6 new descriptors
- Where do they rank vs existing 13?
- Are any complementary (CKA analysis)?

### Exp4: Combined Filtration (carried over)
- Verify combined filtration still wins with higher dims

### Exp5: Ensemble with New Descriptors
- Update best ensembles with new descriptor candidates
- Does adding heat_kernel or ECT to ensembles help?

---

## Implementation Checklist

### Phase 1: Infrastructure
- [x] Create `scripts/run_benchmark3/config.py` ✅
- [ ] Implement new descriptor tools (7 new):
  - [ ] `template_functions` (pervect/TDAvec, 50D)
  - [ ] `complex_polynomial` (gtda ComplexPolynomial, 40D)
  - [ ] `heat_kernel` (gtda HeatKernel, 800D)
  - [ ] `topological_vector` (GUDHI TopologicalVector, 50D)
  - [ ] `persistence_lengths` (GUDHI PersistenceLengths, 20D)
  - [ ] `euler_characteristic_transform` (demeter/dect/custom, 640D)
  - [ ] `edge_histogram` (OpenCV/custom, 80D)
- [ ] Update existing descriptors with new dimensions:
  - [ ] persistence_image: n_bins 7→20 (98D→800D)
  - [ ] persistence_landscapes: n_layers=4, n_bins=100 (96D→800D)
  - [ ] persistence_silhouette: n_bins 50→100 (100D→200D)
  - [ ] betti_curves: n_bins 50→100 (100D→200D)
  - [ ] euler_characteristic_curve: n_bins 50→100 (50D→100D)
  - [ ] minkowski_functionals: n_thresholds=10 (6D→30D)
  - [ ] tropical_coordinates: verify 7 functions (40D→14D)
- [ ] Download/prepare 5 new datasets:
  - [ ] BrainTumorMRI (Kaggle)
  - [ ] MURA (Stanford)
  - [ ] BreakHis
  - [ ] NCT_CRC_HE
  - [ ] MalariaCell (NIH)
- [ ] Verify all 19 descriptors produce correct output dims on test images

### Phase 2: Run Experiments
- [ ] Exp1: Main benchmark (19 x N x 4)
- [ ] Exp2: Dimension ablation
- [ ] Exp3: New descriptor analysis
- [ ] Exp4: Combined filtration verification
- [ ] Exp5: Ensemble update

### Phase 3: Analysis
- [ ] Compare to benchmark2 top results
- [ ] Identify new winners per dataset
- [ ] Update descriptor taxonomy/tiers
- [ ] Generate paper figures

---

## Key Differences: Benchmark2 vs Benchmark3

| Aspect | Benchmark2 | Benchmark3 |
|--------|-----------|-----------|
| Descriptors | 13 | 19 (+7 new, -1 removed) |
| Max dimension | ~100D | 800D |
| Total feature dims | ~830D across all | 4,000D across all |
| Descriptor categories | 3 | 3 (15 PH + 2 Beyond-PH + 2 Baselines) |
| TabPFN version | TabPFN (100D cap) | TabPFN 2.5 (1000D cap) |
| Datasets | 8 | 13 (+5 new) |
| persistence_entropy | 2D (scalar) | 2D (scalar, unchanged) |
| persistence_landscapes | 96D (3×16×2) | 800D (4×100×2) |
| persistence_image | 98D (7×7×2) | 800D (20×20×2) |
| minkowski_functionals | 6D (fixed) | 30D (3×10 thresholds) |

---

## Benchmark2 Baselines to Beat

| Dataset | Best Acc (B2) | Descriptor | Classifier |
|---------|---------------|------------|------------|
| BloodMNIST | 96.18% | persistence_codebook | TabPFN |
| PathMNIST | 95.74% | persistence_codebook | TabPFN |
| OrganAMNIST | 93.10% | lbp_texture | TabPFN_ECOC |
| OCTMNIST | 72.20% | persistence_codebook | TabPFN |
| Kvasir | 69.40% | persistence_statistics | TabPFN |
| RetinaMNIST | 56.85% | lbp_texture | TabPFN |
| ISIC2019 | 45.90% | persistence_statistics | TabPFN |
| TissueMNIST | 42.78% | persistence_codebook | TabPFN |

**Hypothesis:** Higher-dim PI (800D) and persistence_landscapes (800D) should beat persistence_codebook on several datasets. ECT (640D) may dominate on shape-heavy datasets (BloodMNIST, OrganAMNIST). Heat_kernel (800D) provides complementary diffusion-based features to PI.

---

*Created: 2026-01-22*
*Status: Planning*
