# Experiment 4: Dimension and Color Rules

Generated from Exp4 experiments on 2026-02-04.

---

## 1. Complete Dimension Rules (All 15 Descriptors)

### 1.1 persistence_image

| Object Type | Dataset | resolution | Output Dim | Accuracy |
|-------------|---------|------------|------------|----------|
| discrete_cells | BloodMNIST | 10 | **200D** | 0.420 |
| glands_lumens | PathMNIST | 26 | **1352D** | 0.502 |
| vessel_trees | RetinaMNIST | 14 | **392D** | 0.352 |
| surface_lesions | DermaMNIST | 12 | **288D** | 0.432 |

**Rule**: `dim = resolution² × 2`
- Simple structures (cells): resolution=10-12 → 200-288D
- Complex structures (glands): resolution=20-26 → 800-1352D

---

### 1.2 persistence_landscapes

#### With TabPFN (max 2000D):
| Object Type | Dataset | n_layers | Output Dim | Accuracy |
|-------------|---------|----------|------------|----------|
| discrete_cells | BloodMNIST | 12 | **1200D** | 0.901 |
| glands_lumens | PathMNIST | 12 | **1200D** | 0.767 |
| vessel_trees | RetinaMNIST | 7 | **700D** | 0.347 |
| surface_lesions | DermaMNIST | 12 | **1200D** | 0.355 |

#### With XGBoost (no limit):
| Object Type | Dataset | n_layers | Output Dim | Accuracy |
|-------------|---------|----------|------------|----------|
| discrete_cells | BloodMNIST | 35 | **3500D** | 0.891 |
| glands_lumens | PathMNIST | 38 | **3800D** | 0.753 |
| vessel_trees | RetinaMNIST | 15 | **1500D** | 0.362 |
| surface_lesions | DermaMNIST | 28 | **2800D** | 0.374 |
| organ_shape | OrganAMNIST | 35 | **3500D** | 0.633 |

**Rule**: `dim = n_layers × 100`
- TabPFN: Use n_layers=12 (1200D) as default
- XGBoost: Use n_layers=28-38 (2800-3800D) for better accuracy
- vessel_trees: Lower dimensions work better (7-15 layers)

---

### 1.3 betti_curves

| Object Type | Dataset | n_bins | Output Dim | Accuracy |
|-------------|---------|--------|------------|----------|
| discrete_cells | BloodMNIST | 120 | **240D** | 0.949 |
| glands_lumens | PathMNIST | 200 | **400D** | 0.875 |
| vessel_trees | RetinaMNIST | 140 | **280D** | 0.303 |
| surface_lesions | DermaMNIST | 140 | **280D** | 0.389 |

**Rule**: `dim = n_bins × 2`
- Default: n_bins=140 → 280D
- glands_lumens: Higher n_bins=200 → 400D

---

### 1.4 persistence_silhouette

| Object Type | Dataset | n_bins | Output Dim | Accuracy |
|-------------|---------|--------|------------|----------|
| discrete_cells | BloodMNIST | 120 | **240D** | 0.942 |
| glands_lumens | PathMNIST | 120 | **240D** | 0.870 |
| vessel_trees | RetinaMNIST | 140 | **280D** | 0.313 |
| surface_lesions | DermaMNIST | 180 | **360D** | 0.393 |

**Rule**: `dim = n_bins × 2`
- Default: n_bins=120-140 → 240-280D
- surface_lesions: Higher n_bins=180 → 360D

---

### 1.5 persistence_entropy

| Object Type | Dataset | n_bins | Output Dim | Accuracy |
|-------------|---------|--------|------------|----------|
| discrete_cells | BloodMNIST | 200 | **400D** | 0.850 |
| glands_lumens | PathMNIST | 100 | **200D** | 0.839 |
| vessel_trees | RetinaMNIST | 100 | **200D** | 0.333 |
| surface_lesions | DermaMNIST | 40 | **80D** | 0.338 |

**Rule**: `dim = n_bins × 2`
- discrete_cells: n_bins=200 → 400D (⚠️ at boundary)
- glands_lumens/vessel_trees: n_bins=100 → 200D
- surface_lesions: n_bins=40 → 80D (smaller works better)

---

### 1.6 persistence_statistics

| Object Type | Dataset | subset | Output Dim | Accuracy |
|-------------|---------|--------|------------|----------|
| discrete_cells | BloodMNIST | full | **62D** | 0.850 |
| glands_lumens | PathMNIST | full | **62D** | 0.868 |
| vessel_trees | RetinaMNIST | full | **62D** | 0.312 |
| surface_lesions | DermaMNIST | full | **62D** | 0.380 |

**Rule**: Always use `subset='full'` → 62D
- basic: 28D, extended: 42D, full: 62D

---

### 1.7 tropical_coordinates

| Object Type | Dataset | max_terms | Output Dim | Accuracy |
|-------------|---------|-----------|------------|----------|
| discrete_cells | BloodMNIST | 18 | **144D** | 0.855 |
| glands_lumens | PathMNIST | 20 | **160D** | 0.701 |
| vessel_trees | RetinaMNIST | 10 | **80D** | 0.364 |
| surface_lesions | DermaMNIST | 20 | **160D** | 0.406 |

**Rule**: `dim = max_terms × 8`
- vessel_trees: max_terms=10 → 80D (simpler)
- Others: max_terms=18-20 → 144-160D

---

### 1.8 persistence_codebook

| Object Type | Dataset | codebook_size | Output Dim | Accuracy |
|-------------|---------|---------------|------------|----------|
| discrete_cells | BloodMNIST | 48 | **96D** | 0.949 |
| glands_lumens | PathMNIST | 56 | **112D** | 0.929 |
| vessel_trees | RetinaMNIST | 96 | **192D** | 0.360 |
| surface_lesions | DermaMNIST | 24 | **48D** | 0.424 |

**Rule**: `dim = codebook_size × 2`
- discrete_cells: 48 → 96D
- glands_lumens: 56 → 112D
- vessel_trees: 96 → 192D (needs more codebook words)
- surface_lesions: 24 → 48D (smaller works better)

---

### 1.9 ATOL

| Object Type | Dataset | n_centers | Output Dim | Accuracy |
|-------------|---------|-----------|------------|----------|
| discrete_cells | BloodMNIST | 20 | **40D** | 0.950 |
| glands_lumens | PathMNIST | 24 | **48D** | 0.930 |
| vessel_trees | RetinaMNIST | 20 | **40D** | 0.355 |
| surface_lesions | DermaMNIST | 18 | **36D** | 0.468 |

**Rule**: `dim = n_centers × 2`
- Default: n_centers=20 → 40D
- Stable across all object types (18-24 → 36-48D)

---

### 1.10 template_functions

| Object Type | Dataset | n_templates | Output Dim | Accuracy |
|-------------|---------|-------------|------------|----------|
| discrete_cells | BloodMNIST | 81 | **162D** | 0.957 |
| glands_lumens | PathMNIST | 100 | **200D** | 0.903 |
| vessel_trees | RetinaMNIST | 25 | **50D** | 0.374 |
| surface_lesions | DermaMNIST | 16 | **32D** | 0.441 |

**Rule**: `dim = n_templates × 2`
- discrete_cells: 81 → 162D (high complexity)
- glands_lumens: 100 → 200D
- vessel_trees: 25 → 50D (simpler)
- surface_lesions: 16 → 32D (small works best)

---

### 1.11 minkowski_functionals

| Object Type | Dataset | n_thresholds | Output Dim | Accuracy | Note |
|-------------|---------|--------------|------------|----------|------|
| discrete_cells | BloodMNIST | 40 | **120D** | 0.611 | Extended |
| glands_lumens | PathMNIST | 16 | **48D** | 0.312 | |
| vessel_trees | RetinaMNIST | 40 | **120D** | 0.211 | Extended |
| surface_lesions | DermaMNIST | 12 | **36D** | 0.220 | Confirmed |

**Rule**: `dim = n_thresholds × 3`
- discrete_cells: 35-40 → 105-120D (extended from 25, +3.1% accuracy)
- glands_lumens: 16 → 48D
- vessel_trees: 35-40 → 105-120D (extended from 9, marginal +0.2% improvement)
- surface_lesions: 12 → 36D (peak confirmed, extending didn't help)

---

### 1.12 euler_characteristic_curve

| Object Type | Dataset | resolution | Output Dim | Accuracy |
|-------------|---------|------------|------------|----------|
| discrete_cells | BloodMNIST | 160 | **160D** | 0.931 |
| glands_lumens | PathMNIST | 140 | **140D** | 0.876 |
| vessel_trees | RetinaMNIST | 160 | **160D** | 0.318 |
| surface_lesions | DermaMNIST | 160 | **160D** | 0.376 |

**Rule**: `dim = resolution`
- Default: resolution=160 → 160D
- Very stable across all object types

---

### 1.13 euler_characteristic_transform

| Object Type | Dataset | n_directions | Output Dim | Accuracy |
|-------------|---------|--------------|------------|----------|
| discrete_cells | BloodMNIST | 8 | **160D** | 0.304 |
| glands_lumens | PathMNIST | 20 | **400D** | 0.359 |
| vessel_trees | RetinaMNIST | 40 | **800D** | 0.253 |
| surface_lesions | DermaMNIST | - | - | - |

**Rule**: `dim = n_directions × 20` (n_heights=20 fixed)
- discrete_cells: 8 → 160D
- glands_lumens: 20 → 400D
- vessel_trees: 40 → 800D (needs more directions)

---

### 1.14 edge_histogram

| Object Type | Dataset | n_spatial_cells | Output Dim | Accuracy |
|-------------|---------|-----------------|------------|----------|
| discrete_cells | BloodMNIST | 60 | **480D** | 0.669 |
| glands_lumens | PathMNIST | 60 | **480D** | 0.585 |
| vessel_trees | RetinaMNIST | 48 | **384D** | 0.361 |
| surface_lesions | DermaMNIST | 48 | **384D** | 0.253 |

**Rule**: `dim = n_spatial_cells × 8`
- Default: n_spatial_cells=48-60 → 384-480D
- discrete_cells/glands_lumens: 60 → 480D
- vessel_trees/surface_lesions: 48 → 384D

---

### 1.15 lbp_texture

| Object Type | Dataset | n_scales | Output Dim | Accuracy |
|-------------|---------|----------|------------|----------|
| discrete_cells | BloodMNIST | 8 | **304D** | 0.843 |
| glands_lumens | PathMNIST | 7 | **238D** | 0.921 |
| vessel_trees | RetinaMNIST | 8 | **304D** | 0.423 |
| surface_lesions | DermaMNIST | 8 | **304D** | 0.449 |

**Rule**: `dim = 10 + 18 + 26 + 34 + 42 + 50 + 58 + 66` (cumulative per scale)
- Scales: (8,1), (16,2), (24,3), (32,4), (40,5), (48,6), (56,7), (64,8)
- 7 scales → 238D, 8 scales → 304D
- glands_lumens: 7 scales optimal (238D)
- Others: 8 scales optimal (304D)

---

## 2. Color Rules (Exp3)

### Rule: Always use `per_channel` for RGB images

| Object Type | Dataset | Grayscale Acc | Per-Channel Acc | Improvement |
|-------------|---------|---------------|-----------------|-------------|
| discrete_cells | BloodMNIST | baseline | +2.9% | **per_channel** |
| glands_lumens | PathMNIST | baseline | +3.3% | **per_channel** |
| vessel_trees | RetinaMNIST | baseline | +6.1% | **per_channel** |
| surface_lesions | DermaMNIST | baseline | +11.0% | **per_channel** |
| organ_shape | OrganAMNIST | N/A | N/A | grayscale only |

### Per-Channel Details:
- Computes PH separately on R, G, B channels
- Combines: `H0 = H0_R + H0_G + H0_B`, `H1 = H1_R + H1_G + H1_B`
- **Output dimension = grayscale_dim × 3**
- **Computation time = grayscale_time × 3**

### Dimension Adjustment:
When using per_channel with RGB images:
```
per_channel_dim = grayscale_dim × 3

Example:
- betti_curves n_bins=140 → 280D (grayscale) → 840D (per_channel)
- persistence_image res=14 → 392D (grayscale) → 1176D (per_channel)
```

---

## 3. Classifier Rules

| Features | Classifier | Notes |
|----------|------------|-------|
| ≤ 2000D | **TabPFN** | Fast, good for small-medium features |
| > 2000D | **XGBoost** | No feature limit, slightly slower |

### When to Use XGBoost:
- persistence_landscapes with n_layers > 20
- Any descriptor with per_channel on RGB (if >2000D after ×3)
- High-resolution persistence_image (res > 25)

---

## 4. Parameter Rules (Non-Dimension Parameters)

### 4.1 minkowski_functionals: `adaptive=False` ALWAYS better

| Object Type | adaptive=True | adaptive=False | Improvement |
|-------------|---------------|----------------|-------------|
| discrete_cells | 0.580 | **0.664** | +8.4% |
| glands_lumens | 0.312 | **0.411** | +9.9% |
| vessel_trees | 0.209 | **0.216** | +0.7% |
| surface_lesions | 0.220 | **0.287** | +6.7% |
| organ_shape | 0.340 | **0.390** | +5.0% |

**Rule**: Always use `adaptive=False`

### 4.2 persistence_silhouette: `power` varies by object type

| Object Type | Best Power | Accuracy |
|-------------|------------|----------|
| discrete_cells | **0.5** | 0.901 |
| glands_lumens | **0.5** | 0.851 |
| vessel_trees | **2.0** | 0.315 |
| surface_lesions | **1.0** | 0.353 |
| organ_shape | **0.5** | 0.595 |

**Rule**: Use power=0.5 as default, except vessel_trees (2.0) and surface_lesions (1.0)

### 4.3 persistence_image: `sigma` and `weight_function` vary by object type

| Object Type | Best sigma | Best weight_fn | Accuracy | At boundary? |
|-------------|-----------|----------------|----------|-------------|
| discrete_cells | **0.25** | squared | 0.606 | ⚠️ YES |
| glands_lumens | **0.25** | squared | 0.694 | ⚠️ YES |
| vessel_trees | **0.05** | squared | 0.374 | no |
| surface_lesions | **0.15** | linear | 0.441 | no |
| organ_shape | **0.25** | const | 0.342 | ⚠️ YES |

**Rule**: weight_function=squared as default (3/5 wins). sigma varies: low for vessel_trees, mid for surface_lesions, at search boundary for the rest (needs extension to [0.3, 0.35, 0.4]).

### 4.4 edge_histogram: `n_orientation_bins=8` wins 4/5

| Object Type | Best n_orientation_bins | Accuracy |
|-------------|------------------------|----------|
| discrete_cells | **8** | 0.669 |
| glands_lumens | **8** | 0.585 |
| vessel_trees | **8** | 0.361 |
| surface_lesions | 4 | 0.253 |
| organ_shape | **8** | 0.827 |

**Rule**: Use n_orientation_bins=8 as default. DermaMNIST: 4 wins by noise margin (0.2531 vs 0.2529).

### 4.5 Other Parameters (Uniform Across Object Types)

| Descriptor | Parameter | Best Value | Notes |
|------------|-----------|------------|-------|
| persistence_landscapes | combine_dims | **False** | Separate H0/H1 (4/5 wins) |
| persistence_landscapes | n_bins | **50** | Lower bins better (4/5 wins) |
| betti_curves | normalize | False | normalize=False unanimous (5/5) |
| persistence_entropy | mode | "vector" | Per-bin entropy (5/5 wins) |
| persistence_entropy | normalized | True/False | **No effect**: identical accuracy on all 5 datasets |
| template_functions | template_type | "tent" | Slightly better than gaussian |
| lbp_texture | method | "uniform" | All methods produce **identical** results |

---

## 5. Quick Reference: Object Type → Dimension Lookup

### discrete_cells (e.g., BloodMNIST)
```python
{
    'persistence_image': {'resolution': 10},           # 200D
    'persistence_landscapes': {'n_layers': 12},        # 1200D
    'betti_curves': {'n_bins': 120},                   # 240D
    'persistence_silhouette': {'n_bins': 120},         # 240D
    'persistence_entropy': {'n_bins': 200},            # 400D
    'persistence_statistics': {'subset': 'full'},      # 62D
    'tropical_coordinates': {'max_terms': 18},         # 144D
    'persistence_codebook': {'codebook_size': 48},     # 96D
    'ATOL': {'n_centers': 20},                         # 40D
    'template_functions': {'n_templates': 81},         # 162D
    'minkowski_functionals': {'n_thresholds': 40},     # 120D (extended)
    'euler_characteristic_curve': {'resolution': 160}, # 160D
    'euler_characteristic_transform': {'n_directions': 8}, # 160D
    'edge_histogram': {'n_spatial_cells': 60},         # 480D
    'lbp_texture': {'n_scales': 8},                    # 304D
}
```

### glands_lumens (e.g., PathMNIST)
```python
{
    'persistence_image': {'resolution': 26},           # 1352D
    'persistence_landscapes': {'n_layers': 12},        # 1200D
    'betti_curves': {'n_bins': 200},                   # 400D
    'persistence_silhouette': {'n_bins': 120},         # 240D
    'persistence_entropy': {'n_bins': 100},            # 200D
    'persistence_statistics': {'subset': 'full'},      # 62D
    'tropical_coordinates': {'max_terms': 20},         # 160D
    'persistence_codebook': {'codebook_size': 56},     # 112D
    'ATOL': {'n_centers': 24},                         # 48D
    'template_functions': {'n_templates': 100},        # 200D
    'minkowski_functionals': {'n_thresholds': 16},     # 48D
    'euler_characteristic_curve': {'resolution': 140}, # 140D
    'euler_characteristic_transform': {'n_directions': 20}, # 400D
    'edge_histogram': {'n_spatial_cells': 60},         # 480D
    'lbp_texture': {'n_scales': 7},                    # 238D
}
```

### vessel_trees (e.g., RetinaMNIST)
```python
{
    'persistence_image': {'resolution': 14},           # 392D
    'persistence_landscapes': {'n_layers': 7},         # 700D
    'betti_curves': {'n_bins': 140},                   # 280D
    'persistence_silhouette': {'n_bins': 140},         # 280D
    'persistence_entropy': {'n_bins': 100},            # 200D
    'persistence_statistics': {'subset': 'full'},      # 62D
    'tropical_coordinates': {'max_terms': 10},         # 80D
    'persistence_codebook': {'codebook_size': 96},     # 192D
    'ATOL': {'n_centers': 20},                         # 40D
    'template_functions': {'n_templates': 25},         # 50D
    'minkowski_functionals': {'n_thresholds': 40},     # 120D (extended)
    'euler_characteristic_curve': {'resolution': 160}, # 160D
    'euler_characteristic_transform': {'n_directions': 40}, # 800D
    'edge_histogram': {'n_spatial_cells': 48},         # 384D
    'lbp_texture': {'n_scales': 8},                    # 304D
}
```

### surface_lesions (e.g., DermaMNIST)
```python
{
    'persistence_image': {'resolution': 12},           # 288D
    'persistence_landscapes': {'n_layers': 12},        # 1200D
    'betti_curves': {'n_bins': 140},                   # 280D
    'persistence_silhouette': {'n_bins': 180},         # 360D
    'persistence_entropy': {'n_bins': 40},             # 80D
    'persistence_statistics': {'subset': 'full'},      # 62D
    'tropical_coordinates': {'max_terms': 20},         # 160D
    'persistence_codebook': {'codebook_size': 24},     # 48D
    'ATOL': {'n_centers': 18},                         # 36D
    'template_functions': {'n_templates': 16},         # 32D
    'minkowski_functionals': {'n_thresholds': 12},     # 36D
    'euler_characteristic_curve': {'resolution': 160}, # 160D
    'euler_characteristic_transform': {'n_directions': 20}, # 400D (estimated)
    'edge_histogram': {'n_spatial_cells': 48},         # 384D
    'lbp_texture': {'n_scales': 8},                    # 304D
}
```

### organ_shape (e.g., OrganAMNIST)
```python
{
    # Note: OrganAMNIST is grayscale (no per_channel)
    # Limited data available - using defaults where not tested
    'persistence_landscapes': {'n_layers': 35},        # 3500D (XGBoost required)
    # For other descriptors, use similar values to discrete_cells:
    'persistence_image': {'resolution': 12},           # 288D (estimated)
    'betti_curves': {'n_bins': 120},                   # 240D (estimated)
    'persistence_silhouette': {'n_bins': 120},         # 240D (estimated)
    'persistence_entropy': {'n_bins': 100},            # 200D (estimated)
    'persistence_statistics': {'subset': 'full'},      # 62D
    'tropical_coordinates': {'max_terms': 15},         # 120D (estimated)
    'persistence_codebook': {'codebook_size': 48},     # 96D (estimated)
    'ATOL': {'n_centers': 20},                         # 40D (estimated)
    'template_functions': {'n_templates': 49},         # 98D (estimated)
    'minkowski_functionals': {'n_thresholds': 20},     # 60D (estimated)
    'euler_characteristic_curve': {'resolution': 160}, # 160D (estimated)
    'euler_characteristic_transform': {'n_directions': 16}, # 320D (estimated)
    'edge_histogram': {'n_spatial_cells': 48},         # 384D (estimated)
    'lbp_texture': {'n_scales': 8},                    # 304D (estimated)
}
```

---

## 6. Top Performers by Object Type

### discrete_cells (BloodMNIST)
| Rank | Descriptor | Accuracy |
|------|------------|----------|
| 1 | template_functions | **0.957** |
| 2 | ATOL | 0.950 |
| 3 | persistence_codebook | 0.949 |
| 4 | betti_curves | 0.949 |
| 5 | persistence_silhouette | 0.942 |

### glands_lumens (PathMNIST)
| Rank | Descriptor | Accuracy |
|------|------------|----------|
| 1 | ATOL | **0.930** |
| 2 | persistence_codebook | 0.929 |
| 3 | lbp_texture | 0.921 |
| 4 | template_functions | 0.903 |
| 5 | euler_characteristic_curve | 0.876 |

### vessel_trees (RetinaMNIST)
| Rank | Descriptor | Accuracy |
|------|------------|----------|
| 1 | lbp_texture | **0.423** |
| 2 | template_functions | 0.374 |
| 3 | tropical_coordinates | 0.364 |
| 4 | persistence_landscapes (XGB) | 0.362 |
| 5 | edge_histogram | 0.361 |

### surface_lesions (DermaMNIST)
| Rank | Descriptor | Accuracy |
|------|------------|----------|
| 1 | ATOL | **0.468** |
| 2 | lbp_texture | 0.449 |
| 3 | template_functions | 0.441 |
| 4 | persistence_image | 0.432 |
| 5 | persistence_codebook | 0.424 |

### organ_shape (OrganAMNIST)
| Rank | Descriptor | Accuracy | Source |
|------|------------|----------|--------|
| 1 | **edge_histogram** | **0.827** | param search |
| 2 | lbp_texture | 0.729 | param search |
| 3 | template_functions | 0.705 | dim search |
| 4 | betti_curves | 0.671 | param search |
| 5 | persistence_landscapes (XGB) | 0.663 | param search |
| 6 | persistence_image | 0.606 | param search |
| 7 | persistence_silhouette | 0.595 | param search |
| 8 | persistence_entropy | 0.548 | param search |
| 9 | minkowski_functionals | 0.390 | param search |
| 10 | persistence_image (dim) | 0.342 | param search |

Note: edge_histogram dominates for organ_shape (grayscale). Image-based descriptors (edge_histogram, lbp_texture) outperform most PH-based descriptors on this dataset.

---

## 7. Warnings and Limitations

### Boundary Issues - RESOLVED
| Descriptor | Dataset | Original | Extended To | Result |
|------------|---------|----------|-------------|--------|
| minkowski_functionals | BloodMNIST | n_thresh=25 | n_thresh=40 | ✅ Improved: 0.580 → 0.611 |
| minkowski_functionals | RetinaMNIST | n_thresh=9 | n_thresh=40 | ✅ Marginal: 0.209 → 0.211 |
| minkowski_functionals | DermaMNIST | n_thresh=12 | n_thresh=40 | ✅ Confirmed peak at 12 |
| persistence_entropy | BloodMNIST | n_bins=200 | n_bins=300 | ✅ Confirmed peak at 200 |
| persistence_entropy | PathMNIST | n_bins=100 | n_bins=300 | ✅ Confirmed peak at 100 |
| persistence_entropy | RetinaMNIST | n_bins=100 | n_bins=300 | ✅ Confirmed peak at 100 |
| persistence_entropy | DermaMNIST | n_bins=40 | n_bins=300 | ✅ Confirmed peak at 40 |

### Remaining Boundary Issues
| Descriptor | Dataset | At Boundary | Action |
|------------|---------|-------------|--------|
| persistence_image | BloodMNIST, PathMNIST, OrganAMNIST | sigma=0.25 (max tested) | **CRITICAL**: Extend to sigma=[0.3, 0.35, 0.4] |
| lbp_texture | RetinaMNIST, DermaMNIST | 8 scales (max) | May need 9-10 scales (low priority) |

### TabPFN Limitations
- Max 2000 features
- Max 10 classes (uses ECOC wrapper for >10)
- May fail on very high-dimensional descriptors with per_channel

### OrganAMNIST (organ_shape)
- Grayscale dataset (per_channel not applicable)
- All 8 param search descriptors now tested (see Section 6)
- edge_histogram (0.827) is the top performer, followed by lbp_texture (0.729)
