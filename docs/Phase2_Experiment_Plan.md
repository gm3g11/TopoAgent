# Phase 2: Comprehensive Descriptor Exploration Experiments

## Current Status: Phase 1 Complete → Ready for Phase 2

**Phase 1 Complete:** 14 descriptors × 8 datasets × 500 samples benchmarked
**Current Task:** Design and implement experiments from "Descriptor exploration framework.md"

---

## Phase 1 Key Findings (Summary)

| Metric | Winner | Value |
|--------|--------|-------|
| Best Avg Rank | persistence_image | 3.50 (most consistent) |
| Most Wins | persistence_statistics | 3/8 datasets |
| Best Microscopy | atol | 85.6% (blood), 72.2% (path) |
| Best CT/OCT | lbp_texture | 76.0% (organ), 50.8% (oct) |
| Correlation | Most pairs | r < 0.2 (complementary) |

**Key Insight:** No single descriptor dominates → Adaptive selection is justified

---

## Phase 2 Datasets

```python
PHASE2_DATASETS = {
    # === KEEP FROM PHASE 1 (7 datasets) ===
    'bloodmnist': {'modality': 'microscopy', 'expected_filtration': 'superlevel', 'expected_dominant': 'H0'},
    'pathmnist': {'modality': 'histopathology', 'expected_filtration': 'superlevel', 'expected_dominant': 'H0+H1'},
    'retinamnist': {'modality': 'fundus', 'expected_filtration': 'sublevel', 'expected_dominant': 'H1'},
    'organamnist': {'modality': 'ct', 'expected_filtration': 'sublevel', 'expected_dominant': 'H0'},
    'octmnist': {'modality': 'oct', 'expected_filtration': 'sublevel', 'expected_dominant': 'layers'},
    'isic2019': {'modality': 'dermoscopy', 'expected_filtration': 'mixed', 'expected_dominant': 'boundary'},
    'kvasir': {'modality': 'endoscopy', 'expected_filtration': 'mixed', 'expected_dominant': 'H0+H1'},

    # === ADD NEW ===
    'chestmnist': {'modality': 'xray', 'expected_filtration': 'sublevel', 'expected_dominant': 'H0'},

    # === OPTIONAL (if time permits) ===
    # 'breastmnist': {'modality': 'ultrasound', ...},
    # 'tissuemnist': {'modality': 'microscopy', ...},  # Redundant with bloodmnist?
}
```

---

## Revised Experiment List (10 Total)

### CORE EXPERIMENTS (7)

| # | Experiment | Goal | Output |
|---|------------|------|--------|
| 1 | Main benchmark | 5000 samples, 4 classifiers, chance-corrected | Table 1 + Critical diff diagram |
| 2 | H0 vs H1 ablation | Top 6 descriptors × 8 datasets | Table 2: H0/H1 dominance |
| 3 | Filtration comparison | **8 datasets × 1000 samples:** 5 desc × 3 filt | Table 3 + sensitivity heatmap |
| 4 | Sample efficiency | 2-3 datasets, learning curves | Figure: "When does PI catch Stats?" |
| 5 | Stability analysis | 10 seeds, identify high-variance descriptors | Table 4: CV per descriptor |
| 6 | CKA + Ensemble | Complementarity + synergy validation | CKA matrix + Table 5: ensembles |
| 7 | Dataset characterization + meta-learning | **CHEAP features, TRAIN-ONLY, learned rules** | Decision tree + SHAP |

### ANALYSIS/DOCUMENTATION (3)

| # | Analysis | Goal | Output |
|---|----------|------|--------|
| 8 | Descriptor taxonomy | Theory + validation pointers | 1-2 page table |
| 9 | Parameter configs | Small/Medium/Large per descriptor | Pareto plot |
| 10 | Per-class analysis | Best descriptor per class | Table 6 + failure examples |

---

## Phase 2 File Structure

```
scripts/run_benchmark2/
├── __init__.py
├── config.py                      # Shared configs (5000 samples, 4 classifiers, taxonomy)
├── utils.py                       # Metrics (kappa, CKA, Friedman, cheap characterization)
├── exp1_main_benchmark.py         # Main benchmark with statistical tests
├── exp2_h0_h1_ablation.py         # H0/H1 separation
├── exp3_filtration_comparison.py  # BUDGETED: 3 datasets × 5 desc × 3 filt
├── exp4_sample_efficiency.py      # Learning curves
├── exp5_stability_analysis.py     # 10 seeds, CV measurement
├── exp6_cka_ensemble.py           # CKA + ensemble validation
├── exp7_meta_learning.py          # CHEAP features, TRAIN-ONLY, learned rules
├── analysis8_taxonomy.py          # Generate taxonomy table
├── analysis9_pareto.py            # Small/Medium/Large configs, Pareto
├── analysis10_per_class.py        # Per-class F1 + failure examples
└── run_all.py                     # Master runner

results/benchmark2/
├── exp1_main/{dataset}_results.json
├── exp2_h0h1/{dataset}_h0h1.json
├── exp3_filtration/{dataset}_filtration.json
├── exp4_learning_curves/{dataset}_curves.json
├── exp5_stability/stability_report.json
├── exp6_cka_ensemble/cka_matrix.json
├── exp7_meta_learning/
│   ├── decision_rules.json        # Learned rules
│   ├── shap_explanations.json     # SHAP feature importance
│   └── leave_one_out_results.json # LOO validation
├── topoagent/
│   ├── performance_lookup.json
│   └── TOPOAGENT_SKILL.md
└── PHASE2_REPORT.md
```

---

## Key Configuration

### 1. Sample Size: 500 → 5000
```python
N_SAMPLES = 5000  # Fair evaluation for high-dim descriptors
# PI at 5000: 6.25 samples/dim (acceptable)
# PI at 500: 0.6 samples/dim (unfair)
```

### 2. Classifiers (4)
```python
CLASSIFIERS = {
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1),
    'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
    'TabPFN': TabPFNClassifier(N_ensemble_configurations=16),  # Skip for OrganAMNIST (11 classes > 10 limit)
}
```

### 3. Descriptor Dimension Configs (Small/Medium/Large)
```python
DIMENSION_CONFIGS = {
    'persistence_image': {
        'small': {'resolution': 10},   # 10×10×2 = 200D
        'medium': {'resolution': 20},  # 20×20×2 = 800D
        'large': {'resolution': 30},   # 30×30×2 = 1800D
    },
    'persistence_landscapes': {
        'small': {'n_layers': 3, 'n_bins': 50},   # 150D
        'medium': {'n_layers': 5, 'n_bins': 100}, # 500D
        'large': {'n_layers': 10, 'n_bins': 100}, # 1000D
    },
    'betti_curves': {
        'small': {'n_bins': 50},   # 100D
        'medium': {'n_bins': 100}, # 200D
        'large': {'n_bins': 200},  # 400D
    },
    # ... etc
}

# Also keep ~100D fair comparison
DESCRIPTOR_CONFIGS_100D = {
    'persistence_image': {'resolution': 7},           # 98D
    'persistence_landscapes': {'n_layers': 5, 'n_bins': 10},  # 100D
    'betti_curves': {'n_bins': 50},                   # 100D
    'persistence_silhouette': {'n_bins': 50},         # 100D
    'atol': {'n_centers': 50},                        # 100D
    'persistence_statistics': {'subset': 'extended'}, # ~42D
}
```

### 4. Chance-Corrected Metrics + Statistical Tests
```python
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

def compute_metrics(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    chance = 1.0 / n_classes
    return {
        'accuracy': acc,
        'normalized_gain': (acc - chance) / (1 - chance),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }

def statistical_significance(accuracy_matrix):
    """
    accuracy_matrix: shape (n_datasets, n_descriptors)
    """
    # Friedman test: Are there significant differences?
    stat, p_value = friedmanchisquare(*accuracy_matrix.T)

    if p_value < 0.05:
        # Nemenyi post-hoc: Which pairs are significantly different?
        nemenyi_results = posthoc_nemenyi_friedman(accuracy_matrix)
        return {'friedman_p': p_value, 'nemenyi': nemenyi_results}
    else:
        return {'friedman_p': p_value, 'nemenyi': None}
```

### 5. Descriptor Taxonomy (Documentation, not Experiment)
```python
DESCRIPTOR_TAXONOMY = {
    'learned': ['atol', 'persistence_codebook'],  # Need fit() before transform()
    'fixed': ['persistence_statistics', 'betti_curves', 'persistence_image',
              'persistence_landscapes', 'persistence_silhouette', 'persistence_entropy',
              'carlsson_coordinates', 'tropical_coordinates', 'euler_curve',
              'minkowski_functionals', 'lbp_texture'],
    'fast': ['euler_curve', 'betti_curves', 'carlsson_coordinates',
             'persistence_entropy', 'persistence_statistics'],
    'stable': ['persistence_statistics', 'betti_curves', 'persistence_silhouette'],
    'unstable': ['atol', 'persistence_codebook', 'tropical_coordinates'],
}

INTERPRETABILITY = {
    'high': ['betti_curves', 'euler_curve', 'persistence_statistics'],
    'medium': ['persistence_image', 'persistence_landscapes', 'persistence_silhouette'],
    'low': ['atol', 'persistence_codebook', 'tropical_coordinates'],
}
```

---

## Exp 3: Filtration Comparison (8 Datasets with Less Data)

### Design: All datasets with reduced samples
- Full factorial: 8 × 10 × 3 × 4 = **960 runs** (too much)
- **Better design**: 8 × 5 × 3 × 2 = **240 runs** (see FULL modality patterns!)

```python
FILTRATION_STUDY = {
    'datasets': ALL_8_DATASETS,       # Need all to see modality patterns!
    'n_samples': 1000,                # Reduced from 5000 (sufficient for filtration)
    'descriptors': ['persistence_statistics', 'betti_curves',
                    'persistence_image', 'persistence_silhouette', 'atol'],  # 5
    'filtrations': ['sublevel', 'superlevel', 'combined'],  # 3
    'classifiers': ['XGBoost', 'LogisticRegression'],  # 2 (speed)
}
# Total: 8 × 5 × 3 × 2 = 240 runs
# Now we see the FULL pattern across modalities!
```

### Expected Results
| Dataset | Modality | Expected Best | Reasoning |
|---------|----------|---------------|-----------|
| BloodMNIST | microscopy | **Superlevel** | Cells are DARK on light background |
| PathMNIST | histopathology | **Superlevel** | H&E staining (purple nuclei are dark) |
| RetinaMNIST | fundus | Sublevel | Vessels are bright |
| OrganAMNIST | ct | Sublevel | CT intensity meaningful |
| OCTMNIST | oct | Sublevel | Layer intensities meaningful |
| ChestMNIST | xray | Sublevel | Intensity meaningful |
| ISIC2019 | dermoscopy | Mixed | Depends on lesion type |
| Kvasir | endoscopy | Mixed | Varies by polyp type |

---

## Exp 5: Stability Analysis

### Goal: Identify high-variance descriptors

```python
def stability_analysis(features_dict, labels, clf, n_trials=10):
    results = {desc: [] for desc in features_dict}
    for seed in range(n_trials):
        np.random.seed(seed)
        for desc_name, X in features_dict.items():
            clf_copy = clone(clf)
            if hasattr(clf_copy, 'random_state'):
                clf_copy.set_params(random_state=seed)
            scores = cross_val_score(clf_copy, X, labels, cv=5)
            results[desc_name].append(scores.mean())

    stability = {}
    for desc, scores in results.items():
        stability[desc] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'cv': np.std(scores) / np.mean(scores),  # Coefficient of variation
        }
    return stability
```

### Expected CV Ranges
| Descriptor | Expected CV | Reliability |
|------------|-------------|-------------|
| persistence_statistics | < 0.02 | Very stable ✓ |
| betti_curves | < 0.03 | Stable ✓ |
| persistence_image | 0.03-0.05 | Moderate |
| atol | **> 0.05** | **Unstable ⚠️** |
| persistence_codebook | **> 0.05** | **Unstable ⚠️** |

---

## Exp 7: Dataset Characterization + Meta-Learning (REFINED)

### Issue 1: Only 8 Datasets → Simpler Model
```python
# With only 8 data points, max_depth=4 will overfit
tree = DecisionTreeClassifier(
    max_depth=3,         # Reduced from 4
    min_samples_leaf=1,  # At least 1 dataset per leaf (only 8 total)
    min_samples_split=2, # Need 2 datasets to split
)

# Also try: Regularized Logistic Regression as alternative
meta_logreg = LogisticRegression(max_iter=1000, C=0.1)  # Regularized
```

### Issue 2: TRAIN-ONLY Computation (Avoid Leakage)
```python
# ❌ WRONG (leakage):
characteristics = characterize_dataset(X_all, y_all)  # Uses test data!

# ✅ CORRECT:
for train_idx, test_idx in cv.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    characteristics = characterize_dataset(X_train, y_train)  # Train only!
```

### Issue 3: RICH Cheap Features (No PH to Choose PH)
```python
def cheap_characterization(images, labels):
    """
    Fast, PH-free features that PREDICT descriptor performance.
    Cost: ~10-20ms per dataset (vs ~minutes for PH)

    MUST be computed on TRAIN ONLY!
    """
    sample_idx = np.random.choice(len(images), min(100, len(images)), replace=False)
    sample_imgs = images[sample_idx]

    chars = {
        # === Sample statistics ===
        'n_samples': len(images),
        'n_classes': len(np.unique(labels)),
        'samples_per_class': len(images) / len(np.unique(labels)),

        # === Intensity statistics ===
        'intensity_mean': np.mean([img.mean() for img in sample_imgs]),
        'intensity_std': np.mean([img.std() for img in sample_imgs]),
        'intensity_skewness': np.mean([skew(img.ravel()) for img in sample_imgs]),
        'intensity_kurtosis': np.mean([kurtosis(img.ravel()) for img in sample_imgs]),
        'intensity_p95_p5': np.mean([
            np.percentile(img, 95) - np.percentile(img, 5) for img in sample_imgs
        ]),
        'intensity_entropy': np.mean([
            entropy(np.histogram(img.ravel(), bins=64)[0] + 1e-10) for img in sample_imgs
        ]),

        # === Edge/Gradient statistics ===
        'edge_density': np.mean([canny(img).mean() for img in sample_imgs]),
        'gradient_mean': np.mean([np.abs(sobel(img)).mean() for img in sample_imgs]),

        # === Topology proxies (NO PH!) ===
        'otsu_components': np.mean([
            label(img > threshold_otsu(img))[1] for img in sample_imgs
        ]),
        'otsu_holes': np.mean([
            max(0, label(~(img > threshold_otsu(img)))[1] - 1) for img in sample_imgs
        ]),
        'binary_fill_ratio': np.mean([
            (img > threshold_otsu(img)).mean() for img in sample_imgs
        ]),

        # === Class structure ===
        'class_imbalance': np.bincount(labels).max() / (np.bincount(labels).min() + 1),
    }
    return chars

FEATURE_NAMES = [
    'n_samples', 'n_classes', 'samples_per_class',
    'intensity_mean', 'intensity_std', 'intensity_skewness',
    'intensity_kurtosis', 'intensity_p95_p5', 'intensity_entropy',
    'edge_density', 'gradient_mean',
    'otsu_components', 'otsu_holes', 'binary_fill_ratio',
    'class_imbalance',
]
```

### Issue 4: Predict MULTIPLE Outputs (Not Just Descriptor)
```python
def learn_topoagent_rules(all_results, all_characteristics, feature_names):
    """
    Learn THREE decision models:
    1. Best descriptor
    2. Best filtration (from Exp 3)
    3. H0 vs H1 dominance (from Exp 2)
    """
    X_meta = np.array([
        [all_characteristics[d][f] for f in feature_names]
        for d in datasets
    ])

    # Model 1: Best descriptor
    y_descriptor = np.array([all_results[d]['best_descriptor'] for d in datasets])
    tree_descriptor = DecisionTreeClassifier(max_depth=3)

    # Model 2: Best filtration (from Exp 3)
    y_filtration = np.array([all_results[d]['best_filtration'] for d in datasets])
    tree_filtration = DecisionTreeClassifier(max_depth=2)  # Simpler

    # Model 3: H0 vs H1 dominance (from Exp 2)
    y_homology = np.array([all_results[d]['dominant_homology'] for d in datasets])  # 'H0', 'H1', 'both'
    tree_homology = DecisionTreeClassifier(max_depth=2)

    # Leave-one-dataset-out validation for each...

    return {
        'descriptor_tree': tree_descriptor,
        'filtration_tree': tree_filtration,
        'homology_tree': tree_homology,
        'rules': combined_rules,
    }
```

### Issue 5: Confidence + Fallback Strategy
```python
def topoagent_recommend(characteristics, models):
    """
    Recommend with confidence and fallback.
    With only 8 datasets, meta-model might be wrong!
    """
    X = np.array([[characteristics[f] for f in FEATURE_NAMES]])

    recommendations = {}
    for model_name in ['descriptor', 'filtration', 'homology']:
        tree = models[model_name]['model']
        pred = tree.predict(X)[0]
        proba = tree.predict_proba(X)[0]
        confidence = max(proba)

        recommendations[model_name] = {
            'prediction': pred,
            'confidence': confidence,
            'confidence_level': 'high' if confidence > 0.6 else 'low',
        }

    # Fallback for low-confidence descriptor
    if recommendations['descriptor']['confidence_level'] == 'low':
        recommendations['descriptor']['fallback'] = 'persistence_statistics'  # Always robust
        recommendations['descriptor']['note'] = 'Low confidence - consider trying both'

    return recommendations
```

### Why This Matters
| Approach | Problem | Solution |
|----------|---------|----------|
| max_depth=4 | Overfits with 8 datasets | max_depth=3 |
| Single output | Incomplete guidance | Predict descriptor + filtration + homology |
| No confidence | Blindly trusts wrong predictions | Confidence + fallback |
| Hand-written if/else | Heuristic, not rigorous | Learned rules from data |
| Full dataset characterization | Leakage to test set | Train-only computation |
| PH-based features | Expensive + circular | Cheap proxies (Otsu, edges, GLCM) |
| No validation | Can't trust rules | Leave-one-dataset-out |
| Black box | Not publishable | Decision tree + SHAP |

---

## H0 vs H1 Ablation (Exp 2)

### Descriptors Supporting H0/H1 Separation (10 PH-based)
| Descriptor | Full Dim | H0 Dim | H1 Dim |
|------------|----------|--------|--------|
| persistence_image | 800 | 400 | 400 |
| persistence_landscapes | 1000 | 500 | 500 |
| betti_curves | 200 | 100 | 100 |
| persistence_silhouette | 200 | 100 | 100 |
| persistence_statistics | 50 | 25 | 25 |
| atol | 8 | 4 | 4 |

### Expected Biological Insights
| Dataset | Expected Primary | Reasoning |
|---------|------------------|-----------|
| BloodMNIST | **H0** | Discrete cells = counting components |
| RetinaMNIST | **H1** | Vessel loops and branching |
| PathMNIST | **H0+H1** synergy | Glands (H0) + lumens (H1) |
| OrganAMNIST | **H0** | Solid organ regions |

---

## Execution Plan

### Phase 2.0: Setup + Taxonomy Documentation
```bash
# Generate taxonomy table (analysis, not experiment)
python analysis8_taxonomy.py
# Output: results/benchmark2/descriptor_taxonomy.md
```

### Phase 2.1: Main Benchmark (Exp 1)
```bash
# Run with statistical significance tests
python exp1_main_benchmark.py --all-datasets --n-samples 5000
# Includes Friedman + Nemenyi post-hoc
# Output: Table 1 + Critical difference diagram
```

### Phase 2.2: Filtration Comparison (Exp 3) - 8 Datasets × 1000 Samples
```bash
python exp3_filtration_comparison.py \
    --all-datasets \
    --n-samples 1000 \
    --descriptors stats,betti,pi,silhouette,atol \
    --filtrations sublevel,superlevel,combined
# 8 × 5 × 3 × 2 = 240 runs → see FULL modality patterns
# Output: Table 3 + sensitivity heatmap
# Use winning filtration for subsequent experiments!
```

### Phase 2.3: H0/H1 Ablation (Exp 2)
```bash
python exp2_h0_h1_ablation.py --all-datasets \
    --descriptors "PI,betti,stats,atol,landscapes,silhouette"
# Output: Table 2: H0/H1 dominance per dataset
```

### Phase 2.4: Sample Efficiency (Exp 4)
```bash
python exp4_sample_efficiency.py \
    --datasets bloodmnist,pathmnist,retinamnist \
    --train-sizes "0.1,0.25,0.5,0.75,1.0"
# Output: Figure 3: Learning curves
```

### Phase 2.5: Stability Analysis (Exp 5)
```bash
python exp5_stability_analysis.py --n-trials 10 --all-datasets
# Output: Table 4: CV per descriptor
```

### Phase 2.6: CKA + Ensemble (Exp 6)
```bash
python exp6_cka_ensemble.py --all-datasets \
    --ensembles "stats+atol,stats+atol+betti,stats+atol+lbp"
# Output: CKA matrix + Table 5: ensemble synergy
```

### Phase 2.7: Meta-Learning (Exp 7) - CRITICAL
```bash
python exp7_meta_learning.py --all-datasets
# Uses CHEAP features, TRAIN-ONLY computation
# Leave-one-dataset-out validation
# Output: Decision tree rules + SHAP explanations
```

### Phase 2.8: Per-Class Analysis (Analysis 10)
```bash
python analysis10_per_class.py --datasets bloodmnist,organamnist
# Output: Table 6: Best descriptor per class + failure examples
```

### Phase 2.9: Pareto Analysis (Analysis 9)
```bash
python analysis9_pareto.py --all-datasets
# Output: Pareto plot (accuracy vs dim vs time)
```

---

## Expected Paper Outputs

### Tables
| Table | Content | Source |
|-------|---------|--------|
| Table 1 | Chance-corrected accuracy (κ, normalized_gain) + **Friedman p-value** | Exp 1 |
| Table 2 | H0 vs H1 dominance per dataset | Exp 2 |
| Table 3 | Filtration comparison (sublevel vs superlevel) | Exp 3 |
| Table 4 | Stability/reliability ranking (CV per descriptor) | Exp 5 |
| Table 5 | Ensemble synergy results | Exp 6 |
| Table 6 | Best descriptor per class | Analysis 10 |
| Table 7 | Descriptor taxonomy (theory + validation) | Analysis 8 |

### Figures
| Figure | Content | Source |
|--------|---------|--------|
| Fig 1 | **Critical difference diagram (Nemenyi)** | Exp 1 |
| Fig 2 | H0/H1 dominance heatmap | Exp 2 |
| Fig 3 | Learning curves (PI vs Statistics vs ATOL) | Exp 4 |
| Fig 4 | CKA complementarity matrix | Exp 6 |
| Fig 5 | Filtration sensitivity heatmap | Exp 3 |
| Fig 6 | **TopoAgent decision tree + SHAP** | Exp 7 |
| Fig 7 | Pareto frontier (accuracy vs dim vs time) | Analysis 9 |

---

## Summary: What TopoAgent Learns from Phase 2

| Level | What's Learned | How It's Used | Source |
|-------|----------------|---------------|--------|
| **Filtration** | "Microscopy → superlevel, CT → sublevel" | Automatic filtration selection | Exp 3 |
| **Homology** | "BloodMNIST is H0-driven, RetinaMNIST is H1-driven" | Domain-specific defaults | Exp 2 |
| **Reliability** | "Statistics stable (CV<0.02), ATOL unstable (CV>0.05)" | Confidence intervals | Exp 5 |
| **Sample size** | "PI needs N>4000 to beat Statistics" | Threshold-based selection | Exp 4 |
| **Ensemble** | "Stats+ATOL+LBP = +5-7% synergy" | Maximum performance mode | Exp 6 |
| **Meta-learning** | **Learned rules from CHEAP features** | Automatic descriptor selection | Exp 7 |

### TopoAgent Decision Logic (LEARNED, Not Hand-Written)

**Three learned models with confidence + fallback:**

```python
# Generated from exp7_meta_learning.py
# Example learned trees (from leave-one-dataset-out):

# MODEL 1: Best Descriptor (depth=3)
|--- otsu_components <= 15.5
|   |--- intensity_std <= 0.25
|   |   |--- class: persistence_statistics  # Few components, low contrast
|   |--- intensity_std > 0.25
|   |   |--- class: persistence_image  # Few components, high contrast
|--- otsu_components > 15.5
|   |--- edge_density <= 0.12
|   |   |--- class: betti_curves  # Many components, smooth
|   |--- edge_density > 0.12
|   |   |--- class: atol  # Many components, textured

# MODEL 2: Best Filtration (depth=2)
|--- intensity_mean <= 0.5
|   |--- class: superlevel  # Dark objects on light background (microscopy)
|--- intensity_mean > 0.5
|   |--- class: sublevel  # Light objects (CT, X-ray)

# MODEL 3: H0 vs H1 Dominance (depth=2)
|--- otsu_holes <= 3.0
|   |--- class: H0  # Few holes → component-dominated
|--- otsu_holes > 3.0
|   |--- class: H1  # Many holes → loop-dominated

# CONFIDENCE + FALLBACK
def topoagent_full_recommendation(characteristics):
    recs = topoagent_recommend(characteristics, models)

    if recs['descriptor']['confidence_level'] == 'low':
        return {
            'descriptor': recs['descriptor']['fallback'],  # persistence_statistics
            'filtration': recs['filtration']['prediction'],
            'homology': recs['homology']['prediction'],
            'note': 'Low descriptor confidence - using robust fallback'
        }
    return recs

SHAP Feature Importance (Descriptor Model):
  1. otsu_components: 0.32
  2. intensity_std: 0.28
  3. edge_density: 0.21
  4. n_samples: 0.12
  5. class_imbalance: 0.07
```

**Core Contribution:** Not just ranking descriptors, but **learning WHY and WHEN** each descriptor works best using:
- Leave-one-dataset-out validation
- Three learned models (descriptor, filtration, homology)
- Confidence scores + robust fallback strategy
