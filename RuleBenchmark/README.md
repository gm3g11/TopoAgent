# RuleBenchmark: Descriptor Rule Validation

RuleBenchmark validates the descriptor selection and parameter tuning rules used by TopoAgent. It consists of two major benchmarks.

## Benchmark3: Parameter Tuning Rules

Determines optimal parameters for each of the 15 descriptors across 5 morphological object types.

### Experiments

| Experiment | Description |
|-----------|-------------|
| exp1 | Main benchmark: all 15 descriptors across all object types |
| exp4 | Dimension and parameter studies (resolution, sigma, boundaries, color modes) |
| exp6 | CKA ensemble analysis |
| exp7 | Meta-learning for descriptor recommendation |

### Key Outputs

- **Optimal dimensions**: D* values for 75 entries (15 descriptors x 5 object types)
- **Parameter recommendations**: Resolution, sigma, weight functions per descriptor
- **Final rules**: `exp4_final_recommendations.json`

### Configuration

- **15 descriptors** (10 PH-based + 5 image-based)
- **5 object types**: discrete_cells, glands_lumens, vessel_trees, surface_lesions, organ_shape
- **Evaluation**: n=2000 samples, 5-fold stratified cross-validation

## Benchmark4: Large-Scale Evaluation

Exhaustive evaluation providing ground truth for TopoAgent's descriptor selection.

### Scale

- **26 medical image datasets** (MedMNIST + external)
- **15 descriptors** per dataset
- **6 classifiers**: XGBoost, Random Forest, SVM, KNN, Logistic Regression, TabPFN
- **Total**: 2,340 accuracy values (26 x 15 x 6)

### Key Files

| File | Description |
|------|-------------|
| `precompute_ph.py` | Precompute persistent homology for all datasets |
| `evaluate.py` | Run all descriptor x classifier combinations |
| `descriptor_runner.py` | Compute descriptors from PH diagrams |
| `classifier_wrapper.py` | Unified classifier interface |
| `analyze_results.py` | Generate summary statistics and rankings |
| `optimal_rules.py` | Extract optimal descriptor selection rules |
| `config.py` | Dataset and descriptor configuration |
| `data_loader.py` | Dataset loading utilities |

### Usage

```bash
# Step 1: Precompute persistent homology
python RuleBenchmark/benchmark4/precompute_ph.py --dataset BloodMNIST --n-samples 5000

# Step 2: Evaluate all descriptor-classifier combinations
python RuleBenchmark/benchmark4/evaluate.py --dataset BloodMNIST

# Step 3: Analyze results
python RuleBenchmark/benchmark4/analyze_results.py --results-dir results/benchmark4/

# Step 4: Extract selection rules
python RuleBenchmark/benchmark4/optimal_rules.py
```

### Results

Summary results are in `results/benchmark4/summary/`:
- `benchmark4_all_results.csv` — Full accuracy matrix
- `benchmark4_analysis.json` — Statistical analysis
- `benchmark4_summary.md` — Human-readable summary
- `descriptor_rankings_by_type.json` — Rankings by morphological type
- `classifier_rankings.json` — Classifier comparison
