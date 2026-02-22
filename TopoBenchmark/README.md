# TopoBenchmark: Agent Evaluation Framework

TopoBenchmark evaluates whether TopoAgent can select the right TDA descriptor for each medical image dataset. It measures both **process quality** (descriptor selection accuracy) and **outcome quality** (mean balanced accuracy).

## Ground Truth

Ground truth is derived from benchmark4: exhaustive evaluation of **26 datasets x 15 descriptors x 6 classifiers**, identifying the optimal descriptor for each dataset.

## Metrics

| Metric | Description |
|--------|-------------|
| **MBA** (Mean Balanced Accuracy) | Average classification accuracy across datasets |
| **DSA** (Descriptor Selection Accuracy) | Fraction of datasets where the agent picks the best descriptor |
| **Regret** | Accuracy gap between the agent's choice and the oracle-optimal descriptor |

## Protocols

### Protocol 1: Descriptor Selection
Tests whether the agent selects the correct descriptor given dataset characteristics. Does not run classification — only evaluates the selection decision.

### Protocol 2: End-to-End Classification
Runs the full pipeline: image loading, PH computation, descriptor extraction, and classification. Measures both selection quality and downstream accuracy.

## Usage

```python
from TopoBenchmark import load_ground_truth, compute_mba, compute_dsa, compute_regret

# Load ground truth
gt = load_ground_truth()

# Evaluate agent predictions
mba = compute_mba(predictions, ground_truth=gt)
dsa = compute_dsa(predictions, ground_truth=gt)
regret = compute_regret(predictions, ground_truth=gt)
```

### Running Experiments

```bash
# Run full experiment across datasets
python TopoBenchmark/run_experiment.py --model gpt-4o --datasets all

# Run Protocol 1 only (descriptor selection)
python TopoBenchmark/run_protocol1.py --model gpt-4o

# Run Protocol 2 (end-to-end)
python TopoBenchmark/run_protocol2.py --model gpt-4o

# Analyze results
python TopoBenchmark/analyze_experiment.py --results-dir results/topobenchmark/experiment/

# Convergence analysis (how agent improves over time)
python TopoBenchmark/convergence_analysis.py --results-dir results/topobenchmark/convergence/
```

## Key Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Main experiment driver |
| `run_protocol1.py` | Protocol 1 (selection only) |
| `run_protocol2.py` | Protocol 2 (end-to-end) |
| `config.py` | Dataset descriptions and configurations |
| `ground_truth.py` | Ground truth from benchmark4 |
| `metrics.py` | MBA, DSA, Regret computation |
| `baselines.py` | Baseline methods (random, majority, etc.) |
| `analyze.py` | Result analysis |
| `convergence_analysis.py` | Convergence over time |
| `plot_convergence.py` | Convergence visualization |
