# TopoAgent: Adaptive TDA Descriptor Selection for Medical Image Classification

TopoAgent is an intelligent medical AI agent that **adaptively selects the best topological descriptors and parameters** for medical image classification. Unlike fixed TDA pipelines, TopoAgent reasons about image characteristics and selects optimal vectorization strategies from a portfolio of 15 descriptors.

## Architecture

TopoAgent uses a 6-phase agentic pipeline where an LLM makes consequential decisions about descriptor selection:

```
OBSERVE (deterministic) ──> INTERPRET (LLM #1, vision)
    │                            │
    │ load image, compute PH     │ describe image morphology
    │ extract PH statistics      │ (blind to dataset identity)
    v                            v
ANALYZE (LLM #2) ──────> ACT (LLM #3, reconcile)
    │                        │
    │ form hypothesis from   │ choose final descriptor
    │ PH stats + math props  │ reconciling hypothesis
    v                        │ with benchmark evidence
EXTRACT (deterministic) <───┘
    │
    │ compute chosen descriptor
    │ + classifier prediction
    v
REFLECT (LLM #4)
    │
    │ assess feature quality
    │ write to long-term memory
    v
  result
```

**Key design principles:**
- **INTERPRET** is blind (no dataset names) and sees the actual image (multimodal)
- **ANALYZE** reasons from raw PH statistics and descriptor math properties
- **ACT** reconciles the LLM's hypothesis with benchmark evidence
- **REFLECT** checks feature quality and accumulates experience

## Descriptors

TopoAgent selects from 15 descriptors spanning PH-based and image-based methods:

| # | Descriptor | Type | Dimension |
|---|-----------|------|-----------|
| 1 | Persistence Image | PH-based | 800D |
| 2 | Persistence Landscapes | PH-based | 500D |
| 3 | Betti Curves | PH-based | 100D |
| 4 | Persistence Silhouette | PH-based | 100D |
| 5 | Persistence Entropy | PH-based | 200D |
| 6 | Persistence Statistics | PH-based | 20D |
| 7 | Tropical Coordinates | PH-based | 20D |
| 8 | Template Functions | PH-based | 100D |
| 9 | ATOL | PH-based | 50D |
| 10 | Persistence Codebook | PH-based | 50D |
| 11 | Minkowski Functionals | Image-based | 300D |
| 12 | Euler Characteristic Curve | Image-based | 200D |
| 13 | Euler Characteristic Transform | Image-based | 640D |
| 14 | Edge Histogram | Image-based | 80D |
| 15 | LBP Texture | Image-based | 52D |

These are evaluated across 5 morphological object types: discrete cells, glands/lumens, vessel trees, surface lesions, and organ shapes.

## Installation

```bash
# Clone the repository
git clone https://github.com/gm3g11/TopoAgent.git
cd TopoAgent

# Create conda environment
conda create -n topoagent python=3.10
conda activate topoagent

# Install dependencies
pip install -r requirements.txt
```

### API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

TopoAgent supports OpenAI (GPT-4o), Anthropic (Claude), and Google (Gemini) models.

## Quick Start

### Python API

```python
from topoagent import create_topoagent

# Create agent with GPT-4o
agent = create_topoagent(model_name="gpt-4o")

# Classify a medical image
result = agent.classify("path/to/dermoscopy_image.png")
print(f"Class: {result['classification']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Descriptor used: {result['descriptor_used']}")
```

### Command Line

```bash
# Classify a single image
python main.py --image path/to/image.png --model gpt-4o

# List available tools
python main.py --list-tools

# Interactive mode
python main.py --interactive
```

### End-to-End Demo

```bash
# Run the full pipeline on a dataset (e.g., DermaMNIST)
python scripts/demo_topoagent_e2e.py --dataset DermaMNIST --v9 --model gpt-4o

# Evaluate agentic behavior ratio
python scripts/eval_agentic_ratio.py --model gpt-4o --all
```

## Project Structure

```
TopoAgent/
├── topoagent/                  # Core agent package
│   ├── agent.py                # Main TopoAgent class
│   ├── workflow.py             # 6-phase agentic pipeline (v9)
│   ├── prompts.py              # LLM prompts for all phases
│   ├── state.py                # Agent state definition
│   ├── reflection.py           # Reflection engine
│   ├── core/                   # Core TDA features
│   ├── memory/                 # Dual memory system (short-term + long-term)
│   ├── skills/                 # Learned rules and descriptor selection skills
│   ├── tools/                  # 27 TDA tools organized by function
│   │   ├── preprocessing/      # Image loading, analysis
│   │   ├── filtration/         # Sublevel, superlevel, Vietoris-Rips
│   │   ├── homology/           # PH computation (GUDHI, Ripser)
│   │   ├── vectorization/      # All 15 descriptor implementations
│   │   ├── classification/     # PyTorch MLP classifier
│   │   └── ...                 # Advanced, texture, morphology tools
│   └── utils/
├── TopoBenchmark/              # Agent evaluation framework
│   ├── run_experiment.py       # Main experiment driver
│   ├── metrics.py              # MBA, DSA, Regret metrics
│   ├── ground_truth.py         # Ground truth from benchmark4
│   └── ...
├── RuleBenchmark/              # Descriptor rule validation
│   ├── benchmark3/             # Parameter tuning rules (15 desc x 5 types)
│   └── benchmark4/             # Large-scale eval (26 datasets x 15 desc x 6 clf)
├── scripts/                    # Demo, evaluation, and training scripts
│   ├── demo_topoagent_e2e.py   # Main end-to-end demo
│   ├── eval_agentic_ratio.py   # Agentic behavior evaluation
│   ├── tests/                  # Unit tests
│   └── demos/                  # Demo scripts
├── models/                     # Pre-trained classifier weights
├── results/                    # Experiment results and reports
├── main.py                     # CLI entry point
└── requirements.txt
```

## Key Results

### Agentic Behavior (10-dataset evaluation, GPT-4o)

| Metric | Value |
|--------|-------|
| Confirmed hypothesis | 40% |
| Switched to benchmark | 20% |
| Chose alternative | 40% |
| Differs from lookup table | 7/10 datasets |
| Unique descriptors used | 5+ |
| Object type accuracy (vision) | 9/10 |

### Ablation Study

| Condition | Accuracy | Delta |
|-----------|----------|-------|
| C0: Full pipeline | 57.9% | — |
| C1: No skills | 55.3% | -2.6% |
| C2: No memory | 58.1% | +0.2% |
| C3: No reflect | 56.5% | -1.4% |
| C4: No interpret | 56.2% | -1.7% |

Skills (learned rules from benchmarks) are the most impactful component.

## Evaluation

### TopoBenchmark

Evaluates descriptor selection accuracy and classification performance across 26 MedMNIST datasets:

```bash
# Run full experiment
python scripts/demo_topoagent_e2e.py --dataset DermaMNIST --v9 --model gpt-4o

# Run ablation study
python scripts/run_ablation_study.py --model gpt-4o
```

See [TopoBenchmark/README.md](TopoBenchmark/README.md) for details.

### RuleBenchmark

Validates descriptor selection rules from parameter tuning experiments:

```bash
# Results available in results/benchmark4/summary/
```

See [RuleBenchmark/README.md](RuleBenchmark/README.md) for details.

## Supported Datasets

TopoAgent has been evaluated on 26 medical image datasets from MedMNIST and other sources:

**MedMNIST (2D):** BloodMNIST, BreastMNIST, DermaMNIST, OCTMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST, PathMNIST, PneumoniaMNIST, RetinaMNIST, TissueMNIST

**External:** APTOS2019, BreakHis, Camelyon16, Chaoyang, CRC-HE, GlaS, ISIC2019, Kather, KidneyPAS, LC25000, MonuSeg, PCam, ROSE, SkinCancer, WBC

## Citation

```bibtex
@article{topoagent2026,
  title={TopoAgent: Adaptive Topological Descriptor Selection for Medical Image Classification},
  author={Meng, Guanqun},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
