# TopoAgent

**Medical AI Agent for Topological Data Analysis**

TopoAgent is a LangGraph-based AI agent that uses Topological Data Analysis (TDA) for medical image classification. It combines architectural patterns from MedRAX and EndoAgent to provide interpretable, topology-based medical image analysis.

## 📄 Paper

**TopoAgent: An Agentic Framework for Automated Topology Learning in Medical Imaging** (ECCV 2026)

TopoAgent automatically determines the most suitable topological descriptor for each medical image, operating through a **Perception–Reasoning–Action–Reflection (PRAR)** loop with **21 domain-specific tools** (6 perception + 15 descriptor), **dual memory**, and a **distilled skill set** — all without task-specific training. Companion benchmark **TopoBenchmark** spans 26 medical datasets (113,182 samples) across five object types: cells, glands/lumens, organ shapes, vessel trees, and surface lesions.

### Framework

![TopoAgent PRAR framework](figures/fig2_framework.png)

### Main Results

![Main balanced-accuracy results](figures/table1_main_results.png)

TopoAgent obtains **68.21%** average balanced accuracy, outperforming the strongest baseline by **9.32%** and general-purpose LLMs equipped with the same tools by **over 21%**.

Additional figures are in [`figures/`](figures/): motivation (`fig1`), skill set `S` (`fig3`), ablation study (`table2`), case studies (`fig4`), and downstream CNN/Transformer integration (`table3`).

## Architecture

TopoAgent combines:
- **MedRAX**: LangGraph StateGraph workflow + ReAct loop
- **EndoAgent**: Dual-memory mechanism (Ms, Ml) + reflection loop (max 3 rounds)
- **TDA Tools**: 15 specialized tools for topological feature extraction

```
┌─────────────────────────────────────────────────────────────────┐
│                     TopoAgent Workflow                          │
├─────────────────────────────────────────────────────────────────┤
│  Query + Image → Analyze → Select Tool → Execute → Update Ms   │
│                                                     ↓          │
│                    ← Loop (round < 3) ← Reflect → Update Ml    │
│                                                     ↓          │
│                              Generate Answer ← Check Complete  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone and navigate to project
cd /afs/crc.nd.edu/user/g/gmeng/Private/TopoAgent

# Activate conda environment
conda activate medrax

# Install additional dependencies (if needed)
pip install -r requirements.txt
```

## Quick Start

```python
from topoagent import create_topoagent

# Create agent
agent = create_topoagent(model_name="gpt-4o", max_rounds=3)

# Classify an image
result = agent.classify(
    image_path="path/to/dermoscopy.png",
    query="Classify this skin lesion using topological features"
)

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Tools used: {result['tools_used']}")
```

## Command Line Usage

```bash
# Run on single image
python main.py --image path/to/image.png --model gpt-4o

# Interactive mode
python main.py --interactive

# List available tools
python main.py --list-tools

# Batch processing
python scripts/run_agent.py --image-dir path/to/images/ --output results.json

# Evaluate on MedMNIST
python scripts/evaluate.py --dataset dermamnist --n-samples 100

# Run ablation study (with/without reflection)
python scripts/evaluate.py --dataset dermamnist --ablation

# Generate TopoQA benchmark
python scripts/generate_benchmark.py --output benchmark/topoqa_v1.json
```

## TDA Tools (15 total)

### Preprocessing (3)
| Tool | Description |
|------|-------------|
| `image_loader` | Load and normalize medical images (DICOM, PNG, JPEG) |
| `binarization` | Adaptive thresholding (Otsu, adaptive mean/gaussian) |
| `noise_filter` | Gaussian/median/bilateral filtering |

### Filtration (3)
| Tool | Description | Best For |
|------|-------------|----------|
| `sublevel_filtration` | Sublevel set filtration | Bright features (lesions, nodules) |
| `superlevel_filtration` | Superlevel set filtration | Dark features (vessels, cavities) |
| `cubical_complex` | Cubical complex for grids | 2D/3D structured images |

### Homology (3)
| Tool | Description |
|------|-------------|
| `compute_ph` | Compute persistent homology (H0, H1, H2) |
| `persistence_diagram` | Generate and analyze persistence diagrams |
| `persistence_image` | Convert PD to fixed-size vector representation |

### Features (3)
| Tool | Description |
|------|-------------|
| `topological_features` | Extract statistics (persistence, entropy, amplitude) |
| `wasserstein_distance` | Compare persistence diagrams (optimal transport) |
| `bottleneck_distance` | Compare diagrams (max matching cost) |

### Classification (3)
| Tool | Description |
|------|-------------|
| `knn_classifier` | k-Nearest Neighbors on topological features |
| `mlp_classifier` | Neural network classifier |
| `ensemble_classifier` | Combined prediction from multiple classifiers |

## Dual-Memory System

Following EndoAgent's design:

- **Short-term Memory (Ms)**: Recent tool executions in current session
  ```
  Ms = [(tool_1, output_1), (tool_2, output_2), ...]
  ```

- **Long-term Memory (Ml)**: Reflection experiences from past sessions
  ```
  Ml = [ReflectionEntry(round, error_analysis, suggestion, experience), ...]
  ```

Key insight from EndoAgent:
- Reflection alone: **+26.5% visual accuracy**
- Dual-memory: **+1.5% visual, +3.06% language accuracy**

## TopoQA Benchmark

Benchmark for evaluating TopoAgent with 5 task categories:

| Category | % | Description |
|----------|---|-------------|
| Method Selection | 20% | Choose appropriate TDA methods |
| Parameter Tuning | 20% | Optimize parameters |
| Topological Interpretation | 25% | Explain topological features |
| Multi-step Analysis | 25% | Complete classification pipeline |
| Error Recovery | 10% | Handle failure cases |

Datasets: DermaMNIST, PathMNIST, RetinaMNIST, PneumoniaMNIST

## Project Structure

```
TopoAgent/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── CLAUDE.md                   # Development notes
│
├── topoagent/
│   ├── __init__.py
│   ├── agent.py                # Main TopoAgent class
│   ├── state.py                # TopoAgentState definition
│   ├── workflow.py             # LangGraph workflow
│   ├── reflection.py           # Reflection mechanism
│   ├── prompts.py              # Prompt templates
│   │
│   ├── tools/                  # 15 TDA tools
│   │   ├── preprocessing/      # image_loader, binarization, noise_filter
│   │   ├── filtration/         # sublevel, superlevel, cubical
│   │   ├── homology/           # compute_ph, persistence_diagram, persistence_image
│   │   ├── features/           # topological_features, wasserstein, bottleneck
│   │   └── classification/     # knn, mlp, ensemble
│   │
│   ├── memory/                 # Dual-memory system
│   │   ├── short_term.py       # Ms: recent tool outputs
│   │   └── long_term.py        # Ml: reflection experiences
│   │
│   └── utils/
│
├── benchmark/
│   └── topoqa/                 # TopoQA benchmark
│       ├── templates.py        # Question templates
│       └── generator.py        # Benchmark generator
│
├── scripts/
│   ├── run_agent.py            # Run agent on images
│   ├── evaluate.py             # Evaluation script
│   └── generate_benchmark.py   # Generate TopoQA
│
└── tests/
```

## References

- **MedRAX**: LangGraph workflow pattern (`/afs/crc.nd.edu/user/g/gmeng/Private/MedRAX`)
- **EndoAgent**: Dual-memory + reflection (`/afs/crc.nd.edu/user/g/gmeng/Private/EndoAgent`)
- **TDA Libraries**: GUDHI, giotto-tda, ripser, persim

## License

Research use only.
