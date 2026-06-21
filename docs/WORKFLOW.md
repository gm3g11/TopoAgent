# TopoAgent Workflow Documentation

## Overview

TopoAgent is a medical AI agent that uses **Topological Data Analysis (TDA)** for medical image classification. It combines:

- **MedRAX**: LangGraph workflow structure
- **EndoAgent**: Dual-memory mechanism + reflection loop (+26.5% accuracy improvement)
- **TDA Tools**: 27 specialized topological analysis tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TOPOAGENT ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   Query     │     │   Image     │     │   LLM       │                   │
│  │   Input     │     │   Input     │     │  (GPT-4o)   │                   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│         │                   │                   │                           │
│         └─────────┬─────────┘                   │                           │
│                   ↓                             │                           │
│         ┌─────────────────┐                     │                           │
│         │  TopoAgent      │←────────────────────┘                           │
│         │  Workflow       │                                                 │
│         └────────┬────────┘                                                 │
│                  │                                                          │
│    ┌─────────────┼─────────────┐                                           │
│    ↓             ↓             ↓                                           │
│ ┌──────┐    ┌──────┐    ┌──────┐                                          │
│ │ Ms   │    │ Ml   │    │Tools │                                          │
│ │Short │    │Long  │    │ (27) │                                          │
│ │Term  │    │Term  │    │      │                                          │
│ │Memory│    │Memory│    │      │                                          │
│ └──────┘    └──────┘    └──────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Reasoning Workflow

### Main Loop (Algorithm 1 from EndoAgent)

```
for round t = 1 to N (max 3):
    tool_t ← SelectTool(context, Ms, Ml, Tools)
    output_t ← tool_t.invoke(context)
    Ms ← Ms ∪ {(tool_t, output_t)}           # Update short-term memory
    reflection_t ← LLM_reflection(context, Ms, Ml)
    Ml ← Ml ∪ {reflection_t}                  # Update long-term memory
    if IsTaskComplete(output_t, reflection_t):
        return GenerateFinalAnswer()
    context ← UpdateContext(context, output_t, reflection_t)
```

### Detailed Workflow Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START                                                                      │
│    │                                                                        │
│    ↓                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 1. ANALYZE QUERY                                                 │       │
│  │    - Parse user query                                            │       │
│  │    - Initialize state with image path                            │       │
│  │    - Set current_round = 1                                       │       │
│  └────────────────────────────┬────────────────────────────────────┘       │
│                               ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 2. SELECT TOOL                                                   │       │
│  │    - Format prompt with: query, image_path, Ms, Ml, tools        │       │
│  │    - LLM decides which tool to call                              │       │
│  │    - Returns tool_call with name and arguments                   │       │
│  └────────────────────────────┬────────────────────────────────────┘       │
│                               ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 3. EXECUTE TOOL                                                  │       │
│  │    - Invoke selected tool with arguments                         │       │
│  │    - Capture output (success/failure, data)                      │       │
│  │    - Store in _tool_outputs                                      │       │
│  └────────────────────────────┬────────────────────────────────────┘       │
│                               ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 4. UPDATE SHORT-TERM MEMORY                                      │       │
│  │    Ms = Ms ∪ {(tool_name, output)}                               │       │
│  │    - Accumulates tool outputs within session                     │       │
│  └────────────────────────────┬────────────────────────────────────┘       │
│                               ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 5. REFLECT (if enabled)                                          │       │
│  │    - Analyze: What went well? What could improve?                │       │
│  │    - Suggestion: What should be done next?                       │       │
│  │    - Experience: What lesson can be learned?                     │       │
│  └────────────────────────────┬────────────────────────────────────┘       │
│                               ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 6. UPDATE LONG-TERM MEMORY (if enabled)                          │       │
│  │    Ml = Ml ∪ {reflection_entry}                                  │       │
│  │    - Stores lessons for future rounds                            │       │
│  └────────────────────────────┬────────────────────────────────────┘       │
│                               ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ 7. CHECK COMPLETION                                              │       │
│  │    - Has enough information been gathered?                       │       │
│  │    - Has max_rounds been reached?                                │       │
│  │    → If YES: go to GENERATE ANSWER                               │       │
│  │    → If NO: increment round, go back to SELECT TOOL              │       │
│  └────────────────────────────┬────────────────────────────────────┘       │
│                               ↓                                             │
│                    ┌──────────┴──────────┐                                 │
│                    │                     │                                 │
│              [continue]             [finish]                               │
│                    │                     │                                 │
│                    ↓                     ↓                                 │
│            (back to step 2)    ┌─────────────────┐                        │
│                                │ 8. GENERATE     │                        │
│                                │    ANSWER       │                        │
│                                │  - Classification│                       │
│                                │  - Confidence   │                        │
│                                │  - Evidence     │                        │
│                                └────────┬────────┘                        │
│                                         ↓                                  │
│                                        END                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory System

### Short-Term Memory (Ms)

```python
# Session-scoped, cleared between images
short_term_memory: List[Tuple[str, Any]]

# Example:
[
    ("image_loader", {"success": True, "image_array": [...], "shape": (28, 28)}),
    ("compute_ph", {"H0": [...], "H1": [...]}),
    ("topological_features", {"persistence_entropy": 0.85, ...})
]
```

**Purpose**: Track what has been done in the current analysis session.

### Long-Term Memory (Ml)

```python
# Persistent across sessions, stores reflection experiences
long_term_memory: List[ReflectionEntry]

@dataclass
class ReflectionEntry:
    round: int
    error_analysis: str   # What went wrong
    suggestion: str       # What to do next
    experience: str       # Reusable lesson
```

**Purpose**: Learn from past mistakes and successes.

## Available Tools (27 Total)

### Preprocessing (3 tools)
| Tool | Description |
|------|-------------|
| `image_loader` | Load and normalize medical images (DICOM, PNG, JPEG) |
| `binarization` | Adaptive thresholding with Otsu method |
| `noise_filter` | Gaussian/median filtering for topology extraction |

### Filtration (3 tools)
| Tool | Description |
|------|-------------|
| `sublevel_filtration` | For bright features on dark background (lesions, nodules) |
| `superlevel_filtration` | For dark features on bright background (vessels, cavities) |
| `cubical_complex` | Build cubical complex for 2D/3D images |

### Homology (3 tools)
| Tool | Description |
|------|-------------|
| `compute_ph` | Compute persistent homology (H0, H1, H2) |
| `persistence_diagram` | Generate and visualize persistence diagrams |
| `persistence_image` | Convert PD to fixed-size vector representation |

### Features (3 tools)
| Tool | Description |
|------|-------------|
| `topological_features` | Extract statistics (persistence, entropy, amplitude) |
| `wasserstein_distance` | Compare persistence diagrams |
| `bottleneck_distance` | Alternative PD comparison metric |

### Classification (3 tools)
| Tool | Description |
|------|-------------|
| `knn_classifier` | k-Nearest Neighbors on topological features |
| `mlp_classifier` | Neural network classifier |
| `ensemble_classifier` | Combined prediction from multiple classifiers |

### Topology Descriptors (12 tools)
| Tool | Description |
|------|-------------|
| `euler_characteristic` | Compute χ = components - holes + cavities |
| `total_persistence_stats` | L^p total persistence and lifespan statistics |
| `betti_curves` | Betti curves β_k(t) for ML classifiers |
| `persistence_landscapes` | Hilbert space embedding for statistics |
| `persistence_silhouette` | Weighted sum of tent functions |
| `betti_ratios` | β₀/β₁, β₁/β₂ ratios for tissue analysis |
| `minkowski_functionals` | Area, perimeter, Euler for radiomics |
| `anisotropic_mf` | Directional analysis via fabric tensor |
| `fractal_dimension` | Box-counting fractal dimension |
| `lacunarity` | Texture heterogeneity measure |
| `weighted_ect` | Euler Characteristic Transform for 3D |
| `persistent_laplacian` | Spectral graph theory + persistent homology |

## Typical Tool Pipeline

```
Round 1: image_loader
    │
    ↓ (image array)
    │
Round 2: compute_ph OR sublevel_filtration
    │
    ↓ (persistence data)
    │
Round 3: topological_features OR knn_classifier
    │
    ↓ (features or classification)
    │
Final: Generate answer with evidence
```

## Output Format

```json
{
  "classification": "melanocytic nevi",
  "confidence": 85.0,
  "tools_used": ["image_loader", "compute_ph", "topological_features"],
  "reasoning_trace": [
    "Analyzing query: Classify this dermoscopy image",
    "Round 1: Selecting tool...",
    "Executed 1 tool(s), updated short-term memory",
    "Round 2: Selecting tool...",
    "Executed 1 tool(s), updated short-term memory",
    "Generated final answer"
  ],
  "evidence": [
    "H0 features: 15 connected components with avg persistence 0.3",
    "H1 features: 2 loops detected indicating boundary structure"
  ],
  "raw_answer": "### Classification\n1. **Class**: melanocytic nevi\n2. **Confidence**: 85%\n..."
}
```

## Experiment Design

### Experiment 1: Adaptive vs Fixed Pipelines

**Goal**: Compare TopoAgent's adaptive tool selection against fixed TDA pipelines.

| Method | Pipeline |
|--------|----------|
| Pipeline A | sublevel → PH → topological features |
| Pipeline B | superlevel → PH → Betti curves |
| Pipeline C | cubical complex → PH → persistence image |
| TopoAgent | LLM selects tools adaptively |

### Experiment 3: Ablation Study

**Goal**: Validate each component's contribution.

| Configuration | Description |
|---------------|-------------|
| Full | All components enabled (baseline) |
| No Reflection | Disable reflection step |
| No Short Memory | Clear memory between rounds |
| No Long Memory | No experience accumulation |
| 1 Round | Single pass only |
| 2 Rounds | Two rounds maximum |

## Key Design Decisions

### Why Max Rounds = 3?

From EndoAgent ablation study:
- 1 round: baseline
- 2 rounds: +15% improvement
- 3 rounds: +26.5% improvement (optimal)
- 4+ rounds: diminishing returns

### Why Dual Memory?

- **Short-term (Ms)**: Prevents redundant tool calls within a session
- **Long-term (Ml)**: Learns from failures to improve future analyses

### Why Reflection?

EndoAgent showed reflection adds **+26.5%** to visual task accuracy by:
1. Identifying what went wrong
2. Suggesting corrective actions
3. Building reusable experience

## File Structure

```
topoagent/
├── agent.py          # Main TopoAgent class
├── workflow.py       # LangGraph workflow definition
├── state.py          # State TypedDict definition
├── prompts.py        # LLM prompt templates
├── reflection.py     # Reflection engine
├── memory/           # Memory management
│   ├── short_term.py
│   └── long_term.py
└── tools/            # TDA tools organized by category
    ├── preprocessing/
    ├── filtration/
    ├── homology/
    ├── features/
    ├── classification/
    └── invariants/
```

## Usage

```python
from topoagent import create_topoagent

# Create agent
agent = create_topoagent(model_name="gpt-4o-mini", max_rounds=3)

# Classify an image
result = agent.classify(
    image_path="path/to/dermoscopy.png",
    query="Classify this skin lesion"
)

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']}%")
print(f"Tools used: {result['tools_used']}")
print(f"Reasoning: {result['reasoning_trace']}")
```

## Running Experiments

```bash
# Quick test (10 samples)
python experiments/exp1_adaptive_vs_fixed.py \
    --datasets dermamnist \
    --n-samples 10 \
    --seeds 42 \
    --openai-model gpt-4o-mini

# Full experiment (50 samples × 3 datasets × 3 seeds)
python experiments/exp1_adaptive_vs_fixed.py \
    --datasets dermamnist pathmnist bloodmnist \
    --n-samples 50 \
    --seeds 42 123 456 \
    --openai-model gpt-4o-mini
```
