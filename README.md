# TopoAgent

**An Agentic Framework for Automated Topology Learning in Medical Imaging** — ECCV 2026

TopoAgent is a LangGraph-based LLM agent that, given a raw medical image and a task prompt, automatically determines the most suitable **topological descriptor** (and its parameters) and produces a topological feature vector for downstream classification — all without task-specific training. It operates through a **Perception–Reasoning–Action–Reflection (PRAR)** loop backed by 21 domain-specific tools, dual memory, and a benchmark-distilled skill set.

## 📄 Paper

**TopoAgent: An Agentic Framework for Automated Topology Learning in Medical Imaging** (ECCV 2026)

TopoAgent automatically determines the most suitable topological descriptor for each medical image, operating through a **Perception–Reasoning–Action–Reflection (PRAR)** loop with **21 domain-specific tools** (6 perception + 15 descriptor), **dual memory**, and a **distilled skill set** — all without task-specific training. Companion benchmark **TopoBenchmark** spans 26 medical datasets (113,182 samples) across five object types: cells, glands/lumens, organ shapes, vessel trees, and surface lesions.

### Framework

![TopoAgent PRAR framework](figures/fig2_framework.png)

### Main Results

![Main balanced-accuracy results](figures/table1_main_results.png)

TopoAgent obtains **68.21%** average balanced accuracy, outperforming the strongest baseline by **9.32%** and general-purpose LLMs equipped with the same tools by **over 21%**.

Additional figures are in [`figures/`](figures/): motivation (`fig1`), skill set `S` (`fig3`), ablation study (`table2`), case studies (`fig4`), and downstream CNN/Transformer integration (`table3`).

## Why TopoAgent

Persistent homology (PH) captures structural properties — connected components, loops, shape — that pixel-level deep learning often neglects. Many *topological descriptors* convert persistence diagrams into fixed-length feature vectors, but **no single descriptor is best across datasets**: choosing one demands TDA expertise and trial-and-error. TopoAgent automates this per-image determination with full reasoning traces.

## Method: the PRAR loop

- **Perception** — six perception tools compute a PH profile `h(I)` (birth–death counts per homology dimension, average persistence, Betti ratio `β1/β0`), visual statistics `v(I)` (SNR, contrast, edge density), and identify the object type `o ∈ {cells, glands/lumens, organ shapes, vessel trees, surface lesions}` by jointly reasoning over the raw image and its PH.
- **Reasoning** — two steps with *asymmetric information access*. A **proposal** step sees descriptor properties `Sprop` and only *stripped* reasoning patterns (no rankings) to avoid anchoring bias; a **determination** step then integrates the full rankings `Srank` and long-term memory `Ml` to finalize the descriptor `d*` and parameters `θ*`.
- **Action** — runs the determined descriptor tool to produce a feature vector `f ∈ R^{n_d}`, using parameters validated during skill-set construction.
- **Reflection** — the LLM validates `f` against quality criteria (sparsity, variance, kurtosis, skewness, dynamic range, informative-feature ratio). On failure it records a diagnosis in `Ml` and retries (≤ 2 retries within a time budget), else falls back to the top-ranked descriptor for `o`.

## Tools (21)

**6 perception:** `image_loader`, `image_analyzer`, `noise_filter`, `compute_ph` (GUDHI cubical-complex filtration), `topological_features` (PH profile), `betti_ratios`.

**15 descriptors** — 10 PH-derived: `persistence_image`, `persistence_landscapes`, `persistence_silhouette`, `persistence_entropy`, `persistence_statistics`, `betti_curves`, `template_functions`, `atol`, `persistence_codebook`, `tropical_coordinates`; and 5 image-based: `minkowski_functionals`, `euler_characteristic_curve`, `euler_characteristic_transform`, `lbp_texture`, `edge_histogram`.

> Note: `topoagent/tools/__init__.py` registers a larger superset of tools (including legacy classifiers and filtration variants from earlier versions); the PRAR pipeline uses the 21-tool subset above. The current paper pipeline is **v9** (`agentic_v9=True`); v2–v8 remain in the code for reference.

## Skill set S

Distilled offline from [`RuleBenchmark/`](RuleBenchmark/) (15 descriptors × 26 datasets × 6 classifiers), organized at the object-type level to prevent dataset leakage. Encoded in [`topoagent/skills/`](topoagent/skills/):

- **`Sprop`** — descriptor mathematical properties, strengths/weaknesses, and parameter heuristics.
- **`Srank`** — per-object-type tiered rankings, reasoning chains, and threshold-based PH signal rules.
- **`Sparam`** — validated parameters for all descriptor × object-type combinations.

## Dual memory

- **Short-term `Ms`** — tool invocations within a single run (the LLM's context window).
- **Long-term `Ml`** — diagnostic entries across runs *within a dataset* (failed descriptor, diagnosed cause, successful correction); reset between datasets to prevent cross-dataset leakage. Seeds live in `topoagent/memory/*.json` (empty by default).

## TopoBenchmark

A frozen benchmark of **113,182 samples** from **26** public 2D medical datasets across five object types, with convergence-based per-dataset sizing, precomputed PH caches, and fixed fold indices. Construction and evaluation harness in [`TopoBenchmark/`](TopoBenchmark/).

## Installation

```bash
git clone https://github.com/gm3g11/TopoAgent.git
cd TopoAgent
pip install -r requirements.txt

# API keys (see .env.example)
cp .env.example .env   # then add OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
```

## Quick start

```python
from topoagent import create_topoagent

agent = create_topoagent(model_name="gpt-4o", agentic_v9=True)

result = agent.classify(
    image_path="path/to/image.png",
    query="Analyze this medical image, compute its persistent homology, "
          "and determine the most suitable topology descriptor.",
)

print("Descriptor determined:", result["descriptor"])
print("Confidence:", result["confidence"])
print("Tools used:", result["tools_used"])
print("Reasoning trace:", result["reasoning_trace"])
```

Other backbones: `create_topoagent_claude`, `create_topoagent_gemini`, `create_topoagent_ollama`.

## Command line

```bash
python main.py --image path/to/image.png --model gpt-4o   # single image
python main.py --interactive                               # interactive mode
python main.py --list-tools                                # list available tools
```

## Reproducing the paper

```bash
python scripts/run_llm_comparison.py     # Table 1: main results vs. baselines
python scripts/run_ablation_study.py     # Table 2: component/design ablations
python TopoBenchmark/run_experiment.py   # TopoBenchmark evaluation
```

## Datasets

MedMNIST plus 15 external datasets. Paths are read from environment variables (defaults defined in `RuleBenchmark/benchmark{3,4}/config.py`); see [`docs/benchmark3_datasets.md`](docs/benchmark3_datasets.md) for the full roster.

```bash
export MEDMNIST_PATH=~/.medmnist
export EXTERNAL_DATASETS_ROOT=/path/to/datasets
# optional overrides: ISIC_PATH, KVASIR_PATH, DRIVE_ROOT, CUPH_PATH (GPU PH)
```

## Repository structure

```
TopoAgent/
├── topoagent/            # Core PRAR agent
│   ├── agent.py          # TopoAgent class + create_topoagent* factories
│   ├── workflow.py       # LangGraph PRAR workflow (v9 = paper pipeline)
│   ├── state.py, reflection.py, prompts.py
│   ├── tools/            # 21 PRAR tools (perception + descriptors) + registry
│   ├── skills/           # Skill set S: Sprop / Srank / Sparam (rules_data.py)
│   └── memory/           # Dual memory (short_term, long_term)
├── TopoBenchmark/        # Frozen benchmark + evaluation harness (26 datasets)
├── RuleBenchmark/        # Skill-set distillation study (benchmark3/4/5)
├── baselines/            # Fixed-descriptor & general-LLM baselines
├── scripts/              # Reproduction & analysis scripts
├── docs/                 # Architecture, tools, and workflow documentation
├── figures/              # Paper figures
├── main.py               # CLI entry point
└── requirements.txt
```

## Citation

```bibtex
@inproceedings{meng2026topoagent,
  title     = {TopoAgent: An Agentic Framework for Automated Topology Learning in Medical Imaging},
  author    = {Meng, Guangyu and Gu, Pengfei and Li, Xueyang and Shi, Yiyu and Chambers, Erin Wolf and Chen, Danny Z.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgments

Built on [LangGraph](https://github.com/langchain-ai/langgraph); persistent homology via [GUDHI](https://gudhi.inria.fr/). The PRAR architecture and dual-memory design draw on MedRAX and EndoAgent. TDA libraries: GUDHI, giotto-tda, persim.

## License

See [LICENSE](LICENSE).
