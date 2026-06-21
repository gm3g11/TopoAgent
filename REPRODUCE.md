# Reproducing TopoAgent

End-to-end checklist from a fresh clone. There are two tiers:

- **A — Run the agent** on a single image: needs only an install + an LLM API key.
- **B — Reproduce the benchmark tables**: additionally needs the datasets and benchmark dependencies.

---

## A. Run the agent (minimal)

```bash
# 1. Clone + install
git clone https://github.com/gm3g11/TopoAgent.git
cd TopoAgent
pip install -r requirements.txt

# 2. Add an LLM API key
cp .env.example .env          # then edit .env:  OPENAI_API_KEY=sk-...

# 3. Get a test image (optional — any medical PNG works)
python -c "from medmnist import DermaMNIST; img,_=DermaMNIST(split='test',download=True,size=224)[0]; img.save('example.png')"

# 4. Run the v9 PRAR pipeline on it
python main.py --image example.png --model gpt-4o
```

Expected output: the determined topological descriptor, confidence, the tools used, and the
reasoning trace. Equivalent Python API:

```python
from topoagent import create_topoagent
agent = create_topoagent(model_name="gpt-4o", agentic_v9=True)
result = agent.classify(image_path="example.png", query="Determine the most suitable topology descriptor.")
print(result["descriptor"], result["confidence"])
```

Other backbones: `--model` with a Claude/Gemini model (set `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`),
or `create_topoagent_ollama` (local, no key — needs a running Ollama server).

---

## B. Reproduce the benchmark tables

### B1. Install benchmark dependencies

```bash
pip install -r requirements-benchmark.txt    # TabPFN, XGBoost, CatBoost, pytabkit, pandas, h5py
```

### B2. Get the datasets

```bash
# MedMNIST (11) — one command
python scripts/download_medmnist.py           # writes <flag>_224.npz into $MEDMNIST_PATH
```

The 15 external datasets must be downloaded from their sources — see
[docs/benchmark3_datasets.md → Downloading the Datasets](docs/benchmark3_datasets.md#downloading-the-datasets)
for per-dataset links and licenses. A few (MURA, APTOS2019, Chaoyang) require a free account or
data-use agreement.

### B3. Point the environment variables at your data

```bash
export MEDMNIST_PATH=~/.medmnist
export EXTERNAL_DATASETS_ROOT=/path/to/datasets
# per-dataset overrides if your layout differs: ISIC_PATH, KVASIR_PATH, DRIVE_ROOT
```

### B4. Run the reproductions

```bash
python scripts/run_llm_comparison.py     # Table 1 — main results vs. baselines
python scripts/run_ablation_study.py     # Table 2 — component / design ablations
python TopoBenchmark/run_experiment.py   # full TopoBenchmark evaluation
```

These compute persistent homology **live**, so they need only the datasets (B2/B3) plus an LLM API
key — not the (excluded) `results/` caches.

---

## Caveats

- **Gated datasets.** MURA, APTOS2019, and Chaoyang require a personal account / data-use
  agreement and cannot be auto-downloaded.
- **Protocol / oracle metrics.** `TopoBenchmark/run_protocol1.py` and `scripts/protocol2_*.py`
  read precomputed oracle assets (`results/benchmark4/raw/`, `results/topobenchmark/assets/`) that
  are **not** shipped. Regenerate them by running the full `RuleBenchmark/benchmark4` study
  (15 descriptors × 26 datasets × 6 classifiers), or obtain the asset archive from the authors.
- **Exact numbers** depend on the LLM backbone version and the `langchain`/`langgraph` versions;
  pin these in `requirements.txt` to your tested set for stable runs.
