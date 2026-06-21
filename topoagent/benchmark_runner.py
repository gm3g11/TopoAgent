"""TopoBenchmark Runner — Per-Dataset Descriptor Selection & Evaluation.

Makes 1 LLM call per dataset (not per image). Evaluates descriptor selection
quality via downstream classification accuracy.

Supports two modes:
  - Live mode: Extract features + run CV (slow, requires GPU/data)
  - Lookup mode: Look up pre-computed Benchmark4 results (fast, ~30 seconds)

Usage:
    from topoagent.benchmark_runner import TopoBenchmarkRunner

    # Lookup mode (fast)
    runner = TopoBenchmarkRunner(model, lookup_mode=True)
    result = runner.evaluate_dataset("BloodMNIST", method="topoagent")

    # Live mode (original)
    runner = TopoBenchmarkRunner(model)
    result = runner.evaluate_dataset("BloodMNIST", method="topoagent", n_eval=1000)

Methods:
    - "topoagent": Full skill knowledge + learned rules (our method)
    - "topoagent_no_skills": LLM + sample stats, no expert rules
    - "zeroshot": GPT-4 zero-shot baseline (no skill knowledge)
    - "fewshot": GPT-4 few-shot baseline (3 examples from other datasets)
    - "meta_learner": GBR meta-learner trained on cheap dataset features
    - "rule_based": 4 hand-crafted rules from exp7
    - "sba": Single best algorithm (persistence_statistics)
    - "fixed_<descriptor>": Fixed descriptor baseline
    - "oracle": Best descriptor per dataset (upper bound)
    - "random": Random descriptor selection (lower bound)
"""

import json
import sys
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from .state import LLMInteraction, truncate_output_for_prompt
from .prompts import (
    BENCHMARK_SELECT_PROMPT,
    BENCHMARK_SELECT_ZERO_PROMPT,
    BASELINE_ZEROSHOT_PROMPT,
    BASELINE_FEWSHOT_PROMPT,
    BASELINE_NOSKILLS_PROMPT,
    PREDICT_DESCRIPTORS_PROMPT,
    VERIFY_PREDICTION_PROMPT,
    PORTFOLIO_SELECT_PROMPT,
    PORTFOLIO_REFLECT_PROMPT,
)
from .skills import SkillRegistry
from .skills.rules_data import (
    SUPPORTED_DESCRIPTORS,
    ALL_DESCRIPTORS,
    TOP_PERFORMERS,
    BENCHMARK3_TOP_PERFORMERS,
    OBJECT_TYPES,
    DESCRIPTOR_PROPERTIES,
    get_optimal_params,
    get_descriptor_dim,
    build_descriptor_knowledge_text,
    build_complementarity_text,
    get_pairwise_rho,
)
from .memory.skill_memory import SkillMemory


# All 15 descriptors (extends SUPPORTED_DESCRIPTORS with ATOL + persistence_codebook)
EXTENDED_DESCRIPTORS = [
    "persistence_statistics", "persistence_image", "persistence_landscapes",
    "persistence_silhouette", "betti_curves", "persistence_entropy",
    "persistence_codebook", "tropical_coordinates", "ATOL",
    "template_functions", "minkowski_functionals", "euler_characteristic_curve",
    "euler_characteristic_transform", "lbp_texture", "edge_histogram",
]

# Image-based descriptors that skip PH
IMAGE_BASED_DESCRIPTORS = {
    "minkowski_functionals",
    "euler_characteristic_curve",
    "euler_characteristic_transform",
    "edge_histogram",
    "lbp_texture",
}

# Pre-computed assets directory
ASSETS_DIR = Path(__file__).parent.parent / "results" / "topobenchmark" / "assets"

# No-LLM methods (don't require a model)
NO_LLM_METHODS = {"oracle", "oracle_pair", "random", "random_pair",
                   "meta_learner", "meta_learner_ridge",
                   "meta_learner_train_test", "meta_learner_ridge_train_test",
                   "rule_based", "sba", "object_type_default"}

# Train/test split: 5 Benchmark3 datasets for training, 21 for testing
TRAIN_DATASETS = ["BloodMNIST", "PathMNIST", "RetinaMNIST", "DermaMNIST", "OrganAMNIST"]
TEST_DATASETS = [
    # discrete_cells (5)
    "TissueMNIST", "MalariaCell", "PCam", "SIPaKMeD", "AML_Cytomorphology",
    # glands_lumens (5)
    "Kvasir", "BreakHis", "NCT_CRC_HE", "LC25000", "Chaoyang",
    # organ_shape (7)
    "OCTMNIST", "PneumoniaMNIST", "BreastMNIST", "OrganCMNIST", "OrganSMNIST",
    "BrainTumorMRI", "MURA",
    # vessel_trees (2)
    "IDRiD", "APTOS2019",
    # surface_lesions (2)
    "ISIC2019", "GasHisSDB",
]


def _ensure_benchmark4_path():
    """Add benchmark4 to sys.path if needed."""
    project_root = Path(__file__).parent.parent
    b4_dir = project_root / "RuleBenchmark" / "benchmark4"
    if str(b4_dir) not in sys.path:
        sys.path.insert(0, str(b4_dir))
    return b4_dir


class TopoBenchmarkRunner:
    """Benchmark runner: per-dataset descriptor selection and evaluation.

    Supports live mode (extract+classify) and lookup mode (pre-computed results).
    """

    def __init__(self, model=None, skill_memory: Optional[SkillMemory] = None,
                 lookup_mode: bool = False, experiment_mode: str = "train_test"):
        """Initialize benchmark runner.

        Args:
            model: LangChain chat model (required for LLM methods, optional for no-LLM methods)
            skill_memory: Optional SkillMemory instance (creates default if None)
            lookup_mode: If True, use pre-computed benchmark4 results instead of live eval
            experiment_mode: "train_test" (main, no circularity) or "lodo" (supplementary)
                - train_test: TopoAgent uses Benchmark3 rankings only (5 train datasets).
                  Meta-learner trained on Benchmark3 datasets with descriptor ID features.
                  Object-type default uses Benchmark3 top-1 per type. No benchmark4 data leaks.
                - lodo: Original LODO design (supplementary analysis). All methods can
                  use benchmark4 rankings with held-out masking.
        """
        self.model = model
        self.skill_registry = SkillRegistry()
        self.skill_memory = skill_memory or SkillMemory()
        self.lookup_mode = lookup_mode
        self.experiment_mode = experiment_mode
        self._portfolio_memory = None  # Lazy-loaded

        # Pre-computed assets (loaded lazily or on init in lookup mode)
        self._accuracy_lookup = None
        self._oracle_data = None
        self._sba_data = None
        self._cheap_features = None
        self._top_performers_full = None
        self._meta_learner_lodo = None
        self._meta_learner_ridge_lodo = None
        self._meta_learner_train_test = None
        self._meta_learner_ridge_train_test = None
        self._rule_based_predictions = None
        self._fusion_accuracy_lookup = None

        # LODO state: active overrides (only used in lodo mode)
        self._lodo_top_performers = None  # Overrides TOP_PERFORMERS when set
        self._lodo_dataset = None

        if lookup_mode:
            self._load_assets()

    # -------------------------------------------------------------------------
    # Asset Loading
    # -------------------------------------------------------------------------

    def _load_assets(self):
        """Load pre-computed benchmark assets."""
        if not ASSETS_DIR.exists():
            raise FileNotFoundError(
                f"Assets directory not found: {ASSETS_DIR}\n"
                f"Run: python scripts/precompute_benchmark_assets.py"
            )

        def _load_json(name):
            path = ASSETS_DIR / name
            if not path.exists():
                raise FileNotFoundError(f"Missing asset: {path}")
            with open(path, "r") as f:
                return json.load(f)

        self._accuracy_lookup = _load_json("accuracy_lookup.json")
        _oracle_raw = _load_json("oracle.json")
        # oracle.json may be nested under "per_dataset" key — unwrap it
        self._oracle_data = _oracle_raw.get("per_dataset", _oracle_raw)
        self._sba_data = _load_json("sba.json")
        self._top_performers_full = _load_json("top_performers.json")
        self._meta_learner_lodo = _load_json("meta_learner_lodo.json")
        self._rule_based_predictions = _load_json("rule_based_predictions.json")

        # Optional: ridge meta-learner (may not exist yet)
        try:
            self._meta_learner_ridge_lodo = _load_json("meta_learner_ridge_lodo.json")
        except FileNotFoundError:
            self._meta_learner_ridge_lodo = {}

        # Train/test meta-learner (trained on 5 Benchmark3 datasets with descriptor ID)
        try:
            self._meta_learner_train_test = _load_json("meta_learner_train_test.json")
        except FileNotFoundError:
            self._meta_learner_train_test = {}
        try:
            self._meta_learner_ridge_train_test = _load_json("meta_learner_ridge_train_test.json")
        except FileNotFoundError:
            self._meta_learner_ridge_train_test = {}

        # Cheap features: optional (only needed for LLM methods in lookup mode)
        try:
            self._cheap_features = _load_json("cheap_features.json")
        except FileNotFoundError:
            self._cheap_features = {}

        # Fusion accuracy lookup: optional (needed for portfolio methods)
        try:
            self._fusion_accuracy_lookup = _load_json("fusion_accuracy_lookup.json")
        except FileNotFoundError:
            self._fusion_accuracy_lookup = {}

        fusion_count = len(self._fusion_accuracy_lookup) if self._fusion_accuracy_lookup else 0
        print(f"Loaded benchmark assets: {len(self._accuracy_lookup)} datasets, "
              f"{sum(len(v) for v in self._accuracy_lookup.values())} accuracy entries, "
              f"{fusion_count} fusion entries")

    def _get_active_top_performers(self):
        """Get top performers based on experiment mode.

        - train_test mode: Uses BENCHMARK3_TOP_PERFORMERS (5-dataset pilot).
          No Benchmark4 data leaks. This is the only knowledge source.
        - lodo mode: Uses LODO-masked Benchmark4 rankings if active,
          else full Benchmark4 (TOP_PERFORMERS).
        """
        if self.experiment_mode == "train_test":
            # Strict: only Benchmark3 knowledge, no Benchmark4 rankings
            return BENCHMARK3_TOP_PERFORMERS
        # LODO mode
        if self._lodo_top_performers is not None:
            return self._lodo_top_performers
        if self._top_performers_full is not None:
            return self._top_performers_full
        return TOP_PERFORMERS

    # -------------------------------------------------------------------------
    # LODO Support
    # -------------------------------------------------------------------------

    def _apply_lodo_mask(self, dataset_name: str):
        """Load LODO-specific assets for the held-out dataset."""
        lodo_path = ASSETS_DIR / "top_performers_lodo" / f"{dataset_name}.json"
        if lodo_path.exists():
            with open(lodo_path, "r") as f:
                self._lodo_top_performers = json.load(f)
            self._lodo_dataset = dataset_name
        else:
            print(f"  WARNING: No LODO file for {dataset_name}, using full rankings")
            self._lodo_top_performers = None
            self._lodo_dataset = None

    def _restore_full_assets(self):
        """Restore full (non-LODO) assets."""
        self._lodo_top_performers = None
        self._lodo_dataset = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def evaluate_dataset(
        self,
        dataset_name: str,
        method: str = "topoagent",
        n_eval: int = 1000,
        n_context: int = 5,
        cv_folds: int = 5,
        fewshot_holdout: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Evaluate one method on one dataset.

        In lookup mode, accuracy is looked up from pre-computed benchmark4 results.
        In live mode, features are extracted and CV is run.
        """
        start_time = time.time()

        # A1: Build dataset context
        # In lookup mode for non-LLM methods, skip heavy context building
        needs_llm = method not in NO_LLM_METHODS and not method.startswith("fixed_")
        if self.lookup_mode and not needs_llm:
            context = self._build_dataset_context_light(dataset_name)
        else:
            context = self._build_dataset_context(dataset_name, n_context, seed)

        # A2: Select descriptor
        selection = self._select_descriptor(
            context, method, fewshot_holdout=fewshot_holdout or [dataset_name],
        )

        # Portfolio methods return a list of descriptors; single methods return one
        portfolio = selection.get("portfolio")
        descriptor = selection["descriptor_choice"]  # Always set (primary)
        object_type = context["object_type"]
        color_mode = context["color_mode"]

        # A3 + A4: Get accuracy (lookup or live)
        is_portfolio = portfolio and len(portfolio) > 1
        if is_portfolio and self.lookup_mode:
            # Fusion accuracy lookup
            accuracy, eval_details = self._lookup_fusion_accuracy(
                dataset_name, portfolio,
            )
        elif self.lookup_mode:
            accuracy, eval_details = self._lookup_accuracy(dataset_name, descriptor)
        else:
            accuracy, eval_details = self._batch_extract_and_evaluate(
                dataset_name=dataset_name,
                descriptor=descriptor,
                object_type=object_type,
                color_mode=color_mode,
                n_eval=n_eval,
                cv_folds=cv_folds,
                seed=seed,
            )

        # Compute oracle for metrics
        oracle = self._get_oracle_for_dataset(dataset_name) if self.lookup_mode else self._get_oracle(object_type)

        elapsed = time.time() - start_time

        # For portfolio methods, descriptor_selected shows the portfolio
        descriptor_label = "+".join(portfolio) if is_portfolio else descriptor

        result = {
            "dataset": dataset_name,
            "method": method,
            "descriptor_selected": descriptor_label,
            "portfolio": portfolio if is_portfolio else [descriptor],
            "is_portfolio": is_portfolio,
            "object_type": object_type,
            "color_mode": color_mode,
            "accuracy": accuracy,
            "oracle_descriptor": oracle["descriptor"],
            "oracle_accuracy": oracle["accuracy"],
            "regret": oracle["accuracy"] - accuracy,
            "top1_match": descriptor == oracle["descriptor"],
            "top3_match": descriptor in self._get_top3_descriptors(object_type),
            "reasoning": selection.get("descriptor_rationale") or selection.get("reasoning", ""),
            "reasoning_chain": selection.get("reasoning_chain", ""),
            "n_eval": n_eval if not self.lookup_mode else "lookup",
            "cv_folds": cv_folds if not self.lookup_mode else "lookup",
            "feature_dim": eval_details.get("feature_dim", 0),
            "llm_calls": selection.get("_llm_calls", 0 if (method in NO_LLM_METHODS or method.startswith("fixed_")) else 1),
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
            "llm_interaction": selection.get("_interaction"),
            "llm_interactions": selection.get("_interactions", []),
            "fusion_strategy": selection.get("fusion_strategy"),
            "lookup_mode": self.lookup_mode,
        }

        # A5: Learn (only for topoagent methods)
        if method == "topoagent":
            self._learn(dataset_name, result)
        elif method == "topoagent_portfolio":
            self._learn_portfolio(dataset_name, result, selection)

        return result

    def evaluate_all(
        self,
        datasets: List[str],
        methods: Optional[List[str]] = None,
        n_eval: int = 1000,
        cv_folds: int = 5,
        seed: int = 42,
        lodo: bool = False,
    ) -> Dict[str, Any]:
        """Run all methods on all datasets.

        Args:
            datasets: List of dataset names
            methods: Methods to evaluate (default: all)
            n_eval: Number of evaluation images per dataset
            cv_folds: Number of CV folds
            seed: Random seed
            lodo: If True, apply LODO masking for each dataset

        Returns:
            Comprehensive results dict with per-dataset and summary metrics
        """
        if methods is None:
            if self.experiment_mode == "train_test":
                methods = [
                    "oracle", "topoagent", "topoagent_zero", "zeroshot",
                    "meta_learner_ridge_train_test", "object_type_default",
                    "sba", "random",
                ]
            else:
                methods = ["topoagent", "topoagent_no_skills", "meta_learner",
                           "rule_based", "sba", "zeroshot", "oracle"]

        all_results = []
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}")
            print(f"{'='*60}")

            for method in methods:
                if lodo:
                    self._apply_lodo_mask(dataset)

                print(f"  Method: {method}...", end=" ", flush=True)
                try:
                    result = self.evaluate_dataset(
                        dataset_name=dataset,
                        method=method,
                        n_eval=n_eval,
                        cv_folds=cv_folds,
                        fewshot_holdout=[dataset],
                        seed=seed,
                    )
                    print(f"acc={result['accuracy']:.1%}, desc={result['descriptor_selected']}, "
                          f"regret={result['regret']:.3f}")
                    all_results.append(result)
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results.append({
                        "dataset": dataset,
                        "method": method,
                        "error": str(e),
                        "accuracy": 0.0,
                    })

                if lodo:
                    self._restore_full_assets()

        summary = self._compute_summary(all_results)

        return {
            "config": {
                "datasets": datasets,
                "methods": methods,
                "n_eval": n_eval if not self.lookup_mode else "lookup",
                "cv_folds": cv_folds if not self.lookup_mode else "lookup",
                "seed": seed,
                "lodo": lodo,
                "lookup_mode": self.lookup_mode,
                "timestamp": datetime.now().isoformat(),
            },
            "results": all_results,
            "summary": summary,
        }

    # -------------------------------------------------------------------------
    # A1: SAMPLE & CONTEXT
    # -------------------------------------------------------------------------

    def _build_dataset_context_light(self, dataset_name: str) -> Dict[str, Any]:
        """Build lightweight context without loading images (for lookup mode non-LLM methods)."""
        try:
            import importlib.util
            topo_cfg_path = Path(__file__).resolve().parent.parent / "TopoBenchmark" / "config.py"
            spec = importlib.util.spec_from_file_location("topobenchmark_config", str(topo_cfg_path))
            topo_cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(topo_cfg)
            desc = topo_cfg.DATASET_DESCRIPTIONS.get(dataset_name, {})
        except Exception:
            desc = {}

        object_type = desc.get("object_type", "surface_lesions")
        color_mode = desc.get("color_mode", "grayscale")

        desc_text = ""
        if desc:
            desc_text = (
                f"Domain: {desc.get('domain', 'unknown')}\n"
                f"Description: {desc.get('description', 'N/A')}\n"
                f"What matters: {desc.get('what_matters', 'N/A')}\n"
                f"Number of classes: {desc.get('n_classes', '?')}\n"
                f"Color mode: {color_mode}\n"
                f"Object type: {object_type}\n"
            )

        # Load cheap features from assets
        cheap_features = self._get_cheap_features_text(dataset_name)

        return {
            "dataset_name": dataset_name,
            "description": desc_text,
            "object_type": object_type,
            "color_mode": color_mode,
            "n_classes": desc.get("n_classes", 0),
            "sample_stats": "(lookup mode — no sample stats computed)",
            "cheap_features": cheap_features,
        }

    def _build_dataset_context(
        self,
        dataset_name: str,
        n_samples: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Load sample images, compute PH stats, build context for LLM."""
        _ensure_benchmark4_path()
        from data_loader import load_dataset

        # Load a small batch
        images, labels, class_names = load_dataset(dataset_name, n_samples=n_samples, seed=seed)

        # Get dataset description (use importlib to avoid config name collision)
        try:
            import importlib.util
            topo_cfg_path = Path(__file__).resolve().parent.parent / "TopoBenchmark" / "config.py"
            spec = importlib.util.spec_from_file_location("topobenchmark_config", str(topo_cfg_path))
            topo_cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(topo_cfg)
            desc = topo_cfg.DATASET_DESCRIPTIONS.get(dataset_name, {})
        except Exception:
            desc = {}

        object_type = desc.get("object_type", "surface_lesions")
        color_mode = desc.get("color_mode", "grayscale")

        # Compute sample statistics
        sample_stats = self._compute_sample_stats(images, labels, color_mode)

        # Build description string
        desc_text = ""
        if desc:
            desc_text = (
                f"Domain: {desc.get('domain', 'unknown')}\n"
                f"Description: {desc.get('description', 'N/A')}\n"
                f"What matters: {desc.get('what_matters', 'N/A')}\n"
                f"Number of classes: {desc.get('n_classes', '?')}\n"
                f"Color mode: {color_mode}\n"
                f"Object type: {object_type}\n"
            )
        else:
            desc_text = (
                f"Dataset: {dataset_name}\n"
                f"Images: {len(images)}, shape={images[0].shape}\n"
                f"Classes: {len(class_names)}\n"
            )

        # Get cheap features text
        cheap_features = self._get_cheap_features_text(dataset_name)

        return {
            "dataset_name": dataset_name,
            "description": desc_text,
            "object_type": object_type,
            "color_mode": color_mode,
            "n_classes": desc.get("n_classes") or len(class_names),
            "sample_stats": sample_stats,
            "cheap_features": cheap_features,
            "images": images,
            "labels": labels,
        }

    def _get_cheap_features_text(self, dataset_name: str) -> str:
        """Format cheap features as text for LLM prompt."""
        if self._cheap_features and dataset_name in self._cheap_features:
            feat = self._cheap_features[dataset_name]
            if feat is None:
                return "(features not available for this dataset)"
            lines = []
            for k, v in feat.items():
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            return "\n".join(lines)
        return "(no pre-computed features available)"

    def _compute_sample_stats(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        color_mode: str,
    ) -> str:
        """Compute cheap statistics from sample images for LLM context."""
        lines = []
        lines.append(f"Image shape: {images[0].shape}, dtype: {images[0].dtype}")
        lines.append(f"Intensity: mean={images.mean():.3f}, std={images.std():.3f}, "
                      f"min={images.min():.3f}, max={images.max():.3f}")
        lines.append(f"Labels distribution: {np.bincount(labels.astype(int)).tolist()}")

        # Compute quick PH on grayscale versions
        try:
            _ensure_benchmark4_path()
            from precompute_ph import compute_ph

            # Convert to grayscale if needed
            if images.ndim == 4 and images.shape[3] == 3:
                gray = images.mean(axis=3)
            else:
                gray = images

            diagrams, ph_lib = compute_ph(gray, n_jobs=2)

            h0_counts = []
            h1_counts = []
            for diag in diagrams:
                h0 = diag.get("H0")
                h1 = diag.get("H1")
                h0_counts.append(len(h0) if h0 is not None else 0)
                h1_counts.append(len(h1) if h1 is not None else 0)

            lines.append(f"PH statistics (sublevel, {len(diagrams)} samples):")
            lines.append(f"  H0 (components): mean={np.mean(h0_counts):.0f}, "
                          f"range=[{min(h0_counts)}, {max(h0_counts)}]")
            lines.append(f"  H1 (loops): mean={np.mean(h1_counts):.0f}, "
                          f"range=[{min(h1_counts)}, {max(h1_counts)}]")

            # H0/H1 ratio
            h0_mean = np.mean(h0_counts)
            h1_mean = np.mean(h1_counts)
            if h1_mean > 0:
                ratio = h0_mean / h1_mean
                lines.append(f"  H0/H1 ratio: {ratio:.1f} ({'H0-dominant' if ratio > 5 else 'mixed'})")
            else:
                lines.append(f"  H0/H1 ratio: inf (purely H0-dominant, no loops)")

        except Exception as e:
            lines.append(f"PH computation: failed ({e})")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # A2: SELECT DESCRIPTOR
    # -------------------------------------------------------------------------

    def _select_descriptor(
        self,
        context: Dict[str, Any],
        method: str,
        fewshot_holdout: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Select descriptor using the specified method."""

        if method == "topoagent":
            return self._select_topoagent(context)
        elif method == "topoagent_portfolio":
            return self._select_topoagent_portfolio(context)
        elif method == "topoagent_single_pv":
            return self._select_topoagent_single_pv(context)
        elif method == "topoagent_zero":
            return self._select_topoagent_zero(context)
        elif method == "topoagent_no_skills":
            return self._select_noskills(context)
        elif method == "zeroshot":
            return self._select_zeroshot(context)
        elif method == "fewshot":
            return self._select_fewshot(context, holdout=fewshot_holdout or [])
        elif method == "meta_learner":
            return self._select_meta_learner(context, variant="gbr")
        elif method == "meta_learner_ridge":
            return self._select_meta_learner(context, variant="ridge")
        elif method == "meta_learner_train_test":
            return self._select_meta_learner(context, variant="gbr", split="train_test")
        elif method == "meta_learner_ridge_train_test":
            return self._select_meta_learner(context, variant="ridge", split="train_test")
        elif method == "rule_based":
            return self._select_rule_based(context)
        elif method == "sba":
            return self._select_sba(context)
        elif method == "object_type_default":
            return self._select_object_type_default(context)
        elif method.startswith("fixed_"):
            descriptor = method[6:]  # "fixed_persistence_image" -> "persistence_image"
            return {"descriptor_choice": descriptor, "reasoning": f"Fixed baseline: {descriptor}"}
        elif method == "oracle":
            return self._select_oracle(context)
        elif method == "oracle_pair":
            return self._select_oracle_pair(context)
        elif method == "random":
            desc = random.choice(EXTENDED_DESCRIPTORS)
            return {"descriptor_choice": desc, "reasoning": "Random selection"}
        elif method == "random_pair":
            return self._select_random_pair(context)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _select_topoagent(self, context: Dict) -> Dict:
        """Full TopoAgent selection: skills + learned rules + benchmark rankings.

        In train_test mode: Uses Benchmark3 rankings (5-dataset pilot) — no B4 data.
        In lodo mode: Uses LODO-masked Benchmark4 rankings (original design).
        """
        object_type = context["object_type"]
        color_mode = context["color_mode"]

        active_tp = self._get_active_top_performers()

        # Build skill knowledge with rankings (batch mode: include all 15 descriptors)
        skill_knowledge = self.skill_registry.build_skill_context(
            object_type_hint=object_type,
            color_mode=color_mode,
            supported_only=False,  # Batch mode has training data for ATOL/codebook
        )

        # In LODO mode, replace rankings with LODO-masked benchmark4 rankings
        if self.experiment_mode == "lodo" and self._lodo_top_performers is not None:
            skill_knowledge = self._inject_lodo_rankings(skill_knowledge, object_type)

        learned_context = self.skill_memory.get_learned_context(object_type)

        prompt = BENCHMARK_SELECT_PROMPT.format(
            dataset_name=context["dataset_name"],
            dataset_description=context["description"],
            n_context_samples=len(context.get("images", [])),
            sample_stats=context["sample_stats"],
            cheap_features=context.get("cheap_features", "(not available)"),
            skill_knowledge=skill_knowledge,
            learned_context=learned_context,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        result = self._parse_json(response.content)

        interaction = LLMInteraction(
            step="benchmark_select",
            round=1,
            prompt=prompt,
            response=response.content,
        )
        result["_interaction"] = interaction

        # Validate descriptor choice against extended list
        if result.get("descriptor_choice") not in EXTENDED_DESCRIPTORS:
            tp = active_tp.get(object_type, [])
            if tp:
                result["descriptor_choice"] = tp[0].get("descriptor", "template_functions")
            else:
                result["descriptor_choice"] = "template_functions"

        return result

    def _select_topoagent_zero(self, context: Dict) -> Dict:
        """TopoAgent-zero: math descriptions + cheap features, NO empirical rankings.

        Has access to descriptor properties and reasoning chains but NOT the
        benchmark rankings. Tests whether structured TDA knowledge alone
        (without empirical data) helps the LLM make good choices.
        """
        object_type = context["object_type"]
        color_mode = context["color_mode"]

        # Build skill knowledge WITHOUT rankings (batch mode: include all 15 descriptors)
        skill_knowledge = build_descriptor_knowledge_text(
            object_type_hint=object_type,
            color_mode=color_mode,
            include_rankings=False,  # Key difference: no rankings
            supported_only=False,  # Batch mode has training data for ATOL/codebook
        )

        prompt = BENCHMARK_SELECT_ZERO_PROMPT.format(
            dataset_name=context["dataset_name"],
            dataset_description=context["description"],
            n_context_samples=len(context.get("images", [])),
            sample_stats=context["sample_stats"],
            cheap_features=context.get("cheap_features", "(not available)"),
            skill_knowledge=skill_knowledge,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        result = self._parse_json(response.content)

        interaction = LLMInteraction(
            step="topoagent_zero_select",
            round=1,
            prompt=prompt,
            response=response.content,
        )
        result["_interaction"] = interaction

        if result.get("descriptor_choice") not in EXTENDED_DESCRIPTORS:
            result["descriptor_choice"] = "persistence_image"

        return result

    def _select_topoagent_portfolio(self, context: Dict) -> Dict:
        """TopoAgent-portfolio: 3-step predict → verify → portfolio selection.

        Step 1 (PREDICT): LLM predicts top-3 descriptors from math reasoning only (no rankings).
        Step 2 (VERIFY): LLM compares prediction against Benchmark3 rankings, confirms or revises.
        Step 3 (PORTFOLIO): LLM selects 1-3 descriptors as a fusion portfolio.

        Returns selection dict with 'portfolio' list and 'descriptor_choice' (primary).
        """
        object_type = context["object_type"]
        color_mode = context["color_mode"]
        dataset_name = context["dataset_name"]
        interactions = []

        # --- Step 1: PREDICT (no rankings) ---
        descriptor_properties = self._build_descriptor_properties_text()

        predict_prompt = PREDICT_DESCRIPTORS_PROMPT.format(
            dataset_name=dataset_name,
            dataset_description=context["description"],
            cheap_features=context.get("cheap_features", "(not available)"),
            sample_stats=context.get("sample_stats", "(not available)"),
            descriptor_properties=descriptor_properties,
        )

        predict_response = self.model.invoke([HumanMessage(content=predict_prompt)])
        prediction = self._parse_json(predict_response.content)
        interactions.append(LLMInteraction(
            step="portfolio_predict", round=1,
            prompt=predict_prompt, response=predict_response.content,
        ))

        predicted_top3 = [p.get("descriptor", "") for p in prediction.get("predicted_top3", [])]
        if not predicted_top3:
            predicted_top3 = ["persistence_statistics"]

        # --- Step 2: VERIFY (show rankings) ---
        active_tp = self._get_active_top_performers()
        bench_performers = active_tp.get(object_type, [])
        bench_rankings_text = ""
        bench_top1 = ""
        bench_top1_acc = 0.0
        predicted_rank = "?"

        if bench_performers:
            for i, entry in enumerate(bench_performers, 1):
                bench_rankings_text += f"  {i}. {entry['descriptor']} — {entry.get('accuracy', 0):.1%}\n"
            bench_top1 = bench_performers[0]["descriptor"]
            bench_top1_acc = bench_performers[0].get("accuracy", 0)

            # Find predicted #1's rank in benchmark rankings
            for i, entry in enumerate(bench_performers, 1):
                if entry["descriptor"] == predicted_top3[0]:
                    predicted_rank = str(i)
                    break

        bench_top1_in_pred = "— was in your top 3" if bench_top1 in predicted_top3 else "— was NOT in your top 3"

        verify_prompt = VERIFY_PREDICTION_PROMPT.format(
            dataset_name=dataset_name,
            dataset_description=context["description"],
            prediction_json=json.dumps(prediction, indent=2),
            bench_rankings=bench_rankings_text,
            predicted_top1=predicted_top3[0],
            predicted_rank_in_bench=predicted_rank,
            bench_top1=bench_top1,
            bench_top1_accuracy=bench_top1_acc,
            bench_top1_in_prediction=bench_top1_in_pred,
        )

        verify_response = self.model.invoke([HumanMessage(content=verify_prompt)])
        verification = self._parse_json(verify_response.content)
        interactions.append(LLMInteraction(
            step="portfolio_verify", round=2,
            prompt=verify_prompt, response=verify_response.content,
        ))

        confirmed_primary = verification.get("confirmed_primary", predicted_top3[0])
        if confirmed_primary not in EXTENDED_DESCRIPTORS:
            confirmed_primary = bench_top1 if bench_top1 else "persistence_statistics"

        # --- Step 3: PORTFOLIO (complementarity + fusion) ---
        complementarity_text = build_complementarity_text(
            primary=confirmed_primary,
            object_type=object_type,
        )

        # Get fusion context from portfolio memory
        fusion_context = self._get_fusion_context(object_type)

        # Get available fusion results for this dataset
        available_fusions = self._get_available_fusions_text(dataset_name, confirmed_primary)

        portfolio_prompt = PORTFOLIO_SELECT_PROMPT.format(
            dataset_name=dataset_name,
            dataset_description=context["description"],
            confirmed_primary=confirmed_primary,
            complementarity_text=complementarity_text,
            fusion_context=fusion_context,
            available_fusions=available_fusions,
        )

        portfolio_response = self.model.invoke([HumanMessage(content=portfolio_prompt)])
        portfolio_result = self._parse_json(portfolio_response.content)
        interactions.append(LLMInteraction(
            step="portfolio_select", round=3,
            prompt=portfolio_prompt, response=portfolio_response.content,
        ))

        # Extract portfolio
        portfolio = portfolio_result.get("portfolio", [confirmed_primary])
        if not portfolio:
            portfolio = [confirmed_primary]

        # Validate all descriptors in portfolio
        portfolio = [d for d in portfolio if d in EXTENDED_DESCRIPTORS]
        if not portfolio:
            portfolio = [confirmed_primary]

        fusion_strategy = portfolio_result.get("fusion_strategy", "primary_only")

        return {
            "descriptor_choice": portfolio[0],  # Primary descriptor
            "portfolio": portfolio,
            "fusion_strategy": fusion_strategy,
            "descriptor_rationale": portfolio_result.get("portfolio_rationale", ""),
            "reasoning_chain": prediction.get("reasoning_chain", ""),
            "predict_result": prediction,
            "verify_result": verification,
            "portfolio_result": portfolio_result,
            "_interactions": interactions,
            "_llm_calls": 3,
        }

    def _select_topoagent_single_pv(self, context: Dict) -> Dict:
        """TopoAgent-single-PV: predict-verify only (no portfolio/fusion).

        Same as portfolio but stops after Step 2 — uses the confirmed primary
        descriptor only. For ablation: predict-verify vs direct-with-rankings.
        """
        object_type = context["object_type"]
        color_mode = context["color_mode"]
        dataset_name = context["dataset_name"]
        interactions = []

        # Step 1: PREDICT
        descriptor_properties = self._build_descriptor_properties_text()
        predict_prompt = PREDICT_DESCRIPTORS_PROMPT.format(
            dataset_name=dataset_name,
            dataset_description=context["description"],
            cheap_features=context.get("cheap_features", "(not available)"),
            sample_stats=context.get("sample_stats", "(not available)"),
            descriptor_properties=descriptor_properties,
        )
        predict_response = self.model.invoke([HumanMessage(content=predict_prompt)])
        prediction = self._parse_json(predict_response.content)
        interactions.append(LLMInteraction(
            step="pv_predict", round=1,
            prompt=predict_prompt, response=predict_response.content,
        ))

        predicted_top3 = [p.get("descriptor", "") for p in prediction.get("predicted_top3", [])]
        if not predicted_top3:
            predicted_top3 = ["persistence_statistics"]

        # Step 2: VERIFY
        active_tp = self._get_active_top_performers()
        bench_performers = active_tp.get(object_type, [])
        bench_rankings_text = ""
        bench_top1 = ""
        bench_top1_acc = 0.0
        predicted_rank = "?"

        if bench_performers:
            for i, entry in enumerate(bench_performers, 1):
                bench_rankings_text += f"  {i}. {entry['descriptor']} — {entry.get('accuracy', 0):.1%}\n"
            bench_top1 = bench_performers[0]["descriptor"]
            bench_top1_acc = bench_performers[0].get("accuracy", 0)
            for i, entry in enumerate(bench_performers, 1):
                if entry["descriptor"] == predicted_top3[0]:
                    predicted_rank = str(i)
                    break

        bench_top1_in_pred = "— was in your top 3" if bench_top1 in predicted_top3 else "— was NOT in your top 3"

        verify_prompt = VERIFY_PREDICTION_PROMPT.format(
            dataset_name=dataset_name,
            dataset_description=context["description"],
            prediction_json=json.dumps(prediction, indent=2),
            bench_rankings=bench_rankings_text,
            predicted_top1=predicted_top3[0],
            predicted_rank_in_bench=predicted_rank,
            bench_top1=bench_top1,
            bench_top1_accuracy=bench_top1_acc,
            bench_top1_in_prediction=bench_top1_in_pred,
        )
        verify_response = self.model.invoke([HumanMessage(content=verify_prompt)])
        verification = self._parse_json(verify_response.content)
        interactions.append(LLMInteraction(
            step="pv_verify", round=2,
            prompt=verify_prompt, response=verify_response.content,
        ))

        confirmed_primary = verification.get("confirmed_primary", predicted_top3[0])
        if confirmed_primary not in EXTENDED_DESCRIPTORS:
            confirmed_primary = bench_top1 if bench_top1 else "persistence_statistics"

        return {
            "descriptor_choice": confirmed_primary,
            "descriptor_rationale": verification.get("revision_reason", ""),
            "reasoning_chain": prediction.get("reasoning_chain", ""),
            "predict_result": prediction,
            "verify_result": verification,
            "_interactions": interactions,
            "_llm_calls": 2,
        }

    def _select_oracle_pair(self, context: Dict) -> Dict:
        """Oracle-pair: best 2-descriptor concatenation per dataset (upper bound for fusion)."""
        dataset_name = context["dataset_name"]

        if self._fusion_accuracy_lookup:
            # Find the best fusion pair for this dataset
            best_key = None
            best_acc = 0.0
            for key, result in self._fusion_accuracy_lookup.items():
                if result.get("dataset") == dataset_name:
                    acc = result.get("balanced_accuracy", 0)
                    if acc > best_acc:
                        best_acc = acc
                        best_key = key

            if best_key:
                result = self._fusion_accuracy_lookup[best_key]
                d1, d2 = result["desc1"], result["desc2"]
                return {
                    "descriptor_choice": d1,
                    "portfolio": [d1, d2],
                    "reasoning": f"Oracle-pair: {d1}+{d2} ({best_acc:.1%})",
                }

        # Fallback to single oracle
        return self._select_oracle(context)

    def _select_random_pair(self, context: Dict) -> Dict:
        """Random-pair: random 2-descriptor concatenation baseline."""
        d1, d2 = random.sample(EXTENDED_DESCRIPTORS, 2)
        return {
            "descriptor_choice": d1,
            "portfolio": [d1, d2],
            "reasoning": f"Random pair: {d1}+{d2}",
        }

    # -------------------------------------------------------------------------
    # Portfolio Helpers
    # -------------------------------------------------------------------------

    def _build_descriptor_properties_text(self) -> str:
        """Build descriptor mathematical properties text (no rankings)."""
        lines = []
        for name, props in DESCRIPTOR_PROPERTIES.items():
            lines.append(f"**{name}** (input: {props['input']})")
            lines.append(f"  Captures: {props['captures']}")
            lines.append(f"  Best when: {props['best_when']}")
            lines.append(f"  Why: {props['why']}")
            lines.append(f"  Weakness: {props['weakness']}")
            lines.append("")
        return "\n".join(lines)

    def _get_fusion_context(self, object_type: str) -> str:
        """Get fusion context from portfolio memory."""
        if self._portfolio_memory is None:
            try:
                from .memory.portfolio_memory import PortfolioMemory
                self._portfolio_memory = PortfolioMemory()
            except ImportError:
                return "(no fusion history available)"
        return self._portfolio_memory.get_fusion_context(object_type)

    def _get_available_fusions_text(self, dataset_name: str, primary: str) -> str:
        """Format available pre-computed fusion results for this dataset."""
        if not self._fusion_accuracy_lookup:
            return "(no pre-computed fusion results available)"

        lines = []
        for key, result in self._fusion_accuracy_lookup.items():
            if result.get("dataset") != dataset_name:
                continue
            d1, d2 = result["desc1"], result["desc2"]
            acc = result["balanced_accuracy"]
            dim = result.get("total_dim", "?")
            lines.append(f"  - {d1} + {d2}: {acc:.3f} (dim={dim})")

        if not lines:
            return "(no pre-computed fusion results for this dataset)"

        return "Pre-computed fusion accuracies for this dataset:\n" + "\n".join(lines)

    def _lookup_fusion_accuracy(
        self, dataset_name: str, portfolio: List[str],
    ) -> Tuple[float, Dict]:
        """Look up fusion accuracy from pre-computed results.

        Falls back to primary descriptor if fusion result not found.
        """
        if not self._fusion_accuracy_lookup or len(portfolio) < 2:
            # Fall back to single descriptor
            return self._lookup_accuracy(dataset_name, portfolio[0])

        # Build canonical key
        d1, d2 = sorted(portfolio[:2])
        key = f"{dataset_name}|{d1}+{d2}"

        if key in self._fusion_accuracy_lookup:
            result = self._fusion_accuracy_lookup[key]
            return (
                float(result["balanced_accuracy"]),
                {"source": "fusion_lookup", "feature_dim": result.get("total_dim", 0)},
            )

        # Try alternative key ordering
        for k, v in self._fusion_accuracy_lookup.items():
            if (v.get("dataset") == dataset_name and
                    set([v.get("desc1"), v.get("desc2")]) == set(portfolio[:2])):
                return (
                    float(v["balanced_accuracy"]),
                    {"source": "fusion_lookup", "feature_dim": v.get("total_dim", 0)},
                )

        # Fusion not pre-computed — fall back to primary descriptor
        print(f"    WARNING: No fusion result for {dataset_name}/{d1}+{d2}, "
              f"falling back to primary ({portfolio[0]})")
        return self._lookup_accuracy(dataset_name, portfolio[0])

    def _learn_portfolio(self, dataset_name: str, result: Dict, selection: Dict) -> None:
        """Update portfolio memory from benchmark result."""
        if self._portfolio_memory is None:
            try:
                from .memory.portfolio_memory import PortfolioMemory
                self._portfolio_memory = PortfolioMemory()
            except ImportError:
                return

        portfolio = result.get("portfolio", [])
        if len(portfolio) < 2:
            return

        # Get primary descriptor accuracy for comparison
        primary_acc, _ = self._lookup_accuracy(dataset_name, portfolio[0])

        self._portfolio_memory.record(
            dataset=dataset_name,
            object_type=result.get("object_type", ""),
            portfolio=portfolio,
            fusion_accuracy=result["accuracy"],
            primary_accuracy=primary_acc,
            fusion_strategy=result.get("fusion_strategy", "concat_pair"),
        )

    def _inject_lodo_rankings(self, skill_knowledge: str, object_type: str) -> str:
        """Replace benchmark rankings section with LODO variant."""
        if self._lodo_top_performers is None:
            return skill_knowledge

        tp = self._lodo_top_performers.get(object_type, [])
        if not tp:
            return skill_knowledge

        # Build LODO rankings text
        lodo_lines = [f"## Benchmark Rankings for {object_type} (LODO: {self._lodo_dataset} held out)"]
        lodo_lines.append("(aggregated from benchmark4, excluding test dataset)")
        lodo_lines.append("")
        for i, entry in enumerate(tp[:5], 1):
            desc = entry["descriptor"]
            acc = entry["mean_accuracy"]
            n_ds = entry.get("datasets", "?")
            lodo_lines.append(f"  {i}. {desc} — {acc:.1%} accuracy ({n_ds} datasets)")
        lodo_lines.append("")
        lodo_lines.append("The top-ranked descriptor is the empirically best choice. "
                          "You may override if your reasoning suggests otherwise.")
        lodo_text = "\n".join(lodo_lines)

        # Replace existing rankings section
        marker = f"## Benchmark Rankings for {object_type}"
        if marker in skill_knowledge:
            # Find start and next section
            start = skill_knowledge.index(marker)
            # Find next ## section or end
            rest = skill_knowledge[start + len(marker):]
            next_section = rest.find("\n## ")
            if next_section != -1:
                end = start + len(marker) + next_section
                return skill_knowledge[:start] + lodo_text + skill_knowledge[end:]
            else:
                return skill_knowledge[:start] + lodo_text
        else:
            # No existing section, append
            return skill_knowledge + "\n\n" + lodo_text

    def _select_noskills(self, context: Dict) -> Dict:
        """TopoAgent without skill knowledge: LLM + sample stats, no expert rules."""
        descriptor_list = "\n".join(f"- {d}" for d in EXTENDED_DESCRIPTORS)

        prompt = BASELINE_NOSKILLS_PROMPT.format(
            dataset_name=context["dataset_name"],
            dataset_description=context["description"],
            n_context_samples=len(context.get("images", [])),
            sample_stats=context["sample_stats"],
            cheap_features=context.get("cheap_features", "(not available)"),
            descriptor_list=descriptor_list,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        result = self._parse_json(response.content)

        interaction = LLMInteraction(
            step="noskills_select",
            round=1,
            prompt=prompt,
            response=response.content,
        )
        result["_interaction"] = interaction

        if result.get("descriptor_choice") not in EXTENDED_DESCRIPTORS:
            result["descriptor_choice"] = "persistence_image"

        return result

    def _select_zeroshot(self, context: Dict) -> Dict:
        """GPT-4 zero-shot baseline: no skill knowledge."""
        descriptor_list = "\n".join(f"- {d}" for d in EXTENDED_DESCRIPTORS)

        prompt = BASELINE_ZEROSHOT_PROMPT.format(
            dataset_name=context["dataset_name"],
            dataset_description=context["description"],
            descriptor_list=descriptor_list,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        result = self._parse_json(response.content)

        interaction = LLMInteraction(
            step="zeroshot_select",
            round=1,
            prompt=prompt,
            response=response.content,
        )
        result["_interaction"] = interaction

        if result.get("descriptor_choice") not in EXTENDED_DESCRIPTORS:
            result["descriptor_choice"] = "persistence_image"

        return result

    def _select_fewshot(self, context: Dict, holdout: List[str] = None) -> Dict:
        """GPT-4 few-shot baseline: 3 examples from other datasets."""
        holdout = holdout or []

        # Build examples from active TOP_PERFORMERS (exclude holdout datasets)
        active_tp = self._get_active_top_performers()
        examples_lines = []
        example_count = 0
        for ot, performers in active_tp.items():
            if example_count >= 3:
                break
            # Find a dataset for this object type that's not in holdout
            ds_name = self._get_example_dataset(ot, holdout)
            if ds_name and performers:
                best = performers[0]
                acc = best.get("mean_accuracy", best.get("accuracy", 0))
                examples_lines.append(
                    f"- {ds_name} ({ot}): Best descriptor = {best['descriptor']} "
                    f"(accuracy: {acc:.1%})"
                )
                example_count += 1

        descriptor_list = "\n".join(f"- {d}" for d in EXTENDED_DESCRIPTORS)

        prompt = BASELINE_FEWSHOT_PROMPT.format(
            dataset_name=context["dataset_name"],
            dataset_description=context["description"],
            examples="\n".join(examples_lines),
            descriptor_list=descriptor_list,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        result = self._parse_json(response.content)

        interaction = LLMInteraction(
            step="fewshot_select",
            round=1,
            prompt=prompt,
            response=response.content,
        )
        result["_interaction"] = interaction

        if result.get("descriptor_choice") not in EXTENDED_DESCRIPTORS:
            result["descriptor_choice"] = "persistence_image"

        return result

    def _select_sba(self, context: Dict) -> Dict:
        """Single Best Algorithm: always returns the SBA descriptor."""
        if self._sba_data:
            desc = self._sba_data["descriptor"]
            mean_acc = self._sba_data["mean_accuracy"]
            return {
                "descriptor_choice": desc,
                "reasoning": f"SBA: {desc} (mean accuracy {mean_acc:.1%} across "
                             f"{self._sba_data.get('n_datasets', '?')} datasets)",
            }
        # Fallback: persistence_statistics has full coverage
        return {
            "descriptor_choice": "persistence_statistics",
            "reasoning": "SBA fallback: persistence_statistics (full coverage)",
        }

    def _select_object_type_default(self, context: Dict) -> Dict:
        """Object-type default: best descriptor for the object type.

        In train_test mode: Uses Benchmark3 top-1 per type (from rules_data.py).
        In lodo mode: Uses LODO-masked benchmark4 rankings.
        """
        object_type = context["object_type"]
        active_tp = self._get_active_top_performers()
        performers = active_tp.get(object_type, [])

        source = "benchmark3" if self.experiment_mode == "train_test" else "benchmark4"

        if performers:
            best = performers[0]
            desc = best["descriptor"]
            acc = best.get("mean_accuracy", best.get("accuracy", 0))
            n_ds = best.get("datasets", "1 pilot dataset" if self.experiment_mode == "train_test" else "?")
            return {
                "descriptor_choice": desc,
                "reasoning": f"Object-type default ({source}): best for {object_type} is {desc} "
                             f"({acc:.1%})",
            }

        return {
            "descriptor_choice": "template_functions",
            "reasoning": f"Object-type default: no rankings for {object_type}, using fallback",
        }

    def _select_meta_learner(self, context: Dict, variant: str = "gbr",
                             split: str = "lodo") -> Dict:
        """Meta-learner predictions.

        Args:
            variant: "gbr" (GradientBoostingRegressor) or "ridge" (RidgeCV)
            split: "lodo" (leave-one-dataset-out) or "train_test" (5 Benchmark3 train → 21 test)
        """
        dataset_name = context["dataset_name"]

        # Select the right predictions dict based on variant and split
        if split == "train_test":
            if variant == "ridge":
                preds = self._meta_learner_ridge_train_test
                label = "RidgeCV train/test"
            else:
                preds = self._meta_learner_train_test
                label = "RidgeCV train/test"  # Primary is RidgeCV for train/test
        else:
            if variant == "ridge":
                preds = self._meta_learner_ridge_lodo
                label = "RidgeCV LODO"
            else:
                preds = self._meta_learner_lodo
                label = "GBR LODO"

        if preds and dataset_name in preds:
            pred = preds[dataset_name]
            return {
                "descriptor_choice": pred["predicted"],
                "reasoning": f"Meta-learner ({label}): predicted {pred['predicted']} "
                             f"(score={pred.get('predicted_score', 0):.3f}, "
                             f"trained on {pred.get('n_train', '?')} samples)",
            }

        # Fallback
        return {
            "descriptor_choice": "persistence_statistics",
            "reasoning": f"Meta-learner ({label}): no prediction available for {dataset_name}, using fallback",
        }

    def _select_rule_based(self, context: Dict) -> Dict:
        """Rule-based: 4 hand-crafted rules from exp7."""
        dataset_name = context["dataset_name"]

        # Use pre-computed predictions if available
        if self._rule_based_predictions and dataset_name in self._rule_based_predictions:
            pred = self._rule_based_predictions[dataset_name]
            return {
                "descriptor_choice": pred["predicted"],
                "reasoning": f"Rule-based: {pred['rule']}",
            }

        # Compute on-the-fly from cheap features
        feat = None
        if self._cheap_features and dataset_name in self._cheap_features:
            feat = self._cheap_features[dataset_name]

        if feat is None:
            return {
                "descriptor_choice": "ATOL",
                "reasoning": "Rule-based: no features available, using default (ATOL)",
            }

        # 4 rules from exp7 decision tree
        if feat["fg_bg_contrast"] >= 0.326755 and feat["intensity_skewness"] < -0.627868:
            return {
                "descriptor_choice": "betti_curves",
                "reasoning": "Rule 1: high contrast + negative skew → discrete objects",
            }
        if feat["largest_component_ratio"] >= 0.977793:
            return {
                "descriptor_choice": "persistence_statistics",
                "reasoning": "Rule 2: dominant component → diffuse/scattered",
            }
        if feat["intensity_mean"] < 0.195928:
            return {
                "descriptor_choice": "template_functions",
                "reasoning": "Rule 3: dark images → multi-scale/dense",
            }
        return {
            "descriptor_choice": "ATOL",
            "reasoning": "Rule 4: default",
        }

    def _select_oracle(self, context: Dict) -> Dict:
        """Oracle: best descriptor per dataset (from Benchmark4 or Benchmark3)."""
        dataset_name = context["dataset_name"]

        if self.lookup_mode and self._oracle_data and dataset_name in self._oracle_data:
            oracle = self._oracle_data[dataset_name]
            return {
                "descriptor_choice": oracle["descriptor"],
                "reasoning": f"Oracle: best from benchmark4 ({oracle['accuracy']:.1%})",
            }

        # Fallback to Benchmark3 object-type oracle
        oracle = self._get_oracle(context["object_type"])
        return {
            "descriptor_choice": oracle["descriptor"],
            "reasoning": f"Oracle: best from benchmark rankings ({oracle['accuracy']:.1%})",
        }

    def _get_oracle(self, object_type: str) -> Dict:
        """Get the oracle (best) descriptor for an object type."""
        performers = TOP_PERFORMERS.get(object_type, [])
        if performers:
            return performers[0]
        return {"descriptor": "template_functions", "accuracy": 0.0}

    def _get_oracle_for_dataset(self, dataset_name: str) -> Dict:
        """Get oracle for a specific dataset (benchmark4 lookup)."""
        if self._oracle_data and dataset_name in self._oracle_data:
            return self._oracle_data[dataset_name]
        # Fallback
        from .skills.rules_data import DATASET_TO_OBJECT_TYPE
        ot = DATASET_TO_OBJECT_TYPE.get(dataset_name, "surface_lesions")
        return self._get_oracle(ot)

    def _get_top3_descriptors(self, object_type: str) -> List[str]:
        """Get top 3 descriptor names for an object type."""
        active_tp = self._get_active_top_performers()
        performers = active_tp.get(object_type, [])
        return [p.get("descriptor", "") for p in performers[:3]]

    def _get_example_dataset(self, object_type: str, holdout: List[str]) -> Optional[str]:
        """Find a dataset for an object type that's not in holdout."""
        from .skills.rules_data import DATASET_TO_OBJECT_TYPE
        for ds, ot in DATASET_TO_OBJECT_TYPE.items():
            if ot == object_type and ds not in holdout:
                return ds
        return None

    # -------------------------------------------------------------------------
    # A3 + A4: ACCURACY (lookup or live)
    # -------------------------------------------------------------------------

    def _lookup_accuracy(self, dataset_name: str, descriptor: str) -> Tuple[float, Dict]:
        """Look up accuracy from pre-computed benchmark4 results."""
        if self._accuracy_lookup and dataset_name in self._accuracy_lookup:
            ds_accs = self._accuracy_lookup[dataset_name]
            if descriptor in ds_accs:
                acc = float(ds_accs[descriptor])
                return acc, {"source": "benchmark4_lookup", "feature_dim": 0}

        # Missing entry
        return 0.0, {"source": "missing", "feature_dim": 0,
                      "error": f"No benchmark4 result for {dataset_name}/{descriptor}"}

    def _batch_extract_and_evaluate(
        self,
        dataset_name: str,
        descriptor: str,
        object_type: str,
        color_mode: str,
        n_eval: int,
        cv_folds: int,
        seed: int,
        classifiers: Optional[List[str]] = None,
    ) -> Tuple[float, Dict]:
        """Extract features and evaluate via cross-validation (live mode).

        Args:
            classifiers: List of classifier names to evaluate.
                If None, uses single best classifier (legacy behavior).
                Pass ["TabPFN", "XGBoost", "CatBoost", "RandomForest", "TabM", "RealMLP"]
                for full 6-classifier evaluation.
        """
        _ensure_benchmark4_path()
        from data_loader import load_dataset, rgb_to_channels, _to_grayscale_float
        from precompute_ph import compute_ph
        from descriptor_runner import extract_features, extract_features_per_channel
        from optimal_rules import get_rules
        from classifier_wrapper import get_classifier, get_available_classifiers

        # Load full evaluation set
        images, labels, class_names = load_dataset(dataset_name, n_samples=n_eval, seed=seed)

        # Get optimal params
        rules = get_rules()
        config = rules.get_descriptor_config(descriptor, object_type, color_mode)
        params = config["params"]
        dim_per_channel = config["dim_per_channel"]
        total_dim = config["total_dim"]

        # Extract features
        is_ph_based = descriptor not in IMAGE_BASED_DESCRIPTORS
        per_channel = (color_mode == "per_channel")

        if per_channel:
            if images.ndim == 4 and images.shape[3] == 3:
                R, G, B = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
            else:
                R = G = B = images if images.ndim == 3 else images[:, :, :, 0]

            if is_ph_based:
                diags_R, _ = compute_ph(R)
                diags_G, _ = compute_ph(G)
                diags_B, _ = compute_ph(B)
                features = extract_features_per_channel(
                    descriptor,
                    diags_R=diags_R, diags_G=diags_G, diags_B=diags_B,
                    params=params, expected_dim_per_channel=dim_per_channel,
                )
            else:
                features = extract_features_per_channel(
                    descriptor,
                    images_R=R, images_G=G, images_B=B,
                    params=params, expected_dim_per_channel=dim_per_channel,
                )
        else:
            # Grayscale
            if images.ndim == 4:
                gray = _to_grayscale_float(images)
            else:
                gray = images

            if is_ph_based:
                diagrams, _ = compute_ph(gray)
                features = extract_features(
                    descriptor, diagrams=diagrams,
                    params=params, expected_dim=dim_per_channel,
                )
            else:
                features = extract_features(
                    descriptor, images=gray,
                    params=params, expected_dim=dim_per_channel,
                )

        if features is None or len(features) == 0:
            return 0.0, {"error": "Feature extraction returned empty", "feature_dim": 0}

        # Ensure labels are 0-indexed contiguous integers (XGBoost requires this)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        n_classes = len(le.classes_)

        # Determine which classifiers to run
        if classifiers is None:
            # Legacy single-classifier behavior
            clf_name = "XGBoost" if total_dim > 2000 else "TabPFN"
            if n_classes > 10:
                clf_name = "XGBoost"
            classifiers_to_run = [clf_name]
        else:
            # Filter to available classifiers
            available = get_available_classifiers()
            classifiers_to_run = [c for c in classifiers if c in available]
            if not classifiers_to_run:
                classifiers_to_run = ["RandomForest"]

        # Evaluate each classifier via k-fold CV
        # Note: get_classifier() returns a Pipeline with SimpleImputer →
        # StandardScaler → FeatureClipper → Classifier, so no external
        # scaling/clipping is needed here.
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import balanced_accuracy_score

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        folds = list(skf.split(features, labels))

        per_classifier = {}
        for clf_name in classifiers_to_run:
            fold_scores = []

            # Clear GPU cache before each classifier to prevent OOM accumulation
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            _device = 'cuda' if _torch.cuda.is_available() else 'cpu'

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                try:
                    clf = get_classifier(
                        clf_name,
                        n_features=features.shape[1],
                        n_classes=n_classes,
                        seed=seed,
                        device=_device,
                    )
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    fold_scores.append(balanced_accuracy_score(y_test, y_pred))
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and _device == 'cuda':
                        # CUDA OOM — retry this fold on CPU
                        _torch.cuda.empty_cache()
                        try:
                            clf = get_classifier(
                                clf_name,
                                n_features=features.shape[1],
                                n_classes=n_classes,
                                seed=seed,
                                device='cpu',
                            )
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            fold_scores.append(balanced_accuracy_score(y_test, y_pred))
                            _device = 'cpu'  # Use CPU for remaining folds
                        except Exception as e2:
                            per_classifier[clf_name] = {
                                "accuracy": 0.0,
                                "std": 0.0,
                                "error": f"OOM on CUDA, CPU fallback failed: {e2}",
                            }
                            fold_scores = []
                            break
                    else:
                        per_classifier[clf_name] = {
                            "accuracy": 0.0,
                            "std": 0.0,
                            "error": str(e),
                        }
                        fold_scores = []
                        break
                except Exception as e:
                    # Log error but continue
                    per_classifier[clf_name] = {
                        "accuracy": 0.0,
                        "std": 0.0,
                        "error": str(e),
                    }
                    fold_scores = []
                    break

            if fold_scores:
                per_classifier[clf_name] = {
                    "accuracy": float(np.mean(fold_scores)),
                    "std": float(np.std(fold_scores)),
                    "fold_scores": [float(s) for s in fold_scores],
                }

        # Determine best classifier
        best_clf = max(
            per_classifier,
            key=lambda c: per_classifier[c].get("accuracy", 0),
        )
        best_accuracy = per_classifier[best_clf]["accuracy"]

        details = {
            "feature_dim": features.shape[1],
            "n_samples": len(features),
            "best_classifier": best_clf,
            "per_classifier": per_classifier,
            "fold_scores": per_classifier[best_clf].get("fold_scores", []),
            "std": per_classifier[best_clf].get("std", 0),
            "classifier": best_clf,
        }

        return best_accuracy, details

    # -------------------------------------------------------------------------
    # A5: LEARN
    # -------------------------------------------------------------------------

    def _learn(self, dataset_name: str, result: Dict) -> None:
        """Update skill memory from benchmark result."""
        self.skill_memory.update_from_benchmark(
            agent_choice=result["descriptor_selected"],
            agent_accuracy=result["accuracy"],
            oracle_choice=result.get("oracle_descriptor"),
            oracle_accuracy=result.get("oracle_accuracy"),
            object_type=result.get("object_type", "surface_lesions"),
            dataset=dataset_name,
            params_used=None,
            source="benchmark_mode_a",
        )

        # Record verbal reflection
        was_optimal = result.get("top1_match", False)
        outcome = "optimal" if was_optimal else "suboptimal"
        regret = result.get("regret", 0)

        self.skill_memory.record_reflection(
            context=f"{result.get('object_type', '?')} on {dataset_name}",
            choice=result["descriptor_selected"],
            outcome=outcome,
            reflection_text=(
                f"Selected {result['descriptor_selected']} for {dataset_name} "
                f"({result.get('object_type', '?')}). "
                f"Accuracy: {result['accuracy']:.1%}. "
                f"Oracle: {result.get('oracle_descriptor', '?')} at {result.get('oracle_accuracy', 0):.1%}. "
                f"Regret: {regret:.3f}."
            ),
            lesson=(
                f"For {result.get('object_type', '?')} ({dataset_name}): "
                f"{result['descriptor_selected']} achieved {result['accuracy']:.1%}"
                + (f" (optimal)" if was_optimal else f" (regret={regret:.3f}, oracle={result.get('oracle_descriptor', '?')})")
            ),
        )

    # -------------------------------------------------------------------------
    # Summary & Metrics
    # -------------------------------------------------------------------------

    def _compute_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate metrics per method + dataset-method accuracy matrix."""
        from collections import defaultdict

        by_method = defaultdict(list)
        for r in results:
            if "error" not in r:
                by_method[r["method"]].append(r)

        # Get SBA and oracle means for gap calculation
        sba_accs = [r["accuracy"] for r in by_method.get("sba", [])]
        oracle_accs = [r["accuracy"] for r in by_method.get("oracle", [])]
        sba_mean = float(np.mean(sba_accs)) if sba_accs else None
        oracle_mean = float(np.mean(oracle_accs)) if oracle_accs else None

        # Build per-dataset SBA accuracy for win rate
        sba_per_ds = {}
        for r in by_method.get("sba", []):
            sba_per_ds[r["dataset"]] = r["accuracy"]

        per_method = {}
        for method, method_results in by_method.items():
            accuracies = [r["accuracy"] for r in method_results]
            regrets = [r.get("regret", 0) for r in method_results]
            top1 = [r.get("top1_match", False) for r in method_results]
            top3 = [r.get("top3_match", False) for r in method_results]

            mean_acc = float(np.mean(accuracies))

            # Gap closed: (method - sba) / (oracle - sba) * 100
            gap_closed = None
            if (sba_mean is not None and oracle_mean is not None
                    and (oracle_mean - sba_mean) > 0.005):
                gap_closed = (mean_acc - sba_mean) / (oracle_mean - sba_mean) * 100

            # Win rate vs SBA (per-dataset: method_acc >= sba_acc)
            wins_vs_sba = 0
            n_compared = 0
            for r in method_results:
                sba_acc = sba_per_ds.get(r["dataset"])
                if sba_acc is not None:
                    n_compared += 1
                    if r["accuracy"] >= sba_acc:
                        wins_vs_sba += 1
            win_rate_vs_sba = float(wins_vs_sba / n_compared) if n_compared > 0 else None

            per_method[method] = {
                "n_datasets": len(method_results),
                "mean_accuracy": mean_acc,
                "std_accuracy": float(np.std(accuracies)),
                "mean_regret": float(np.mean(regrets)),
                "max_regret": float(np.max(regrets)) if regrets else 0,
                "top1_rate": float(np.mean(top1)),
                "top3_rate": float(np.mean(top3)),
                "win_rate_vs_sba": win_rate_vs_sba,
                "total_llm_calls": sum(r.get("llm_calls", 0) for r in method_results),
                "gap_closed": gap_closed,
            }

        # Build accuracy matrix: {method: {dataset: accuracy}}
        matrix = defaultdict(dict)
        desc_matrix = defaultdict(dict)
        for r in results:
            if "error" not in r:
                matrix[r["method"]][r["dataset"]] = r["accuracy"]
                desc_matrix[r["method"]][r["dataset"]] = r.get("descriptor_selected", "?")

        return {
            "per_method": per_method,
            "accuracy_matrix": dict(matrix),
            "descriptor_matrix": dict(desc_matrix),
        }

    def format_matrix_table(self, summary: Dict, datasets: List[str]) -> str:
        """Format the accuracy matrix as a markdown table.

        Args:
            summary: Output from _compute_summary
            datasets: Ordered list of dataset names

        Returns:
            Formatted markdown table string
        """
        matrix = summary.get("accuracy_matrix", {})
        per_method = summary.get("per_method", {})

        # Method display names
        METHOD_NAMES = {
            "oracle": "Oracle (upper bound)",
            "topoagent": "TopoAgent (ours)",
            "topoagent_zero": "TopoAgent-zero",
            "topoagent_no_skills": "TopoAgent (no skills)",
            "object_type_default": "ObjType default",
            "meta_learner": "Meta-learner (GBR)",
            "meta_learner_ridge": "Meta-learner (Ridge)",
            "meta_learner_train_test": "Meta-learner (GBR t/t)",
            "meta_learner_ridge_train_test": "Meta-learner (Ridge)",
            "rule_based": "Rule-based (4 rules)",
            "sba": f"SBA ({self._sba_data['descriptor'][:15]})" if self._sba_data else "SBA (persist_stats)",
            "fewshot": "GPT-4o (few-shot)",
            "zeroshot": "LLM zero-shot",
            "random": "Random (lower bound)",
        }

        # Determine column widths
        ds_short = [d.replace("MNIST", "") if d.endswith("MNIST") else d for d in datasets]
        col_w = max(10, max((len(s) for s in ds_short), default=10))

        # Build method order
        METHOD_ORDER = [
            "oracle", "topoagent", "topoagent_zero", "topoagent_no_skills",
            "meta_learner_ridge_train_test", "meta_learner_train_test",
            "object_type_default", "meta_learner", "meta_learner_ridge",
            "rule_based", "sba",
            "fewshot", "zeroshot", "random",
        ]
        all_methods = list(matrix.keys())
        fixed_methods = sorted(m for m in all_methods if m.startswith("fixed_"))
        ordered = [m for m in METHOD_ORDER if m in all_methods]
        ordered.extend(m for m in fixed_methods)
        ordered.extend(m for m in all_methods if m not in ordered)

        lines = []

        # Header
        header_parts = [f"{'Method':<26}"]
        header_parts.extend(f"{s:>{col_w}}" for s in ds_short)
        header_parts.append(f"{'Avg':>{col_w}}")
        header_parts.append(f"{'Regret':>{col_w}}")
        header_parts.append(f"{'% Gap':>{col_w}}")
        lines.append(" | ".join(header_parts))

        # Separator
        sep_parts = ["-" * 26]
        sep_parts.extend("-" * col_w for _ in datasets)
        sep_parts.append("-" * col_w)
        sep_parts.append("-" * col_w)
        sep_parts.append("-" * col_w)
        lines.append("-|-".join(sep_parts))

        # Find best accuracy per dataset (excluding oracle) for bolding
        best_per_ds = {}
        for ds in datasets:
            best = 0
            for m in ordered:
                if m == "oracle":
                    continue
                acc = matrix.get(m, {}).get(ds, 0)
                if acc > best:
                    best = acc
            best_per_ds[ds] = best

        # Rows
        for method in ordered:
            name = METHOD_NAMES.get(method, method)
            if method.startswith("fixed_"):
                desc = method[6:]
                name = f"Fixed ({desc[:16]})"

            row_parts = [f"{name:<26}"]
            accs = []
            for ds in datasets:
                acc = matrix.get(method, {}).get(ds)
                if acc is not None:
                    accs.append(acc)
                    # Bold the best non-oracle method per dataset
                    if method != "oracle" and abs(acc - best_per_ds.get(ds, 0)) < 1e-6 and acc > 0:
                        row_parts.append(f"{'**' + f'{acc:.1%}' + '**':>{col_w}}")
                    else:
                        row_parts.append(f"{acc:>{col_w}.1%}")
                else:
                    row_parts.append(f"{'--':>{col_w}}")

            avg = float(np.mean(accs)) if accs else 0
            stats = per_method.get(method, {})
            regret = stats.get("mean_regret", 0)
            gap = stats.get("gap_closed")

            # Avg column
            if method != "oracle" and method in matrix:
                non_oracle_avgs = [
                    float(np.mean(list(matrix[m].values())))
                    for m in ordered if m != "oracle" and m in matrix and matrix[m]
                ]
                if non_oracle_avgs and avg == max(non_oracle_avgs):
                    row_parts.append(f"{'**' + f'{avg:.1%}' + '**':>{col_w}}")
                else:
                    row_parts.append(f"{avg:>{col_w}.1%}")
            else:
                row_parts.append(f"{avg:>{col_w}.1%}")

            row_parts.append(f"{regret:>{col_w}.3f}")

            # % Gap column
            if gap is not None and method not in ("oracle", "random"):
                row_parts.append(f"{gap:>{col_w}.1f}%")
            else:
                row_parts.append(f"{'--':>{col_w}}")

            lines.append(" | ".join(row_parts))

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _parse_json(self, text: str) -> Dict:
        """Extract JSON from LLM response."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        return {"descriptor_choice": "template_functions", "reasoning": "JSON parse failed, using default"}
