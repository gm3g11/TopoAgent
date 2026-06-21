"""Fixed TDA Pipeline Baselines for TopoAgent Experiments.

This module provides fixed TDA pipelines and direct classification baselines
for comparison with the adaptive TopoAgent.

Baselines:
1. Fixed TDA Pipelines (no LLM):
   - Pipeline A: Sublevel filtration + stats + k-NN
   - Pipeline B: Cubical complex + landscapes + MLP
   - Pipeline C: Multi-vectorization ensemble

2. LLM-Only Baselines:
   - GPT4VDirectClassifier: Direct image classification with GPT-4V

3. TDA-Only Baselines (no LLM orchestration):
   - PIMlpDirectClassifier: Direct PI features + trained MLP
"""

from .fixed_pipelines import (
    FixedTDAPipeline,
    PipelineA_Sublevel,
    PipelineB_Cubical,
    PipelineC_MultiVec,
    get_all_pipelines,
)

from .run_baselines import (
    run_single_image,
    run_dataset_evaluation,
    compute_metrics,
    compare_pipelines,
)

# Import new baselines (may fail if dependencies not installed)
try:
    from .gpt4v_direct import (
        GPT4VDirectClassifier,
        evaluate_gpt4v_direct,
    )
except ImportError:
    GPT4VDirectClassifier = None
    evaluate_gpt4v_direct = None

try:
    from .pi_mlp_direct import (
        PIMlpDirectClassifier,
        evaluate_pi_mlp_direct,
    )
except ImportError:
    PIMlpDirectClassifier = None
    evaluate_pi_mlp_direct = None


def get_all_baseline_methods():
    """Get dictionary of all available baseline methods.

    Returns:
        Dictionary of method_name -> evaluation function
    """
    methods = {
        # Fixed TDA pipelines
        "pipeline_a": lambda **kwargs: run_dataset_evaluation(
            pipeline_name="pipeline_a", **kwargs
        ),
        "pipeline_b": lambda **kwargs: run_dataset_evaluation(
            pipeline_name="pipeline_b", **kwargs
        ),
        "pipeline_c": lambda **kwargs: run_dataset_evaluation(
            pipeline_name="pipeline_c", **kwargs
        ),
    }

    # Add GPT-4V if available
    if evaluate_gpt4v_direct is not None:
        methods["gpt4v_direct"] = evaluate_gpt4v_direct

    # Add PI+MLP direct if available
    if evaluate_pi_mlp_direct is not None:
        methods["pi_mlp_direct"] = evaluate_pi_mlp_direct

    return methods


__all__ = [
    # Fixed pipeline classes
    "FixedTDAPipeline",
    "PipelineA_Sublevel",
    "PipelineB_Cubical",
    "PipelineC_MultiVec",
    "get_all_pipelines",
    # Runner functions
    "run_single_image",
    "run_dataset_evaluation",
    "compute_metrics",
    "compare_pipelines",
    # New baselines
    "GPT4VDirectClassifier",
    "evaluate_gpt4v_direct",
    "PIMlpDirectClassifier",
    "evaluate_pi_mlp_direct",
    # Utility
    "get_all_baseline_methods",
]
