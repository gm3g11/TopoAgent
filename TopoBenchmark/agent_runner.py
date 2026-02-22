"""Agent Runner for TopoBenchmark.

Runs TopoAgent on Protocol 1 (per-dataset strategic selection) and
Protocol 2 (per-image end-to-end).

Protocol 1 uses a direct LLM call (no tool execution) to select a descriptor.
Protocol 2 uses the full agent workflow with tool execution.

Supports ablation configs by toggling skills, reflection, and memory.
"""

import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    DATASET_DESCRIPTIONS,
    ABLATION_CONFIGS,
    build_agent_query,
)
from .ground_truth import GroundTruth, load_ground_truth, ALL_DESCRIPTORS


# Where to save results
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "topobenchmark"


# =============================================================================
# Descriptor parsing utilities
# =============================================================================

def _parse_descriptor_from_text(text: str) -> Optional[str]:
    """Parse descriptor name from free-form text.

    Checks exact names first, then common abbreviations.
    Returns the FIRST match found (to handle texts mentioning multiple).
    """
    if not text:
        return None

    text_lower = text.lower()

    # First, check for structured output patterns like "Selected: persistence_statistics"
    structured_patterns = [
        r"(?:selected|recommended|choice|pick|choose|best)[\s:]+(\w+(?:_\w+)*)",
        r"(?:descriptor|method)[\s:]+(\w+(?:_\w+)*)",
        r"\*\*(\w+(?:_\w+)*)\*\*",  # Bold markdown
    ]
    for pattern in structured_patterns:
        match = re.search(pattern, text_lower)
        if match:
            candidate = match.group(1)
            for desc in ALL_DESCRIPTORS:
                if desc.lower() == candidate:
                    return desc

    # Check each descriptor name (ordered by specificity — longer names first)
    sorted_descriptors = sorted(ALL_DESCRIPTORS, key=len, reverse=True)
    for desc in sorted_descriptors:
        if desc.lower() in text_lower:
            return desc

    # Check common abbreviations
    abbrevs = {
        "persistence image": "persistence_image",
        "persistence landscapes": "persistence_landscapes",
        "betti curve": "betti_curves",
        "persistence silhouette": "persistence_silhouette",
        "persistence entropy": "persistence_entropy",
        "persistence statistics": "persistence_statistics",
        "tropical coordinate": "tropical_coordinates",
        "persistence codebook": "persistence_codebook",
        "template function": "template_functions",
        "minkowski functional": "minkowski_functionals",
        "euler characteristic curve": "euler_characteristic_curve",
        "euler characteristic transform": "euler_characteristic_transform",
        "edge histogram": "edge_histogram",
        "lbp texture": "lbp_texture",
        "lbp": "lbp_texture",
        "ect": "euler_characteristic_transform",
        "ecc": "euler_characteristic_curve",
    }
    for abbrev, full_name in abbrevs.items():
        if abbrev in text_lower:
            return full_name

    return None


def _parse_descriptor_from_state(state: Dict[str, Any]) -> Optional[str]:
    """Extract the descriptor choice from agent final state.

    Checks multiple sources:
    1. skill_descriptor (set by skills mode when descriptor tool is executed)
    2. Short-term memory (tool names that match descriptor names)
    3. Final answer text (parse descriptor name mentions)
    """
    # Source 1: skill_descriptor (most reliable in skills mode)
    skill_desc = state.get("skill_descriptor")
    if skill_desc and skill_desc in ALL_DESCRIPTORS:
        return skill_desc

    # Source 2: Short-term memory — find descriptor tool executions
    stm = state.get("short_term_memory", [])
    for tool_name, output in reversed(stm):
        if tool_name in ALL_DESCRIPTORS:
            return tool_name

    # Source 3: Parse from final answer
    answer = state.get("final_answer", "")
    if answer:
        return _parse_descriptor_from_text(answer)

    return None


def _serialize_skill_params(params):
    """Make skill params JSON-serializable."""
    if params is None:
        return None
    serializable = {}
    for k, v in params.items():
        if hasattr(v, "item"):  # numpy scalar
            serializable[k] = v.item()
        else:
            serializable[k] = v
    return serializable


# =============================================================================
# Protocol 1: Direct LLM call for descriptor selection
# =============================================================================

PROTOCOL1_SYSTEM_PROMPT = """You are TopoAgent, an expert in Topological Data Analysis (TDA) for medical image classification.

Your task is to select the best TDA descriptor for a given medical image dataset.

{skill_knowledge}

INSTRUCTIONS:
1. Analyze the dataset description, object type, and image characteristics.
2. Reason about which descriptor would work best based on your knowledge.
3. Select exactly ONE descriptor from the available list.
4. State your selection clearly.

IMPORTANT: You MUST end your response with a line in this exact format:
SELECTED: <descriptor_name>

Where <descriptor_name> is one of: {descriptor_list}
"""

PROTOCOL1_NO_SKILLS_PROMPT = """You are TopoAgent, an expert in Topological Data Analysis (TDA) for medical image classification.

Your task is to select the best TDA descriptor for a given medical image dataset.

Available descriptors:
- persistence_image: Discretized persistence diagram on a grid. Good for general features.
- persistence_landscapes: Statistical summary of persistence diagrams via landscapes.
- betti_curves: Count of topological features as function of filtration parameter.
- persistence_silhouette: Weighted average of persistence landscape functions.
- persistence_entropy: Shannon entropy of persistence diagram.
- persistence_statistics: Statistical summaries (mean, std, etc.) of persistence pairs.
- tropical_coordinates: Coordinates in tropical polynomial framework.
- persistence_codebook: Learned codebook of persistence diagram patches.
- ATOL: Automatic Topologically-Oriented Learning of persistence diagrams.
- template_functions: Point distributions via tent/Gaussian basis functions.
- minkowski_functionals: Area, perimeter, Euler characteristic at multiple thresholds.
- euler_characteristic_curve: Euler characteristic as function of filtration threshold.
- euler_characteristic_transform: Directional Euler characteristic curves.
- edge_histogram: Edge orientation distribution across spatial grid.
- lbp_texture: Local Binary Pattern texture features.

INSTRUCTIONS:
1. Analyze the dataset description, object type, and image characteristics.
2. Reason about which descriptor would work best.
3. Select exactly ONE descriptor.

IMPORTANT: You MUST end your response with a line in this exact format:
SELECTED: <descriptor_name>
"""


def _build_protocol1_prompt(
    dataset: str,
    skills_mode: bool = True,
) -> str:
    """Build the system + user prompt for Protocol 1.

    Args:
        dataset: Dataset name
        skills_mode: Whether to include skill knowledge

    Returns:
        Full prompt string
    """
    desc = DATASET_DESCRIPTIONS[dataset]
    descriptor_list = ", ".join(ALL_DESCRIPTORS)

    if skills_mode:
        try:
            from topoagent.skills import SkillRegistry
            registry = SkillRegistry()
            skill_knowledge = registry.build_skill_context(
                object_type_hint=desc["object_type"],
                color_mode=desc["color_mode"],
            )
        except ImportError:
            skill_knowledge = ""
            skills_mode = False

        system = PROTOCOL1_SYSTEM_PROMPT.format(
            skill_knowledge=skill_knowledge,
            descriptor_list=descriptor_list,
        )
    else:
        system = PROTOCOL1_NO_SKILLS_PROMPT

    user_query = build_agent_query(dataset)

    return f"{system}\n\n---\n\n{user_query}"


def run_protocol1_single(
    dataset: str,
    model_name: str = "gpt-4o",
    config_name: str = "full",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    **kwargs,
) -> Dict[str, Any]:
    """Run Protocol 1 on a single dataset using direct LLM call.

    The agent receives dataset context and selects ONE descriptor.
    No tool execution — just LLM reasoning.

    Args:
        dataset: Dataset name (e.g. "BloodMNIST")
        model_name: LLM model name
        config_name: Ablation config name (see ABLATION_CONFIGS)
        api_key: API key (reads from env if None)
        base_url: API base URL
        temperature: LLM temperature

    Returns:
        Dict with dataset, descriptor selection, reasoning, timing, etc.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from dotenv import load_dotenv

    load_dotenv()

    config = ABLATION_CONFIGS[config_name]

    # Build API connection
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL")

    openai_kwargs = {}
    if api_key:
        openai_kwargs["api_key"] = api_key
    if base_url:
        openai_kwargs["base_url"] = base_url

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        **openai_kwargs,
    )

    # Build prompt
    prompt = _build_protocol1_prompt(
        dataset=dataset,
        skills_mode=config["skills_mode"],
    )

    start_time = time.time()
    try:
        # Single LLM call
        response = model.invoke([HumanMessage(content=prompt)])
        reasoning = response.content
        elapsed = time.time() - start_time

        # Parse descriptor from response
        descriptor = None

        # Try structured format first: "SELECTED: descriptor_name"
        selected_match = re.search(
            r"SELECTED:\s*(\w+(?:_\w+)*)", reasoning, re.IGNORECASE
        )
        if selected_match:
            candidate = selected_match.group(1)
            for desc in ALL_DESCRIPTORS:
                if desc.lower() == candidate.lower():
                    descriptor = desc
                    break

        # Fall back to free-form parsing
        if descriptor is None:
            descriptor = _parse_descriptor_from_text(reasoning)

        result = {
            "dataset": dataset,
            "config": config_name,
            "model": model_name,
            "descriptor": descriptor,
            "reasoning": reasoning,
            "prompt": prompt,
            "elapsed_seconds": elapsed,
            "success": descriptor is not None,
            "timestamp": datetime.now().isoformat(),
            "skills_mode": config["skills_mode"],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            "dataset": dataset,
            "config": config_name,
            "model": model_name,
            "descriptor": None,
            "reasoning": "",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
            "success": False,
            "timestamp": datetime.now().isoformat(),
        }

    return result


# =============================================================================
# Protocol 1: Full workflow mode (alternative — uses actual agent)
# =============================================================================

def run_protocol1_workflow(
    dataset: str,
    model_name: str = "gpt-4o",
    config_name: str = "full",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_rounds: int = 4,
) -> Dict[str, Any]:
    """Run Protocol 1 using the full agent workflow.

    The agent runs the complete workflow (analyze_query -> select_tool -> execute_tool
    -> reflect -> ...). This tests the full reasoning pipeline including tool
    selection and reflection.

    Note: Tool execution will fail since no real images are loaded for Protocol 1.
    The agent must still select a descriptor through its reasoning trace.
    """
    from topoagent.agent import create_topoagent

    config = ABLATION_CONFIGS[config_name]
    query = build_agent_query(dataset)

    agent = create_topoagent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_rounds=max_rounds,
        skills_mode=config["skills_mode"],
    )

    # Override ablation flags
    agent.workflow.enable_reflection = config["enable_reflection"]
    agent.workflow.enable_short_memory = config["enable_short_memory"]
    agent.workflow.enable_long_memory = config["enable_long_memory"]
    agent.workflow.workflow = agent.workflow._build_workflow()

    dummy_image_path = f"datasets/{dataset}/sample.png"

    start_time = time.time()
    try:
        final_state = agent.workflow.invoke(
            query=query,
            image_path=dummy_image_path,
        )
        elapsed = time.time() - start_time
        descriptor = _parse_descriptor_from_state(final_state)

        result = {
            "dataset": dataset,
            "config": config_name,
            "model": model_name,
            "descriptor": descriptor,
            "reasoning": final_state.get("final_answer", ""),
            "reasoning_trace": final_state.get("reasoning_trace", []),
            "tools_used": [t for t, _ in final_state.get("short_term_memory", [])],
            "rounds_used": final_state.get("current_round", 0) - 1,
            "skill_descriptor": final_state.get("skill_descriptor"),
            "skill_params": _serialize_skill_params(final_state.get("skill_params")),
            "elapsed_seconds": elapsed,
            "success": descriptor is not None,
            "timestamp": datetime.now().isoformat(),
            "mode": "workflow",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            "dataset": dataset,
            "config": config_name,
            "model": model_name,
            "descriptor": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "mode": "workflow",
        }

    return result


# =============================================================================
# Batch runners
# =============================================================================

def run_protocol1(
    datasets: Optional[List[str]] = None,
    model_name: str = "gpt-4o",
    config_name: str = "full",
    output_dir: Optional[Path] = None,
    mode: str = "direct",
    **kwargs,
) -> Dict[str, Dict]:
    """Run Protocol 1 on all (or specified) datasets.

    Args:
        datasets: List of dataset names (all 26 if None)
        model_name: LLM model name
        config_name: Ablation config name
        output_dir: Where to save results
        mode: "direct" (single LLM call) or "workflow" (full agent pipeline)
        **kwargs: Additional args passed to runner function

    Returns:
        {dataset: result_dict}
    """
    if datasets is None:
        datasets = list(DATASET_DESCRIPTIONS.keys())

    if output_dir is None:
        output_dir = RESULTS_DIR / "protocol1"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = run_protocol1_single if mode == "direct" else run_protocol1_workflow

    all_results = {}
    for i, dataset in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] Running {dataset}...")
        result = runner(
            dataset=dataset,
            model_name=model_name,
            config_name=config_name,
            **kwargs,
        )
        all_results[dataset] = result

        # Save individual result
        fname = f"{dataset}_{config_name}_{model_name.replace('/', '_')}.json"
        with open(output_dir / fname, "w") as f:
            json.dump(result, f, indent=2)

        status = "OK" if result["success"] else "FAIL"
        desc = result.get("descriptor", "None")
        elapsed = result.get("elapsed_seconds", 0)
        print(f"  [{status}] -> {desc} ({elapsed:.1f}s)")

    # Save combined results
    combined_path = output_dir / f"all_{config_name}_{model_name.replace('/', '_')}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def extract_selections_from_results(
    results: Dict[str, Dict],
) -> Dict[str, str]:
    """Extract {dataset: descriptor} from protocol 1 results.

    Args:
        results: {dataset: result_dict} from run_protocol1

    Returns:
        {dataset: descriptor} for datasets with successful selections
    """
    selections = {}
    for dataset, result in results.items():
        if result.get("success") and result.get("descriptor"):
            selections[dataset] = result["descriptor"]
    return selections


def load_protocol1_results(
    results_dir: Optional[Path] = None,
    config_name: str = "full",
    model_name: str = "gpt-4o",
) -> Dict[str, Dict]:
    """Load saved Protocol 1 results from disk.

    Args:
        results_dir: Path to protocol1 results directory
        config_name: Ablation config name
        model_name: Model name

    Returns:
        {dataset: result_dict}
    """
    if results_dir is None:
        results_dir = RESULTS_DIR / "protocol1"

    combined_path = results_dir / f"all_{config_name}_{model_name.replace('/', '_')}.json"
    if combined_path.exists():
        with open(combined_path) as f:
            return json.load(f)

    # Fall back to loading individual files
    results = {}
    pattern = f"*_{config_name}_{model_name.replace('/', '_')}.json"
    for fpath in sorted(results_dir.glob(pattern)):
        if fpath.name.startswith("all_"):
            continue
        with open(fpath) as f:
            data = json.load(f)
        dataset = data.get("dataset")
        if dataset:
            results[dataset] = data

    return results
