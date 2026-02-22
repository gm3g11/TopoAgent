"""TopoAgent State Definition.

This module defines the state structure for TopoAgent, combining patterns from:
- MedRAX: LangGraph workflow with messages and tool calls
- EndoAgent: Dual-memory mechanism (short-term Ms, long-term Ml)
"""

from typing import TypedDict, List, Tuple, Optional, Any, Annotated, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime
import operator


@dataclass
class ToolOutput:
    """Represents the output from a tool execution."""
    tool_name: str
    output: Any
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ReflectionEntry:
    """Represents a reflection entry stored in long-term memory.

    Based on EndoAgent's reflection mechanism which adds +26.5% visual accuracy.
    """
    round: int
    error_analysis: str  # What went wrong or could be improved
    suggestion: str      # Specific suggestion for next step
    experience: str      # Reusable experience/lesson learned
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class LLMInteraction:
    """Represents a single LLM interaction (prompt + response).

    Used for detailed logging of GPT reasoning.
    """
    step: str  # e.g., "tool_selection", "reflection", "completion_check", "final_answer"
    round: int
    prompt: str
    response: str
    tool_calls: Optional[List[dict]] = None  # For tool selection steps
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class AgentReport:
    """Structured output from the TopoAgent skills pipeline.

    Captures the full reasoning trace: selection, execution, verification,
    and reflection — suitable for paper case studies and structured logging.
    """
    # Core selection
    descriptor: str = ""
    object_type: str = ""
    reasoning_chain: str = ""

    # Reasoning
    image_analysis: str = ""
    descriptor_rationale: str = ""
    alternatives_considered: str = ""

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    feature_dimension: int = 0
    color_mode: str = "grayscale"

    # Quality
    actual_dimension: int = 0
    sparsity_pct: float = 0.0
    variance: float = 0.0
    nan_count: int = 0

    # Verification
    ph_confirms_object_type: bool = True
    dimension_correct: bool = True
    quality_ok: bool = True
    issues: List[str] = field(default_factory=list)

    # Reflection
    error_analysis: str = ""
    experience: str = ""

    # LLM's OBSERVE decisions (agentic pipeline v7)
    observe_object_type: str = ""
    observe_color_mode: str = ""
    observe_filtration: str = ""
    observe_reasoning: str = ""
    object_type_correct: Optional[bool] = None  # vs ground truth
    benchmark_stance: str = ""  # "FOLLOW" or "DEVIATE"
    ph_interpretation: str = ""  # LLM's PH analysis
    descriptor_intuition: str = ""  # Pre-benchmark descriptor preference

    # Execution metadata
    n_llm_calls: int = 2
    n_tools_executed: int = 3
    retry_used: bool = False
    total_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON serialization."""
        return asdict(self)


class TopoAgentState(TypedDict):
    """State definition for TopoAgent workflow.

    Combines MedRAX's message-based state with EndoAgent's dual-memory mechanism.

    Attributes:
        query: The user's input query/task
        image_path: Path to the input medical image
        short_term_memory: List of (tool_name, output) tuples from current session
        long_term_memory: List of ReflectionEntry objects with past experiences
        current_round: Current reasoning round (1-3)
        max_rounds: Maximum reasoning rounds (default 3, from EndoAgent ablation)
        reasoning_trace: List of reasoning steps for explainability
        messages: LangGraph message list for tool calling
        final_answer: The final classification/answer
        confidence: Confidence score for the final answer
        evidence: List of evidence supporting the answer
        task_complete: Whether the task is complete
    """
    # Input
    query: str
    image_path: str

    # Short-term memory Ms (from EndoAgent)
    # Stores recent tool outputs: [(tool_name, output), ...]
    short_term_memory: List[Tuple[str, Any]]

    # Long-term memory Ml (from EndoAgent)
    # Stores reflection entries with lessons learned
    long_term_memory: List[ReflectionEntry]

    # Reasoning control
    current_round: int
    max_rounds: int  # Default 3 (from EndoAgent ablation study)
    reasoning_trace: List[str]

    # LangGraph message list (for tool calling)
    messages: Annotated[List[Any], operator.add]

    # Output
    final_answer: Optional[str]
    confidence: float
    evidence: List[str]

    # Control flow
    task_complete: bool

    # Internal state for passing between nodes
    _tool_outputs: Optional[List[dict]]
    _current_reflection: Optional[Any]

    # Detailed LLM interaction logs
    llm_interactions: List[Any]  # List of LLMInteraction objects

    # Skills system fields (v4: benchmark-validated adaptive pipeline)
    skill_descriptor: Optional[str]          # e.g., "template_functions" (set after LLM picks)
    skill_params: Optional[Dict[str, Any]]   # e.g., {"n_templates": 81, ...} (from benchmark rules)
    skill_color_mode: Optional[str]          # "grayscale" or "per_channel"

    # Optimized skills pipeline fields (v4.1: 2-LLM-call architecture)
    _plan: Optional[Dict[str, Any]]           # Structured plan from plan_descriptor LLM call
    _ph_stats: Optional[Dict[str, Any]]       # Extracted PH statistics from pre_execute
    _feature_quality: Optional[Dict[str, Any]]  # Feature quality metrics from execute_descriptor
    _needs_retry: bool                         # Whether to retry with alternative descriptor
    _retry_count: int                          # Number of retries attempted (max 3)
    _failed_descriptors: List[str]              # Descriptors that failed quality checks

    # Agentic pipeline fields (v6: LLM-driven 3-phase ReAct)
    _agentic_phase: Optional[str]             # "observe" | "act" | "reflect"
    _observe_turns: int                        # Turns spent in observe phase
    _act_turns: int                            # Turns spent in act phase
    _retry_feedback: Optional[str]             # Feedback from reflect for retry
    _reflect_decision: Optional[str]           # "COMPLETE" | "RETRY_DESCRIPTOR" | "RETRY_PH"
    _reflect_history: List[Dict[str, Any]]    # All reflect decisions (one per round)
    _color_mode: Optional[str]               # "grayscale" or "per_channel" (from act phase)

    # Agentic v7: Genuine agency fields
    _observe_decisions: Optional[Dict[str, Any]]  # LLM's JSON decisions from OBSERVE R1
    _benchmark_stance: Optional[str]               # "FOLLOW" or "DEVIATE"
    _observe_ph_interpretation: Optional[str]      # LLM's PH interpretation text
    _per_channel_persistence: Optional[Dict[str, Any]]  # {"R": ph_data, "G": ph_data, "B": ph_data}
    _per_channel_images: Optional[Dict[str, Any]]       # {"R": array, "G": array, "B": array}
    _ph_signals: Optional[List[Dict[str, Any]]]         # PH-based signals from compute_ph_signals()

    # Agentic v8: 5-phase genuinely agentic pipeline fields
    _v8_mode: bool                                      # Whether running v8 pipeline
    _v8_perceive_turns: int                             # Turns spent in perceive ReAct loop
    _v8_image_analysis: Optional[Dict[str, Any]]        # Output from image_analyzer tool
    _v8_analysis_context: Optional[Dict[str, Any]]      # LLM's ANALYZE phase output (JSON)
    _v8_plan_context: Optional[Dict[str, Any]]          # LLM's PLAN phase output (JSON)
    _v8_reflect_experience: Optional[Dict[str, Any]]    # LLM's REFLECT experience entry for LTM
    _perceive_decisions: Optional[Dict[str, Any]]       # LLM's PERCEIVE_DECIDE output (filtration, denoising)

    # Agentic v9: Hypothesis-first genuinely agentic pipeline
    _v9_mode: bool                                      # Whether running v9 pipeline
    _v9_scrubbed_query: Optional[str]                   # Query with dataset-identifying info removed
    _v9_interpret_output: Optional[Dict[str, Any]]      # LLM#1 INTERPRET output (blind observation)
    _v9_hypothesis: Optional[Dict[str, Any]]            # LLM#2 ANALYZE output (hypothesis formation)
    _v9_act_decision: Optional[Dict[str, Any]]          # LLM#3 ACT output (reconciliation + decision)
    _v9_reflect_output: Optional[Dict[str, Any]]        # LLM#4 REFLECT output (evaluation + learning)
    _v9_memory_stats: Optional[Dict[str, Any]]          # Memory influence statistics


def create_initial_state(
    query: str,
    image_path: str,
    max_rounds: int = 4
) -> TopoAgentState:
    """Create an initial TopoAgentState.

    Args:
        query: User's query/task
        image_path: Path to input image
        max_rounds: Maximum reasoning rounds (default 4 for v2 pipeline)

    Returns:
        Initialized TopoAgentState
    """
    return TopoAgentState(
        query=query,
        image_path=image_path,
        short_term_memory=[],
        long_term_memory=[],
        current_round=0,
        max_rounds=max_rounds,
        reasoning_trace=[],
        messages=[],
        final_answer=None,
        confidence=0.0,
        evidence=[],
        task_complete=False,
        _tool_outputs=None,
        _current_reflection=None,
        llm_interactions=[],
        # Skills fields (None = not using skills mode)
        skill_descriptor=None,
        skill_params=None,
        skill_color_mode=None,
        # Optimized skills pipeline fields
        _plan=None,
        _ph_stats=None,
        _feature_quality=None,
        _needs_retry=False,
        _retry_count=0,
        _failed_descriptors=[],
        # Agentic pipeline fields (v6)
        _agentic_phase="observe",
        _observe_turns=0,
        _act_turns=0,
        _retry_feedback=None,
        _reflect_decision=None,
        _reflect_history=[],
        _color_mode=None,
        # Agentic v7: Genuine agency fields
        _observe_decisions=None,
        _benchmark_stance=None,
        _observe_ph_interpretation=None,
        _per_channel_persistence=None,
        _per_channel_images=None,
        _ph_signals=None,
        # Agentic v8: 5-phase pipeline fields
        _v8_mode=False,
        _v8_perceive_turns=0,
        _v8_image_analysis=None,
        _v8_analysis_context=None,
        _v8_plan_context=None,
        _v8_reflect_experience=None,
        _perceive_decisions=None,
        # Agentic v9 fields
        _v9_mode=False,
        _v9_scrubbed_query=None,
        _v9_interpret_output=None,
        _v9_hypothesis=None,
        _v9_act_decision=None,
        _v9_reflect_output=None,
        _v9_memory_stats=None,
    )


def update_short_term_memory(
    state: TopoAgentState,
    tool_name: str,
    output: Any
) -> TopoAgentState:
    """Update short-term memory with new tool output.

    Ms = Ms ∪ {(tool_t, output_t)}

    Args:
        state: Current state
        tool_name: Name of executed tool
        output: Tool output

    Returns:
        Updated state
    """
    new_memory = state["short_term_memory"].copy()
    new_memory.append((tool_name, output))
    return {**state, "short_term_memory": new_memory}


def update_long_term_memory(
    state: TopoAgentState,
    reflection: ReflectionEntry
) -> TopoAgentState:
    """Update long-term memory with new reflection entry.

    Ml = Ml ∪ {reflection_t}

    Args:
        state: Current state
        reflection: Reflection entry to add

    Returns:
        Updated state
    """
    new_memory = state["long_term_memory"].copy()
    new_memory.append(reflection)
    return {**state, "long_term_memory": new_memory}


def _summarize_output(tool_name: str, output: Any) -> str:
    """Create a concise summary of tool output for LLM context.

    This prevents memory explosion by not including full arrays/vectors
    in the prompt. Full data is stored in short_term_memory for
    programmatic access.

    Args:
        tool_name: Name of the tool
        output: Tool output dictionary

    Returns:
        Concise summary string (< 200 chars)
    """
    if not isinstance(output, dict):
        return str(output)[:200]

    if not output.get("success", True):
        error = output.get("error", "Unknown error")
        return f"FAILED: {error[:100]}"

    if tool_name == "image_loader":
        shape = output.get("shape", "unknown")
        # Calculate stats from image_array if present
        img_arr = output.get("image_array")
        if img_arr and isinstance(img_arr, list):
            try:
                import numpy as np
                arr = np.array(img_arr)
                mean_val = float(arr.mean())
                std_val = float(arr.std())
                return f"shape={shape}, mean={mean_val:.3f}, std={std_val:.3f}"
            except:
                pass
        return f"shape={shape}, loaded successfully"

    elif tool_name == "compute_ph":
        stats = output.get("statistics", {})
        h0_count = stats.get("H0", {}).get("count", 0)
        h1_count = stats.get("H1", {}).get("count", 0)
        filt_type = output.get("filtration_type", "unknown")
        return f"H0={h0_count} pts, H1={h1_count} pts, filtration={filt_type}"

    elif tool_name == "persistence_image":
        vec_len = output.get("vector_length", 0)
        nonzero = output.get("nonzero_ratio", 0)
        resolution = output.get("resolution", "unknown")
        return f"dim={vec_len}, nonzero={nonzero:.1%}, resolution={resolution}"

    elif tool_name == "image_analyzer":
        recs = output.get("recommendations", {})
        filt = recs.get("filtration_type", "unknown")
        sigma = recs.get("pi_sigma", 0)
        complexity = output.get("metrics", {}).get("complexity_score", 0)
        return f"recommended: filtration={filt}, sigma={sigma:.2f}, complexity={complexity:.2f}"

    elif tool_name == "pytorch_classifier":
        pred = output.get("predicted_class", "unknown")
        conf = output.get("confidence", 0)
        return f"prediction={pred}, confidence={conf:.1f}%"

    elif tool_name in ["persistence_landscapes", "persistence_silhouette", "betti_curves"]:
        vec_len = output.get("vector_length", output.get("total_dimension", 0))
        return f"dim={vec_len}"

    elif tool_name == "topological_features":
        n_feats = len(output.get("features", {}))
        return f"{n_feats} features extracted"

    elif tool_name == "binarization":
        shape = output.get("shape", "unknown")
        method = output.get("method", "unknown")
        fg = output.get("foreground_ratio", 0)
        return f"method={method}, shape={shape}, foreground={fg:.1%}"

    else:
        # Generic fallback - exclude large arrays
        summary_parts = []
        for key, value in output.items():
            if key in ["image_array", "combined_vector", "persistence", "vector",
                       "persistence_data", "feature_vector", "landscapes", "silhouette",
                       "binary_image"]:
                continue  # Skip large array fields
            if isinstance(value, (list, dict)) and len(str(value)) > 50:
                continue  # Skip large nested structures
            summary_parts.append(f"{key}={value}")
            if len(", ".join(summary_parts)) > 150:
                break
        return ", ".join(summary_parts)[:200] if summary_parts else "completed successfully"


def format_short_term_memory(state: TopoAgentState) -> str:
    """Format short-term memory for LLM context.

    Uses concise summaries to avoid memory explosion from large arrays.
    Full data remains in state["short_term_memory"] for programmatic access.

    Args:
        state: Current state

    Returns:
        Formatted string of recent tool executions (token-efficient)
    """
    if not state["short_term_memory"]:
        return "No previous tool executions in this session."

    lines = ["Recent tool executions:"]
    for i, (tool_name, output) in enumerate(state["short_term_memory"], 1):
        summary = _summarize_output(tool_name, output)
        lines.append(f"{i}. {tool_name}: {summary}")
    return "\n".join(lines)


def format_long_term_memory(state: TopoAgentState) -> str:
    """Format long-term memory for LLM context.

    Args:
        state: Current state

    Returns:
        Formatted string of past reflections/experiences
    """
    if not state["long_term_memory"]:
        return "No past experiences recorded."

    lines = ["Past experiences and reflections:"]
    for entry in state["long_term_memory"]:
        lines.append(f"Round {entry.round}:")
        lines.append(f"  - Analysis: {entry.error_analysis}")
        lines.append(f"  - Suggestion: {entry.suggestion}")
        lines.append(f"  - Experience: {entry.experience}")
    return "\n".join(lines)


# =============================================================================
# Auto-Injection Helpers (v2 Data Passing Fix)
# =============================================================================

def get_persistence_data(state: TopoAgentState) -> Optional[Dict[str, Any]]:
    """Extract persistence_data from compute_ph output in memory.

    This enables automatic data passing between tools, fixing the issue
    where LLM fails to pass data between tool calls.

    Args:
        state: Current state

    Returns:
        Persistence data dict or None if not found
    """
    for tool_name, output in reversed(state["short_term_memory"]):
        if tool_name == "compute_ph" and isinstance(output, dict):
            if output.get("success", False) and "persistence" in output:
                return output["persistence"]
    return None


def get_feature_vector(state: TopoAgentState) -> Optional[List[float]]:
    """Extract feature_vector from persistence_image output.

    This enables automatic data passing from PI to classifier.

    Args:
        state: Current state

    Returns:
        Combined feature vector or None if not found
    """
    for tool_name, output in reversed(state["short_term_memory"]):
        if tool_name == "persistence_image" and isinstance(output, dict):
            if output.get("success", False) and "combined_vector" in output:
                return output["combined_vector"]
    return None


def get_image_array(state: TopoAgentState) -> Optional[List]:
    """Extract image_array from image_loader output.

    Args:
        state: Current state

    Returns:
        Image array or None if not found
    """
    for tool_name, output in reversed(state["short_term_memory"]):
        if tool_name == "image_loader" and isinstance(output, dict):
            if output.get("success", False) and "image_array" in output:
                return output["image_array"]
    return None


# =============================================================================
# v3 Adaptive Pipeline Helpers
# =============================================================================

def get_image_analyzer_results(state: TopoAgentState) -> Optional[Dict[str, Any]]:
    """Extract recommendations from image_analyzer output.

    Used for adaptive configuration of compute_ph and persistence_image.

    Args:
        state: Current state

    Returns:
        Dictionary with recommendations or None if not found
    """
    for tool_name, output in reversed(state["short_term_memory"]):
        if tool_name == "image_analyzer" and isinstance(output, dict):
            if output.get("success", False) and "recommendations" in output:
                return output["recommendations"]
    return None


def get_recommended_filtration(state: TopoAgentState) -> Optional[str]:
    """Get recommended filtration type from image_analyzer.

    Args:
        state: Current state

    Returns:
        'sublevel' or 'superlevel' or None
    """
    recommendations = get_image_analyzer_results(state)
    if recommendations:
        return recommendations.get("filtration_type")
    return None


def get_recommended_sigma(state: TopoAgentState) -> Optional[float]:
    """Get recommended PI sigma from image_analyzer.

    Args:
        state: Current state

    Returns:
        Sigma value or None
    """
    recommendations = get_image_analyzer_results(state)
    if recommendations:
        return recommendations.get("pi_sigma")
    return None


# =============================================================================
# Skills System Helpers
# =============================================================================

# =============================================================================
# v5 State Definition
# =============================================================================

class TopoAgentStateV5(TypedDict):
    """State definition for TopoAgent v5 workflow.

    Streamlined state for the 2-3 LLM call architecture:
    - plan: Structured JSON from ANALYZE & PLAN step
    - execution_trace: Log of deterministic tool executions
    - verification: Results from VERIFY step
    - report: Final output report
    """
    # Input
    query: str
    image_path: str

    # V5 core state
    plan: Optional[Dict[str, Any]]           # JSON from ANALYZE_AND_PLAN
    execution_trace: List[Dict[str, Any]]     # Tool execution log
    verification: Optional[Dict[str, Any]]    # Verification result
    report: Optional[str]                     # Final report text
    feature_vector: Optional[Any]             # Extracted feature vector (numpy array)
    retry_used: bool                          # Whether alternative descriptor was tried

    # Internal data (for tool chain)
    _image_array: Optional[Any]              # Loaded image
    _persistence_data: Optional[Dict]        # PH output
    _ph_stats: Optional[Dict]               # PH statistics

    # Config
    verify_mode: str                          # "quick" or "thorough"
    dataset_name: Optional[str]              # For Mode B verification

    # LLM interaction logs
    llm_interactions: List[Any]

    # Skills fields (carried from v4)
    skill_descriptor: Optional[str]
    skill_params: Optional[Dict[str, Any]]
    skill_color_mode: Optional[str]


def create_initial_state_v5(
    query: str,
    image_path: str,
    verify_mode: str = "quick",
    dataset_name: Optional[str] = None,
) -> TopoAgentStateV5:
    """Create an initial state for v5 workflow.

    Args:
        query: User's query/task
        image_path: Path to input image
        verify_mode: "quick" (LLM reasoning) or "thorough" (TabPFN on reference batch)
        dataset_name: Dataset name for Mode B verification

    Returns:
        Initialized TopoAgentStateV5
    """
    return TopoAgentStateV5(
        query=query,
        image_path=image_path,
        plan=None,
        execution_trace=[],
        verification=None,
        report=None,
        feature_vector=None,
        retry_used=False,
        _image_array=None,
        _persistence_data=None,
        _ph_stats=None,
        verify_mode=verify_mode,
        dataset_name=dataset_name,
        llm_interactions=[],
        skill_descriptor=None,
        skill_params=None,
        skill_color_mode=None,
    )


def truncate_output_for_prompt(output, max_chars=2000):
    """Truncate tool output for inclusion in LLM prompts.

    Large arrays (images, persistence data, feature vectors) would use
    hundreds of thousands of tokens if stringified. We replace them with
    compact summaries while keeping the actual data available for auto-injection.
    """
    import numpy as np

    if isinstance(output, dict):
        truncated = {}
        for k, v in output.items():
            if isinstance(v, np.ndarray):
                truncated[k] = f"<ndarray shape={v.shape} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}>"
            elif isinstance(v, list):
                # Check if nested list (e.g. 224x224 image, 1x1818 binary)
                if v and isinstance(v[0], (list, np.ndarray)):
                    try:
                        arr = np.asarray(v)
                        truncated[k] = f"<array shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}>"
                    except (ValueError, TypeError):
                        truncated[k] = f"<nested_list len={len(v)}>"
                elif len(v) > 20:
                    truncated[k] = f"<list len={len(v)}>"
                elif len(str(v)) > max_chars:
                    truncated[k] = f"<list len={len(v)}>"
                else:
                    truncated[k] = v
            elif isinstance(v, dict) and len(str(v)) > max_chars:
                inner = {}
                for ik, iv in v.items():
                    if isinstance(iv, (list, np.ndarray)):
                        if isinstance(iv, np.ndarray):
                            inner[ik] = f"<ndarray shape={iv.shape}>"
                        elif len(iv) > 10:
                            inner[ik] = f"<list len={len(iv)}>"
                        else:
                            inner[ik] = iv
                    else:
                        inner[ik] = iv
                truncated[k] = inner
            elif isinstance(v, str) and len(v) > max_chars:
                truncated[k] = v[:max_chars] + "...[truncated]"
            else:
                truncated[k] = v
        return truncated
    else:
        s = str(output)
        if len(s) > max_chars:
            return s[:max_chars] + "...[truncated]"
        return output


# =============================================================================
# Skills System Helpers
# =============================================================================

def format_skill_context(state: TopoAgentState) -> str:
    """Format current skill state for injection into LLM prompts.

    Shows the currently selected descriptor and parameters (if any).
    This is a lightweight summary — the full knowledge context is
    injected separately via SkillRegistry.build_skill_context().

    Args:
        state: Current state with skill fields

    Returns:
        Formatted string of current skill state, or empty string
    """
    descriptor = state.get("skill_descriptor")
    if descriptor is None:
        return ""

    lines = ["## Current Skill Selection"]

    lines.append(f"Descriptor: {descriptor}")

    color_mode = state.get("skill_color_mode", "grayscale")
    lines.append(f"Color Mode: {color_mode}")

    params = state.get("skill_params", {})
    if params:
        lines.append("Applied Parameters:")
        for key, val in params.items():
            if key not in ("total_dim", "classifier", "color_mode", "dim"):
                lines.append(f"  - {key}: {val}")
        lines.append(f"  - total_dim: {params.get('total_dim', 'N/A')}")
        lines.append(f"  - classifier: {params.get('classifier', 'N/A')}")

    return "\n".join(lines)
