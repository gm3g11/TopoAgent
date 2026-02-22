"""Descriptor Selection Skill — Knowledge Generator.

Builds rich knowledge context for LLM-based descriptor selection.
Instead of deterministically picking a descriptor, this module provides
the LLM with descriptor properties, reasoning chains, and benchmark
rankings so it can make an informed decision.
"""

from typing import Dict, List, Optional

from .rules_data import (
    TOP_PERFORMERS,
    SUPPORTED_DESCRIPTORS,
    DESCRIPTOR_PROPERTIES,
    REASONING_CHAINS,
    get_top_descriptors,
    get_optimal_params,
    get_descriptor_dim,
    build_descriptor_knowledge_text,
)


def build_knowledge_context(
    object_type_hint: Optional[str] = None,
    color_mode: str = "grayscale",
    supported_only: bool = True,
) -> str:
    """Build rich knowledge context for LLM descriptor selection.

    Returns a formatted string with:
    1. Descriptor properties (what each captures, when to use)
    2. Reasoning chains (image property -> descriptor match)
    3. Benchmark rankings (if object_type known)
    4. Parameter preview (what params will be applied)

    Args:
        object_type_hint: If known, include specific benchmark rankings
        color_mode: 'grayscale' or 'per_channel'
        supported_only: If True, exclude training-based descriptors (ATOL,
            persistence_codebook) which require training data.

    Returns:
        Formatted knowledge string for prompt injection
    """
    return build_descriptor_knowledge_text(
        object_type_hint=object_type_hint,
        color_mode=color_mode,
        include_rankings=True,
        supported_only=supported_only,
    )


def get_all_descriptor_info(object_type: str) -> List[Dict]:
    """Get info for all 13 supported descriptors for an object type.

    Args:
        object_type: Object type

    Returns:
        List of dicts with descriptor info, sorted by accuracy (best first)
    """
    # Build accuracy lookup from top performers
    acc_lookup = {}
    for entry in TOP_PERFORMERS.get(object_type, []):
        acc_lookup[entry["descriptor"]] = entry["accuracy"]

    result = []
    for desc in SUPPORTED_DESCRIPTORS:
        params = get_optimal_params(desc, object_type)
        result.append({
            "descriptor": desc,
            "accuracy": acc_lookup.get(desc),
            "dim": params.get("dim", 0),
            "params": params,
        })

    # Sort by accuracy (known first, descending), then unknown
    result.sort(key=lambda x: (x["accuracy"] is not None, x["accuracy"] or 0), reverse=True)
    return result


def build_selection_prompt(object_type: str, color_mode: str = "grayscale") -> str:
    """Build prompt for LLM-assisted descriptor selection.

    Presents benchmark-validated rankings and asks LLM to confirm or override.

    Args:
        object_type: Object type
        color_mode: 'grayscale' or 'per_channel'

    Returns:
        Formatted prompt string
    """
    top3 = get_top_descriptors(object_type, n=3)
    all_descriptors = get_all_descriptor_info(object_type)

    # Format top 3
    top_lines = []
    for i, entry in enumerate(top3, 1):
        dim = entry["dim"]
        if color_mode == "per_channel":
            dim *= 3
        top_lines.append(
            f"{i}. **{entry['descriptor']}** — accuracy {entry['accuracy']:.1%}, "
            f"dim={dim}"
        )

    # Format full table
    table_lines = ["| Descriptor | Accuracy | Dim |"]
    table_lines.append("|---|---|---|")
    for entry in all_descriptors:
        acc = f"{entry['accuracy']:.1%}" if entry["accuracy"] else "N/A"
        dim = entry["dim"]
        if color_mode == "per_channel":
            dim *= 3
        table_lines.append(f"| {entry['descriptor']} | {acc} | {dim} |")

    return f"""Select the best TDA descriptor for **{object_type}** images.

## Benchmark-Validated Rankings
Top performers for {object_type}:
{chr(10).join(top_lines)}

## All Available Descriptors ({color_mode} mode)
{chr(10).join(table_lines)}

## Decision
Select ONE descriptor. Prefer the top-ranked unless you have specific reason to deviate.
Output format: DESCRIPTOR: <name>
"""
