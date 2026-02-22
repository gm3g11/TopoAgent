"""Parameter Configuration Skill.

Configures descriptor parameters using deterministic lookup from benchmark rules.
No LLM call needed — this is pure rule application.
"""

from typing import Any, Dict

from .rules_data import (
    DIMENSION_RULES,
    PARAMETER_RULES,
    get_optimal_params,
    get_descriptor_dim,
    get_classifier,
)


def configure(descriptor: str, object_type: str, color_mode: str = "grayscale") -> Dict[str, Any]:
    """Get full parameter configuration for a descriptor.

    Merges dimension parameters and tunable parameters into a single dict.
    Also includes classifier recommendation and total dimension.

    Args:
        descriptor: Descriptor name
        object_type: Object type
        color_mode: 'grayscale' or 'per_channel'

    Returns:
        Dict with keys: all params + 'total_dim', 'classifier', 'color_mode'
    """
    params = get_optimal_params(descriptor, object_type)
    total_dim = get_descriptor_dim(descriptor, object_type, color_mode)
    classifier = get_classifier(descriptor, object_type, color_mode)

    return {
        **params,
        "total_dim": total_dim,
        "classifier": classifier,
        "color_mode": color_mode,
    }


def format_params_summary(descriptor: str, object_type: str, color_mode: str = "grayscale") -> str:
    """Format a human-readable summary of parameter configuration.

    Args:
        descriptor: Descriptor name
        object_type: Object type
        color_mode: 'grayscale' or 'per_channel'

    Returns:
        Formatted summary string
    """
    config = configure(descriptor, object_type, color_mode)

    lines = [f"Parameters for {descriptor} on {object_type} ({color_mode}):"]

    # Dimension params
    dim_rules = DIMENSION_RULES.get(descriptor, {})
    control = dim_rules.get("control_param", "N/A")
    if control in config:
        lines.append(f"  {control} = {config[control]}")

    # Extra dimension params (e.g., n_heights for ECT)
    for key in ["n_heights"]:
        if key in config and key != control:
            lines.append(f"  {key} = {config[key]}")

    # Tunable params
    param_rules = PARAMETER_RULES.get(descriptor, {})
    for param in param_rules.get("tunable_params", []):
        if param in config:
            lines.append(f"  {param} = {config[param]}")

    lines.append(f"  total_dim = {config['total_dim']}")
    lines.append(f"  classifier = {config['classifier']}")

    return "\n".join(lines)
