"""TopoAgent Skills System.

Encodes benchmark-validated rules (from Benchmark3/4: 15 descriptors x 5 object types)
as knowledge that the LLM uses to make informed descriptor/parameter choices.

Design principle: Skills suggest with reasoning -> LLM confirms or overrides
-> parameters auto-applied from benchmark rules once descriptor is chosen.

Usage:
    from topoagent.skills import SkillRegistry

    registry = SkillRegistry()

    # Build knowledge context for LLM prompt injection
    knowledge = registry.build_skill_context(object_type_hint="discrete_cells")

    # After LLM picks a descriptor, get benchmark-validated params
    params = registry.configure_after_selection(
        descriptor="template_functions",
        object_type="discrete_cells",
        color_mode="grayscale",
    )

Individual skills can also be used directly:
    from topoagent.skills import descriptor_skill, parameter_skill
    knowledge = descriptor_skill.build_knowledge_context("discrete_cells")
    params = parameter_skill.configure("template_functions", "discrete_cells")
"""

from typing import Any, Dict, Optional

from . import descriptor_skill
from . import parameter_skill
from . import color_mode_skill
from .rules_data import (
    SUPPORTED_DESCRIPTORS,
    ALL_DESCRIPTORS,
    OBJECT_TYPES,
    OBJECT_TYPE_KEYWORDS,
    TOP_PERFORMERS,
    DATASET_TO_OBJECT_TYPE,
    get_optimal_params,
    get_top_descriptors,
    get_descriptor_dim,
    get_classifier,
    get_object_type_for_dataset,
    get_color_mode_for_dataset,
)


class SkillRegistry:
    """Registry and orchestrator for all skills.

    Provides a unified interface for the workflow to:
    1. Build knowledge context for LLM descriptor selection
    2. Configure parameters after the LLM chooses a descriptor
    3. (v5) Include learned rules and past reflections in context
    """

    def __init__(self):
        self.skills = {
            "descriptor": descriptor_skill,
            "parameter": parameter_skill,
            "color_mode": color_mode_skill,
        }
        self._skill_memory = None

    @property
    def skill_memory(self):
        """Lazy-load SkillMemory to avoid circular imports."""
        if self._skill_memory is None:
            from ..memory.skill_memory import SkillMemory
            self._skill_memory = SkillMemory()
        return self._skill_memory

    def build_skill_context(
        self,
        object_type_hint: Optional[str] = None,
        color_mode: str = "grayscale",
        include_learned: bool = True,
        supported_only: bool = True,
    ) -> str:
        """Build knowledge context string for injection into LLM prompts.

        Returns descriptor properties, reasoning chains, benchmark
        rankings, and (v5) learned rules + reflections.

        Args:
            object_type_hint: If known, include specific benchmark rankings
            color_mode: 'grayscale' or 'per_channel'
            include_learned: Whether to include learned rules/reflections (v5)
            supported_only: If True, exclude training-based descriptors (ATOL,
                persistence_codebook) which require training data.

        Returns:
            Formatted knowledge string
        """
        base = descriptor_skill.build_knowledge_context(
            object_type_hint=object_type_hint,
            color_mode=color_mode,
            supported_only=supported_only,
        )

        if include_learned:
            learned = self.skill_memory.get_learned_context(object_type_hint)
            if learned and learned != "No learned rules or reflections yet.":
                base = base + "\n\n" + learned

        return base

    def configure_after_selection(
        self,
        descriptor: str,
        object_type: str,
        color_mode: str = "grayscale",
    ) -> Dict[str, Any]:
        """After LLM picks descriptor, return benchmark-validated params.

        Args:
            descriptor: Descriptor name chosen by the LLM
            object_type: Object type (identified by LLM or from dataset)
            color_mode: 'grayscale' or 'per_channel'

        Returns:
            Dict with all optimal parameters + total_dim, classifier, color_mode
        """
        return parameter_skill.configure(descriptor, object_type, color_mode)

    def select_color_mode(
        self,
        dataset_name: Optional[str] = None,
        n_channels: Optional[int] = None,
        image_path: Optional[str] = None,
    ) -> str:
        """Select grayscale or per_channel mode."""
        return color_mode_skill.select(dataset_name, n_channels, image_path)

    def update_skills(
        self,
        agent_choice: str,
        agent_accuracy: float,
        oracle_choice: Optional[str] = None,
        oracle_accuracy: Optional[float] = None,
        object_type: str = "surface_lesions",
        dataset: str = "unknown",
        params_used: Optional[Dict] = None,
    ) -> None:
        """Update skill memory after benchmark evaluation.

        Args:
            agent_choice: Descriptor the agent selected
            agent_accuracy: Accuracy achieved
            oracle_choice: Best descriptor (if known)
            oracle_accuracy: Best accuracy (if known)
            object_type: Object type
            dataset: Dataset name
            params_used: Parameters used
        """
        self.skill_memory.update_from_benchmark(
            agent_choice=agent_choice,
            agent_accuracy=agent_accuracy,
            oracle_choice=oracle_choice,
            oracle_accuracy=oracle_accuracy,
            object_type=object_type,
            dataset=dataset,
            params_used=params_used,
        )

    def infer_object_type_hint(
        self,
        query: str,
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Try to infer object type from query/path for ranking context.

        Uses keyword matching and dataset name lookup. Returns None if
        uncertain — the LLM will identify the object type via reasoning.

        Args:
            query: User query text
            image_path: Optional image file path

        Returns:
            Object type string or None if not confidently determined
        """
        query_lower = query.lower()

        # Check for known dataset names in query
        for ds_name, obj_type in DATASET_TO_OBJECT_TYPE.items():
            if ds_name.lower() in query_lower:
                return obj_type

        # Check image path for dataset names
        if image_path:
            path_lower = image_path.lower()
            for ds_name, obj_type in DATASET_TO_OBJECT_TYPE.items():
                if ds_name.lower() in path_lower:
                    return obj_type

        # Keyword scoring (only return if strong match)
        scores = {}
        for obj_type, keywords in OBJECT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[obj_type] = score

        if image_path:
            path_lower = image_path.lower()
            for obj_type, keywords in OBJECT_TYPE_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in path_lower)
                if score > 0:
                    scores[obj_type] = scores.get(obj_type, 0) + score

        if scores:
            best = max(scores, key=scores.get)
            # Only return if reasonably confident (>=2 keyword matches)
            if scores[best] >= 2:
                return best
            # Single keyword match — still return as hint
            return best

        return None


__all__ = [
    "SkillRegistry",
    "descriptor_skill",
    "parameter_skill",
    "color_mode_skill",
    # Re-export commonly used functions
    "get_optimal_params",
    "get_top_descriptors",
    "get_descriptor_dim",
    "get_classifier",
    "get_object_type_for_dataset",
    "get_color_mode_for_dataset",
    "SUPPORTED_DESCRIPTORS",
    "ALL_DESCRIPTORS",
    "OBJECT_TYPES",
    "TOP_PERFORMERS",
]
