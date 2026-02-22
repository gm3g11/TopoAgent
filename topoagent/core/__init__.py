"""TopoAgent Core: Deterministic, Auditable Feature Extraction.

This module provides deterministic topology feature extraction without
LLM dependency, ensuring reproducibility and auditability for medical AI.
"""

from .topo_features import (
    extract_topo_features,
    TopoConfig,
    TopoFeatureResult,
)

__all__ = [
    "extract_topo_features",
    "TopoConfig",
    "TopoFeatureResult",
]
