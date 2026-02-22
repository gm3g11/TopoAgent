"""Topological Features Tool for TopoAgent.

Extract statistical features from persistence diagrams.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class TopologicalFeaturesInput(BaseModel):
    """Input schema for TopologicalFeaturesTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )


class TopologicalFeaturesTool(BaseTool):
    """Extract statistical features from persistence diagrams.

    These features summarize the topological information in a form
    suitable for machine learning classifiers.
    """

    name: str = "topological_features"
    description: str = (
        "Extract statistical features from persistence diagrams for classification. "
        "Features include: persistence statistics, entropy, amplitude, Betti numbers. "
        "These features are numerical summaries of topological information. "
        "Input: persistence data from compute_ph tool. "
        "Output: feature vector with named features for each dimension."
    )
    args_schema: Type[BaseModel] = TopologicalFeaturesInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Extract topological features.

        Args:
            persistence_data: Persistence pairs by dimension

        Returns:
            Dictionary with extracted features
        """
        try:
            features = {}
            feature_vector = []
            feature_names = []

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]

                if not pairs or not isinstance(pairs, list):
                    # Add zeros for missing dimensions
                    dim_features = self._empty_features(dim_key)
                else:
                    # Skip if not proper persistence pairs
                    valid_pairs = [
                        p for p in pairs
                        if isinstance(p, dict) and "birth" in p and "death" in p
                    ]
                    if not valid_pairs:
                        dim_features = self._empty_features(dim_key)
                    else:
                        dim_features = self._compute_dimension_features(dim_key, valid_pairs)

                features[dim_key] = dim_features

                # Add to feature vector
                for name, value in dim_features.items():
                    feature_vector.append(value)
                    feature_names.append(f"{dim_key}_{name}")

            # Add cross-dimension features
            cross_features = self._compute_cross_dimension_features(features)
            features["cross_dimension"] = cross_features
            for name, value in cross_features.items():
                feature_vector.append(value)
                feature_names.append(f"cross_{name}")

            return {
                "success": True,
                "features_by_dimension": features,
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "num_features": len(feature_vector)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _compute_dimension_features(
        self,
        dim_key: str,
        pairs: List[Dict]
    ) -> Dict[str, float]:
        """Compute features for a single dimension.

        Args:
            dim_key: Dimension key (H0, H1, H2)
            pairs: Persistence pairs

        Returns:
            Dictionary of features
        """
        # Extract values
        births = [p["birth"] for p in pairs]
        deaths = [p["death"] for p in pairs]
        persistences = [p.get("persistence", abs(p["death"] - p["birth"])) for p in pairs]
        midpoints = [(b + d) / 2 for b, d in zip(births, deaths)]

        features = {}

        # Count features
        features["count"] = float(len(pairs))

        # Persistence statistics
        if persistences:
            features["pers_sum"] = float(sum(persistences))
            features["pers_mean"] = float(np.mean(persistences))
            features["pers_std"] = float(np.std(persistences))
            features["pers_max"] = float(max(persistences))
            features["pers_min"] = float(min(persistences))
            features["pers_median"] = float(np.median(persistences))

            # Percentiles
            features["pers_25th"] = float(np.percentile(persistences, 25))
            features["pers_75th"] = float(np.percentile(persistences, 75))
            features["pers_iqr"] = features["pers_75th"] - features["pers_25th"]

            # Persistence entropy
            total = sum(persistences)
            if total > 0:
                probs = [p / total for p in persistences]
                entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                features["entropy"] = float(entropy)
            else:
                features["entropy"] = 0.0

            # Amplitude (max persistence / range)
            birth_range = max(births) - min(births) if len(births) > 1 else 1
            features["amplitude"] = float(max(persistences) / max(birth_range, 1e-10))

            # Total persistence (L2 norm) - NEW
            features["pers_L2"] = float(np.sqrt(sum(p**2 for p in persistences)))

            # Persistence range - NEW
            features["pers_range"] = float(max(persistences) - min(persistences))

            # Normalized entropy - NEW
            if len(persistences) > 1:
                max_entropy = np.log(len(persistences))
                features["normalized_entropy"] = float(features["entropy"] / max(max_entropy, 1e-10))
            else:
                features["normalized_entropy"] = 0.0

            # 90th percentile - NEW
            features["pers_90th"] = float(np.percentile(persistences, 90))

        else:
            # No pairs - all zeros
            features["pers_sum"] = 0.0
            features["pers_mean"] = 0.0
            features["pers_std"] = 0.0
            features["pers_max"] = 0.0
            features["pers_min"] = 0.0
            features["pers_median"] = 0.0
            features["pers_25th"] = 0.0
            features["pers_75th"] = 0.0
            features["pers_iqr"] = 0.0
            features["entropy"] = 0.0
            features["amplitude"] = 0.0
            features["pers_L2"] = 0.0
            features["pers_range"] = 0.0
            features["normalized_entropy"] = 0.0
            features["pers_90th"] = 0.0

        # Birth/death statistics
        if births:
            features["birth_mean"] = float(np.mean(births))
            features["death_mean"] = float(np.mean(deaths))
            features["midpoint_mean"] = float(np.mean(midpoints))
        else:
            features["birth_mean"] = 0.0
            features["death_mean"] = 0.0
            features["midpoint_mean"] = 0.0

        return features

    def _empty_features(self, dim_key: str) -> Dict[str, float]:
        """Return empty features for a dimension.

        Args:
            dim_key: Dimension key

        Returns:
            Dictionary with all zero features
        """
        return {
            "count": 0.0,
            "pers_sum": 0.0,
            "pers_mean": 0.0,
            "pers_std": 0.0,
            "pers_max": 0.0,
            "pers_min": 0.0,
            "pers_median": 0.0,
            "pers_25th": 0.0,
            "pers_75th": 0.0,
            "pers_iqr": 0.0,
            "entropy": 0.0,
            "amplitude": 0.0,
            "pers_L2": 0.0,
            "pers_range": 0.0,
            "normalized_entropy": 0.0,
            "pers_90th": 0.0,
            "birth_mean": 0.0,
            "death_mean": 0.0,
            "midpoint_mean": 0.0
        }

    def _compute_cross_dimension_features(
        self,
        features: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute features that span multiple dimensions.

        Args:
            features: Features by dimension

        Returns:
            Cross-dimension features
        """
        cross = {}

        # Total persistence across all dimensions
        total_pers = sum(
            f.get("pers_sum", 0) for f in features.values()
            if isinstance(f, dict)
        )
        cross["total_persistence"] = float(total_pers)

        # Total feature count
        total_count = sum(
            f.get("count", 0) for f in features.values()
            if isinstance(f, dict)
        )
        cross["total_count"] = float(total_count)

        # H0/H1 ratio (components vs loops)
        h0_count = features.get("H0", {}).get("count", 0)
        h1_count = features.get("H1", {}).get("count", 0)
        cross["h0_h1_ratio"] = float(h0_count / max(h1_count, 1))

        # Entropy difference
        h0_entropy = features.get("H0", {}).get("entropy", 0)
        h1_entropy = features.get("H1", {}).get("entropy", 0)
        cross["entropy_diff"] = float(abs(h0_entropy - h1_entropy))

        # Total persistence L2 across all dimensions - NEW
        total_pers_L2_sq = sum(
            f.get("pers_L2", 0) ** 2 for f in features.values()
            if isinstance(f, dict)
        )
        cross["total_persistence_L2"] = float(np.sqrt(total_pers_L2_sq))

        # H0/H1 persistence ratio - NEW
        h0_pers = features.get("H0", {}).get("pers_sum", 0)
        h1_pers = features.get("H1", {}).get("pers_sum", 0)
        cross["h0_h1_pers_ratio"] = float(h0_pers / max(h1_pers, 1e-10))

        # H1/H2 ratio (if H2 exists) - NEW
        h2_count = features.get("H2", {}).get("count", 0)
        h2_pers = features.get("H2", {}).get("pers_sum", 0)
        if h2_count > 0:
            cross["h1_h2_ratio"] = float(h1_count / max(h2_count, 1))
            cross["h1_h2_pers_ratio"] = float(h1_pers / max(h2_pers, 1e-10))
        else:
            cross["h1_h2_ratio"] = 0.0
            cross["h1_h2_pers_ratio"] = 0.0

        return cross

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
