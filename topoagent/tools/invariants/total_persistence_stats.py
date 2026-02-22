"""Total Persistence Statistics Tool for TopoAgent.

Compute comprehensive lifespan and persistence statistics from persistence diagrams.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class TotalPersistenceStatsInput(BaseModel):
    """Input schema for TotalPersistenceStatsTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool (format: {'H0': [{'birth': x, 'death': y}, ...], 'H1': [...]})"
    )
    power: float = Field(
        1.0, description="Power for L^p total persistence computation (default: 1.0)"
    )


class TotalPersistenceStatsTool(BaseTool):
    """Compute total persistence and lifespan statistics.

    This tool provides comprehensive statistics about persistence features:
    - Total persistence (L^p norms)
    - Lifespan distribution statistics
    - Feature counts and density metrics
    - Cross-dimension comparisons

    References:
    - Total persistence is a standard feature in medical TDA papers
    - Used in Giotto-TDA, GUDHI, and other TDA libraries
    """

    name: str = "total_persistence_stats"
    description: str = (
        "Compute total persistence and lifespan statistics from persistence diagrams. "
        "Provides L^p total persistence, lifespan distribution stats, and feature counts. "
        "Input: persistence data from compute_ph tool. "
        "Output: comprehensive persistence statistics by dimension and combined. "
        "Useful for quick numerical summaries of topological complexity."
    )
    args_schema: Type[BaseModel] = TotalPersistenceStatsInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        power: float = 1.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute total persistence statistics.

        Args:
            persistence_data: Persistence pairs by dimension
            power: Power for L^p computation

        Returns:
            Dictionary with comprehensive persistence statistics
        """
        try:
            stats_by_dim = {}
            all_persistences = []
            all_lifespans = []

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]

                if not pairs or not isinstance(pairs, list):
                    stats_by_dim[dim_key] = self._empty_stats()
                    continue

                # Extract valid pairs
                valid_pairs = [
                    p for p in pairs
                    if isinstance(p, dict) and "birth" in p and "death" in p
                ]

                if not valid_pairs:
                    stats_by_dim[dim_key] = self._empty_stats()
                    continue

                # Compute statistics for this dimension
                dim_stats = self._compute_dimension_stats(valid_pairs, power)
                stats_by_dim[dim_key] = dim_stats

                # Collect for cross-dimension stats
                all_persistences.extend(dim_stats["persistence_values"])
                all_lifespans.extend(dim_stats["lifespan_values"])

            # Compute cross-dimension statistics
            cross_stats = self._compute_cross_dimension_stats(
                stats_by_dim, all_persistences, all_lifespans, power
            )

            # Create feature vector for ML
            feature_vector, feature_names = self._create_feature_vector(
                stats_by_dim, cross_stats
            )

            return {
                "success": True,
                "tool_name": self.name,
                "stats_by_dimension": {
                    k: {key: val for key, val in v.items()
                        if key not in ["persistence_values", "lifespan_values"]}
                    for k, v in stats_by_dim.items()
                },
                "cross_dimension_stats": cross_stats,
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "num_features": len(feature_vector),
                "power": power,
                "interpretation": self._generate_interpretation(stats_by_dim, cross_stats)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_dimension_stats(
        self,
        pairs: List[Dict],
        power: float
    ) -> Dict[str, Any]:
        """Compute statistics for a single dimension.

        Args:
            pairs: List of persistence pairs
            power: Power for L^p computation

        Returns:
            Dictionary with dimension statistics
        """
        births = np.array([p["birth"] for p in pairs])
        deaths = np.array([p["death"] for p in pairs])
        persistences = np.array([
            p.get("persistence", abs(p["death"] - p["birth"]))
            for p in pairs
        ])

        # Filter out infinite persistence (keep finite only)
        finite_mask = np.isfinite(persistences)
        finite_persistences = persistences[finite_mask]

        if len(finite_persistences) == 0:
            return self._empty_stats()

        # Lifespan = death - birth (same as persistence for finite features)
        lifespans = finite_persistences

        stats = {
            "count": len(pairs),
            "finite_count": len(finite_persistences),

            # Total persistence (L^p norms)
            "total_persistence_L1": float(np.sum(finite_persistences)),
            "total_persistence_L2": float(np.sqrt(np.sum(finite_persistences ** 2))),
            "total_persistence_Lp": float(np.sum(finite_persistences ** power) ** (1/power)) if power > 0 else 0,
            "total_persistence_Linf": float(np.max(finite_persistences)),

            # Lifespan statistics
            "lifespan_mean": float(np.mean(lifespans)),
            "lifespan_std": float(np.std(lifespans)),
            "lifespan_median": float(np.median(lifespans)),
            "lifespan_min": float(np.min(lifespans)),
            "lifespan_max": float(np.max(lifespans)),
            "lifespan_range": float(np.max(lifespans) - np.min(lifespans)),

            # Percentiles
            "lifespan_25th": float(np.percentile(lifespans, 25)),
            "lifespan_75th": float(np.percentile(lifespans, 75)),
            "lifespan_iqr": float(np.percentile(lifespans, 75) - np.percentile(lifespans, 25)),
            "lifespan_90th": float(np.percentile(lifespans, 90)),

            # Distribution shape
            "lifespan_skewness": float(self._skewness(lifespans)),
            "lifespan_kurtosis": float(self._kurtosis(lifespans)),

            # Birth/death statistics
            "birth_mean": float(np.mean(births)),
            "birth_std": float(np.std(births)),
            "death_mean": float(np.mean(deaths)),
            "death_std": float(np.std(deaths)),

            # Persistence entropy
            "persistence_entropy": float(self._entropy(finite_persistences)),
            "normalized_entropy": float(self._normalized_entropy(finite_persistences)),

            # Store values for cross-dimension computation (will be removed from output)
            "persistence_values": finite_persistences.tolist(),
            "lifespan_values": lifespans.tolist()
        }

        return stats

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics."""
        return {
            "count": 0,
            "finite_count": 0,
            "total_persistence_L1": 0.0,
            "total_persistence_L2": 0.0,
            "total_persistence_Lp": 0.0,
            "total_persistence_Linf": 0.0,
            "lifespan_mean": 0.0,
            "lifespan_std": 0.0,
            "lifespan_median": 0.0,
            "lifespan_min": 0.0,
            "lifespan_max": 0.0,
            "lifespan_range": 0.0,
            "lifespan_25th": 0.0,
            "lifespan_75th": 0.0,
            "lifespan_iqr": 0.0,
            "lifespan_90th": 0.0,
            "lifespan_skewness": 0.0,
            "lifespan_kurtosis": 0.0,
            "birth_mean": 0.0,
            "birth_std": 0.0,
            "death_mean": 0.0,
            "death_std": 0.0,
            "persistence_entropy": 0.0,
            "normalized_entropy": 0.0,
            "persistence_values": [],
            "lifespan_values": []
        }

    def _compute_cross_dimension_stats(
        self,
        stats_by_dim: Dict[str, Dict],
        all_persistences: List[float],
        all_lifespans: List[float],
        power: float
    ) -> Dict[str, Any]:
        """Compute statistics across all dimensions.

        Args:
            stats_by_dim: Statistics by dimension
            all_persistences: All persistence values combined
            all_lifespans: All lifespan values combined
            power: Power for L^p computation

        Returns:
            Cross-dimension statistics
        """
        all_pers = np.array(all_persistences) if all_persistences else np.array([0])
        all_life = np.array(all_lifespans) if all_lifespans else np.array([0])

        # Get individual dimension totals
        h0_total = stats_by_dim.get("H0", {}).get("total_persistence_L1", 0)
        h1_total = stats_by_dim.get("H1", {}).get("total_persistence_L1", 0)
        h2_total = stats_by_dim.get("H2", {}).get("total_persistence_L1", 0)

        h0_count = stats_by_dim.get("H0", {}).get("count", 0)
        h1_count = stats_by_dim.get("H1", {}).get("count", 0)
        h2_count = stats_by_dim.get("H2", {}).get("count", 0)

        return {
            # Combined total persistence
            "total_persistence_all_L1": float(np.sum(all_pers)),
            "total_persistence_all_L2": float(np.sqrt(np.sum(all_pers ** 2))),
            "total_persistence_all_Linf": float(np.max(all_pers)) if len(all_pers) > 0 else 0,

            # Combined statistics
            "total_feature_count": int(h0_count + h1_count + h2_count),
            "combined_lifespan_mean": float(np.mean(all_life)) if len(all_life) > 0 else 0,
            "combined_lifespan_std": float(np.std(all_life)) if len(all_life) > 0 else 0,

            # Dimension ratios
            "H0_H1_persistence_ratio": float(h0_total / max(h1_total, 1e-10)),
            "H1_H2_persistence_ratio": float(h1_total / max(h2_total, 1e-10)) if h2_total > 0 else 0,
            "H0_H1_count_ratio": float(h0_count / max(h1_count, 1)),
            "H1_H2_count_ratio": float(h1_count / max(h2_count, 1)) if h2_count > 0 else 0,

            # Persistence distribution across dimensions
            "H0_persistence_fraction": float(h0_total / max(np.sum(all_pers), 1e-10)),
            "H1_persistence_fraction": float(h1_total / max(np.sum(all_pers), 1e-10)),
            "H2_persistence_fraction": float(h2_total / max(np.sum(all_pers), 1e-10)),

            # Combined entropy
            "combined_entropy": float(self._entropy(all_pers)),
        }

    def _create_feature_vector(
        self,
        stats_by_dim: Dict[str, Dict],
        cross_stats: Dict[str, Any]
    ) -> tuple:
        """Create feature vector for ML.

        Args:
            stats_by_dim: Statistics by dimension
            cross_stats: Cross-dimension statistics

        Returns:
            Tuple of (feature_vector, feature_names)
        """
        features = []
        names = []

        # Per-dimension features
        key_features = [
            "total_persistence_L1", "total_persistence_L2",
            "lifespan_mean", "lifespan_std", "lifespan_max",
            "persistence_entropy", "count"
        ]

        for dim in ["H0", "H1", "H2"]:
            dim_stats = stats_by_dim.get(dim, self._empty_stats())
            for feat in key_features:
                features.append(dim_stats.get(feat, 0))
                names.append(f"{dim}_{feat}")

        # Cross-dimension features
        cross_features = [
            "total_persistence_all_L1", "total_feature_count",
            "H0_H1_persistence_ratio", "H0_H1_count_ratio",
            "combined_entropy"
        ]
        for feat in cross_features:
            features.append(cross_stats.get(feat, 0))
            names.append(f"cross_{feat}")

        return features, names

    def _entropy(self, values: np.ndarray) -> float:
        """Compute persistence entropy.

        Args:
            values: Persistence values

        Returns:
            Entropy value
        """
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0

        probs = values / np.sum(values)
        probs = probs[probs > 0]  # Remove zeros
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def _normalized_entropy(self, values: np.ndarray) -> float:
        """Compute normalized entropy (0 to 1 scale).

        Args:
            values: Persistence values

        Returns:
            Normalized entropy
        """
        if len(values) <= 1:
            return 0.0

        entropy = self._entropy(values)
        max_entropy = np.log(len(values))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _skewness(self, values: np.ndarray) -> float:
        """Compute skewness of distribution.

        Args:
            values: Array of values

        Returns:
            Skewness value
        """
        if len(values) < 3:
            return 0.0

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0

        n = len(values)
        return float(n / ((n-1) * (n-2)) * np.sum(((values - mean) / std) ** 3))

    def _kurtosis(self, values: np.ndarray) -> float:
        """Compute excess kurtosis of distribution.

        Args:
            values: Array of values

        Returns:
            Excess kurtosis value
        """
        if len(values) < 4:
            return 0.0

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0

        n = len(values)
        m4 = np.mean((values - mean) ** 4)
        return float(m4 / (std ** 4) - 3)

    def _generate_interpretation(
        self,
        stats_by_dim: Dict[str, Dict],
        cross_stats: Dict[str, Any]
    ) -> str:
        """Generate human-readable interpretation.

        Args:
            stats_by_dim: Statistics by dimension
            cross_stats: Cross-dimension statistics

        Returns:
            Interpretation string
        """
        total_L1 = cross_stats.get("total_persistence_all_L1", 0)
        total_count = cross_stats.get("total_feature_count", 0)
        h0_fraction = cross_stats.get("H0_persistence_fraction", 0)

        if total_count == 0:
            return "No persistence features detected."

        complexity = "low" if total_L1 < 1 else "moderate" if total_L1 < 10 else "high"

        parts = [
            f"Total persistence (L1): {total_L1:.3f} ({complexity} topological complexity)",
            f"Total features: {total_count}"
        ]

        if h0_fraction > 0.8:
            parts.append("Dominated by connected components (H0)")
        elif h0_fraction < 0.3:
            parts.append("Significant higher-dimensional topology (loops/cavities)")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
