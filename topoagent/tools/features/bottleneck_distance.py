"""Bottleneck Distance Tool for TopoAgent.

Compare persistence diagrams using bottleneck distance.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class BottleneckDistanceInput(BaseModel):
    """Input schema for BottleneckDistanceTool."""
    diagram1: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="First persistence diagram"
    )
    diagram2: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Second persistence diagram"
    )


class BottleneckDistanceTool(BaseTool):
    """Compare persistence diagrams using bottleneck distance.

    Bottleneck distance measures the maximum matching cost between
    two persistence diagrams (infinity-norm transport).
    """

    name: str = "bottleneck_distance"
    description: str = (
        "Compare two persistence diagrams using bottleneck distance. "
        "Bottleneck distance is the maximum (worst-case) matching cost. "
        "More sensitive to outliers than Wasserstein distance. "
        "Use when detecting presence/absence of specific features matters. "
        "Input: two persistence diagrams. "
        "Output: distance value for each dimension."
    )
    args_schema: Type[BaseModel] = BottleneckDistanceInput

    def _run(
        self,
        diagram1: Dict[str, List[Dict[str, float]]],
        diagram2: Dict[str, List[Dict[str, float]]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute bottleneck distance between diagrams.

        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram

        Returns:
            Dictionary with distances by dimension
        """
        try:
            distances = {}

            # Get all dimensions
            all_dims = set(diagram1.keys()) | set(diagram2.keys())

            for dim in sorted(all_dims):
                pairs1 = self._extract_points(diagram1.get(dim, []))
                pairs2 = self._extract_points(diagram2.get(dim, []))

                # Compute bottleneck distance
                try:
                    # Try persim library
                    import persim
                    if pairs1.size > 0 and pairs2.size > 0:
                        dist = persim.bottleneck(pairs1, pairs2)
                    elif pairs1.size == 0 and pairs2.size == 0:
                        dist = 0.0
                    else:
                        # One empty
                        non_empty = pairs1 if pairs1.size > 0 else pairs2
                        persistences = np.abs(non_empty[:, 1] - non_empty[:, 0])
                        dist = float(np.max(persistences) / 2)
                except ImportError:
                    # Fallback
                    dist = self._compute_bottleneck(pairs1, pairs2)

                distances[dim] = float(dist)

            total_distance = max(distances.values()) if distances else 0.0

            return {
                "success": True,
                "distances_by_dimension": distances,
                "max_distance": total_distance,
                "interpretation": self._interpret_distances(distances)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_points(self, pairs: List[Dict]) -> np.ndarray:
        """Extract birth-death points.

        Args:
            pairs: Persistence pairs

        Returns:
            Nx2 array
        """
        if not pairs or not isinstance(pairs, list):
            return np.array([]).reshape(0, 2)

        valid_pairs = [
            (p["birth"], p["death"]) for p in pairs
            if isinstance(p, dict) and "birth" in p and "death" in p
        ]

        if not valid_pairs:
            return np.array([]).reshape(0, 2)

        return np.array(valid_pairs)

    def _compute_bottleneck(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> float:
        """Compute bottleneck distance (fallback).

        Args:
            points1: First diagram
            points2: Second diagram

        Returns:
            Bottleneck distance
        """
        if points1.size == 0 and points2.size == 0:
            return 0.0

        if points1.size == 0:
            persistences = np.abs(points2[:, 1] - points2[:, 0])
            return float(np.max(persistences) / 2)

        if points2.size == 0:
            persistences = np.abs(points1[:, 1] - points1[:, 0])
            return float(np.max(persistences) / 2)

        # Compute pairwise distances (L-infinity)
        from scipy.spatial.distance import cdist

        # Include diagonal projections
        diag1 = np.column_stack([
            (points1[:, 0] + points1[:, 1]) / 2,
            (points1[:, 0] + points1[:, 1]) / 2
        ])
        diag2 = np.column_stack([
            (points2[:, 0] + points2[:, 1]) / 2,
            (points2[:, 0] + points2[:, 1]) / 2
        ])

        n1, n2 = len(points1), len(points2)

        # Extended diagrams
        ext1 = np.vstack([points1, diag2])
        ext2 = np.vstack([points2, diag1])

        # L-infinity distance
        cost = cdist(ext1, ext2, metric='chebyshev')

        # Binary search for bottleneck
        candidates = np.unique(cost)

        def can_match(threshold):
            """Check if matching exists with given threshold."""
            from scipy.optimize import linear_sum_assignment
            valid = (cost <= threshold).astype(float)
            valid[valid == 0] = 1e10
            valid[valid == 1] = 0

            try:
                row_ind, col_ind = linear_sum_assignment(valid)
                return valid[row_ind, col_ind].max() < 1
            except:
                return False

        # Binary search
        low, high = 0, len(candidates) - 1
        while low < high:
            mid = (low + high) // 2
            if can_match(candidates[mid]):
                high = mid
            else:
                low = mid + 1

        return float(candidates[min(low, len(candidates) - 1)])

    def _interpret_distances(self, distances: Dict[str, float]) -> str:
        """Interpret distances.

        Args:
            distances: Distances by dimension

        Returns:
            Interpretation
        """
        interpretations = []

        for dim, dist in sorted(distances.items()):
            if dist < 0.05:
                level = "nearly identical"
            elif dist < 0.15:
                level = "similar"
            elif dist < 0.3:
                level = "moderately different"
            else:
                level = "significantly different"

            interpretations.append(f"{dim}: {level} (max-diff={dist:.3f})")

        return "; ".join(interpretations)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
