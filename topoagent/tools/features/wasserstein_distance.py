"""Wasserstein Distance Tool for TopoAgent.

Compare persistence diagrams using Wasserstein distance.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class WassersteinDistanceInput(BaseModel):
    """Input schema for WassersteinDistanceTool."""
    diagram1: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="First persistence diagram"
    )
    diagram2: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Second persistence diagram"
    )
    p: int = Field(2, description="Order of Wasserstein distance (1 or 2)")


class WassersteinDistanceTool(BaseTool):
    """Compare persistence diagrams using Wasserstein distance.

    Wasserstein distance measures the optimal transport cost between
    two persistence diagrams.
    """

    name: str = "wasserstein_distance"
    description: str = (
        "Compare two persistence diagrams using Wasserstein distance. "
        "Wasserstein distance measures similarity between topological signatures. "
        "Lower distance = more similar topological structure. "
        "Use this to compare images with reference patterns or templates. "
        "Input: two persistence diagrams. "
        "Output: distance value for each dimension."
    )
    args_schema: Type[BaseModel] = WassersteinDistanceInput

    def _run(
        self,
        diagram1: Dict[str, List[Dict[str, float]]],
        diagram2: Dict[str, List[Dict[str, float]]],
        p: int = 2,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Wasserstein distance between diagrams.

        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            p: Order of Wasserstein distance

        Returns:
            Dictionary with distances by dimension
        """
        try:
            distances = {}

            # Get all dimensions present in either diagram
            all_dims = set(diagram1.keys()) | set(diagram2.keys())

            for dim in sorted(all_dims):
                pairs1 = self._extract_points(diagram1.get(dim, []))
                pairs2 = self._extract_points(diagram2.get(dim, []))

                # Compute Wasserstein distance
                try:
                    # Try to use persim library
                    import persim
                    if pairs1.size > 0 and pairs2.size > 0:
                        dist = persim.wasserstein(pairs1, pairs2, matching=False)
                    elif pairs1.size == 0 and pairs2.size == 0:
                        dist = 0.0
                    else:
                        # One empty - compute distance to diagonal
                        dist = self._distance_to_diagonal(
                            pairs1 if pairs1.size > 0 else pairs2, p
                        )
                except ImportError:
                    # Fallback implementation
                    dist = self._compute_wasserstein(pairs1, pairs2, p)

                distances[dim] = float(dist)

            # Compute total distance (sum across dimensions)
            total_distance = sum(distances.values())

            return {
                "success": True,
                "distances_by_dimension": distances,
                "total_distance": total_distance,
                "order": p,
                "interpretation": self._interpret_distances(distances)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_points(self, pairs: List[Dict]) -> np.ndarray:
        """Extract birth-death points from persistence pairs.

        Args:
            pairs: List of persistence pair dictionaries

        Returns:
            Nx2 array of (birth, death) points
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

    def _compute_wasserstein(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        p: int
    ) -> float:
        """Compute Wasserstein distance (fallback implementation).

        This is an approximation using simple matching.

        Args:
            points1: First diagram points
            points2: Second diagram points
            p: Order

        Returns:
            Approximate Wasserstein distance
        """
        if points1.size == 0 and points2.size == 0:
            return 0.0

        if points1.size == 0:
            return self._distance_to_diagonal(points2, p)
        if points2.size == 0:
            return self._distance_to_diagonal(points1, p)

        # Project points to diagonal
        diag1 = np.column_stack([
            (points1[:, 0] + points1[:, 1]) / 2,
            (points1[:, 0] + points1[:, 1]) / 2
        ])
        diag2 = np.column_stack([
            (points2[:, 0] + points2[:, 1]) / 2,
            (points2[:, 0] + points2[:, 1]) / 2
        ])

        # Compute cost matrix for Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        # Extend diagrams with diagonal projections
        n1, n2 = len(points1), len(points2)
        size = max(n1, n2)

        extended1 = np.vstack([points1, diag2[:size-n1]] if n1 < size else points1[:size])
        extended2 = np.vstack([points2, diag1[:size-n2]] if n2 < size else points2[:size])

        # Cost matrix
        cost = cdist(extended1, extended2, metric='minkowski', p=p)

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        total_cost = cost[row_ind, col_ind].sum()

        return total_cost ** (1/p) if p > 1 else total_cost

    def _distance_to_diagonal(self, points: np.ndarray, p: int) -> float:
        """Compute total distance of points to diagonal.

        Args:
            points: Diagram points
            p: Order

        Returns:
            Total distance to diagonal
        """
        if points.size == 0:
            return 0.0

        # Distance to diagonal = persistence / sqrt(2)
        persistences = np.abs(points[:, 1] - points[:, 0])
        distances = persistences / np.sqrt(2)

        return float(np.sum(distances ** p) ** (1/p))

    def _interpret_distances(self, distances: Dict[str, float]) -> str:
        """Generate interpretation of distances.

        Args:
            distances: Distances by dimension

        Returns:
            Interpretation string
        """
        interpretations = []

        for dim, dist in sorted(distances.items()):
            if dist < 0.1:
                similarity = "very similar"
            elif dist < 0.3:
                similarity = "similar"
            elif dist < 0.5:
                similarity = "moderately different"
            else:
                similarity = "very different"

            interpretations.append(f"{dim}: {similarity} (distance={dist:.3f})")

        return "; ".join(interpretations)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
