"""Carlsson Coordinates Tool for TopoAgent.

Compute 4D coordinate functions from persistence diagrams.
Output: 4D per homology dimension.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class CarlssonCoordinatesInput(BaseModel):
    """Input schema for CarlssonCoordinatesTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )


class CarlssonCoordinatesTool(BaseTool):
    """Compute Carlsson coordinates from persistence diagrams.

    Carlsson coordinates are 4 canonical coordinate functions that
    summarize a persistence diagram into a fixed 4D vector:

    1. c1 = sum(persistence): Total persistence
    2. c2 = sum(persistence^2): Sum of squared persistence
    3. c3 = max(persistence): Maximum persistence
    4. c4 = sum(birth * persistence): Birth-weighted persistence

    These coordinates are:
    - Stable under small perturbations
    - Interpretable
    - Fast to compute (O(n) where n = number of points)
    - Sufficient statistics for certain distributions

    References:
    - Carlsson et al., "Topological approaches to big data" (2014)
    - Used in protein structure analysis, shape classification
    """

    name: str = "carlsson_coordinates"
    description: str = (
        "Compute 4D Carlsson coordinates from persistence diagrams. "
        "c1=total persistence, c2=squared persistence sum, c3=max persistence, "
        "c4=birth-weighted persistence. "
        "Fast, stable, and interpretable summary. "
        "Input: persistence data from compute_ph tool. "
        "Output: 4 values per homology dimension."
    )
    args_schema: Type[BaseModel] = CarlssonCoordinatesInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Carlsson coordinates.

        Args:
            persistence_data: Persistence pairs by dimension

        Returns:
            Dictionary with coordinate values
        """
        try:
            all_coords = {}

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]
                coords = self._compute_coordinates(pairs)
                all_coords[dim_key] = {
                    'c1_total_persistence': coords[0],
                    'c2_squared_persistence_sum': coords[1],
                    'c3_max_persistence': coords[2],
                    'c4_birth_weighted_persistence': coords[3],
                    'vector': coords.tolist()
                }

            # Create combined vector
            combined_vector = []
            for dim_key in sorted(all_coords.keys()):
                combined_vector.extend(all_coords[dim_key]['vector'])

            return {
                "success": True,
                "tool_name": self.name,
                "coordinates": all_coords,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "interpretation": self._interpret(all_coords)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_coordinates(self, pairs: List[Dict]) -> np.ndarray:
        """Compute 4D Carlsson coordinates for a persistence diagram.

        Args:
            pairs: List of persistence pairs

        Returns:
            4D numpy array [c1, c2, c3, c4]
        """
        if not pairs:
            return np.zeros(4)

        # Extract birth, death, persistence
        births = []
        persistences = []

        for p in pairs:
            if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                continue

            birth = p["birth"]
            death = p["death"]

            if not np.isfinite(death):
                continue

            pers = death - birth
            if pers > 0:
                births.append(birth)
                persistences.append(pers)

        if not persistences:
            return np.zeros(4)

        births = np.array(births)
        persistences = np.array(persistences)

        # Compute 4 coordinates
        c1 = np.sum(persistences)                    # Total persistence
        c2 = np.sum(persistences ** 2)               # Sum of squared persistence
        c3 = np.max(persistences)                    # Max persistence
        c4 = np.sum(births * persistences)           # Birth-weighted persistence

        return np.array([c1, c2, c3, c4])

    def _interpret(self, all_coords: Dict) -> str:
        """Generate interpretation of coordinates."""
        parts = []

        for dim_key in sorted(all_coords.keys()):
            coords = all_coords[dim_key]
            parts.append(
                f"{dim_key}: total_pers={coords['c1_total_persistence']:.4f}, "
                f"max_pers={coords['c3_max_persistence']:.4f}"
            )

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
