"""Tropical Coordinates Tool for TopoAgent.

Compute tropical rational function coordinates from persistence diagrams.
Output: 10-20D per homology dimension.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class TropicalCoordinatesInput(BaseModel):
    """Input schema for TropicalCoordinatesTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    max_terms: int = Field(
        5, description="Maximum number of top persistence terms to use"
    )


class TropicalCoordinatesTool(BaseTool):
    """Compute tropical coordinates from persistence diagrams.

    Tropical coordinates are based on tropical rational functions,
    which provide sufficient statistics for persistence diagrams
    in tropical geometry. For each of the top-k most persistent features,
    we record 4 values:

    1. birth: Birth time
    2. death: Death time
    3. persistence: death - birth
    4. midlife: (birth + death) / 2

    This gives 4*k coordinates total per homology dimension.

    Properties:
    - Captures the most significant topological features
    - Stable under small perturbations
    - Algebraically meaningful in tropical geometry
    - Good for classification tasks

    References:
    - Kališnik, "Tropical coordinates on space of barcodes" (2019)
    - Tropical geometry approach to TDA
    """

    name: str = "tropical_coordinates"
    description: str = (
        "Compute tropical coordinates from persistence diagrams. "
        "Records (birth, death, persistence, midlife) for top-k most persistent features. "
        "Based on tropical rational functions (sufficient statistics). "
        "Input: persistence data from compute_ph tool. "
        "Output: 4*k values per homology dimension (default k=5 gives 20D)."
    )
    args_schema: Type[BaseModel] = TropicalCoordinatesInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        max_terms: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute tropical coordinates.

        Args:
            persistence_data: Persistence pairs by dimension
            max_terms: Number of top persistence terms to use

        Returns:
            Dictionary with coordinate values
        """
        try:
            all_coords = {}

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]
                coords, terms = self._compute_coordinates(pairs, max_terms)
                all_coords[dim_key] = {
                    'coordinates': coords.tolist(),
                    'top_terms': terms,
                    'n_terms': len(terms)
                }

            # Create combined vector
            combined_vector = []
            for dim_key in sorted(all_coords.keys()):
                combined_vector.extend(all_coords[dim_key]['coordinates'])

            return {
                "success": True,
                "tool_name": self.name,
                "coordinates": all_coords,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "max_terms": max_terms,
                "interpretation": self._interpret(all_coords)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_coordinates(self, pairs: List[Dict], max_terms: int) -> tuple:
        """Compute tropical coordinates for a persistence diagram.

        Args:
            pairs: List of persistence pairs
            max_terms: Number of top terms

        Returns:
            Tuple of (coordinates array, list of term details)
        """
        if not pairs:
            return np.zeros(max_terms * 4), []

        # Extract and compute persistence for all valid pairs
        valid_pairs = []
        for p in pairs:
            if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                continue

            birth = p["birth"]
            death = p["death"]

            if not np.isfinite(death):
                continue

            pers = death - birth
            if pers > 0:
                valid_pairs.append({
                    'birth': birth,
                    'death': death,
                    'persistence': pers,
                    'midlife': (birth + death) / 2
                })

        if not valid_pairs:
            return np.zeros(max_terms * 4), []

        # Sort by persistence (descending)
        valid_pairs.sort(key=lambda x: -x['persistence'])

        # Take top-k terms
        top_terms = valid_pairs[:max_terms]

        # Build coordinate vector
        # Use -1 as padding value to indicate "no feature" (distinguishable from valid points)
        # Note: (0, 0, 0, 0) could be a valid point near origin, so we use -1
        PAD_VALUE = -1.0
        coords = []
        for i in range(max_terms):
            if i < len(top_terms):
                t = top_terms[i]
                coords.extend([t['birth'], t['death'], t['persistence'], t['midlife']])
            else:
                # Pad with -1 to indicate "no feature" (fewer than max_terms)
                coords.extend([PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE])

        return np.array(coords), top_terms

    def _interpret(self, all_coords: Dict) -> str:
        """Generate interpretation of coordinates."""
        parts = []

        for dim_key in sorted(all_coords.keys()):
            coord = all_coords[dim_key]
            n_terms = coord['n_terms']
            if n_terms > 0 and coord['top_terms']:
                max_pers = coord['top_terms'][0]['persistence']
                parts.append(
                    f"{dim_key}: {n_terms} significant features, "
                    f"max persistence={max_pers:.4f}"
                )
            else:
                parts.append(f"{dim_key}: no significant features")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
