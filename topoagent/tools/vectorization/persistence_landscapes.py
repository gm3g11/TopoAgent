"""Persistence Landscapes Tool for TopoAgent.

Compute persistence landscapes from persistence diagrams.
Provides stable vectorization with theoretical guarantees.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import giotto-tda
try:
    from gtda.diagrams import PersistenceLandscape
    HAS_GIOTTO = True
except ImportError:
    HAS_GIOTTO = False


class PersistenceLandscapesInput(BaseModel):
    """Input schema for PersistenceLandscapesTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    n_layers: int = Field(
        5, description="Number of landscape layers"
    )
    n_bins: int = Field(
        100, description="Resolution of each layer"
    )
    combine_dims: bool = Field(
        False, description="If True, combine all homology dims into one diagram "
                           "before computing landscapes (output: n_layers × n_bins). "
                           "If False, compute per-dim and concatenate (output: n_dims × n_layers × n_bins)."
    )


class PersistenceLandscapesTool(BaseTool):
    """Compute persistence landscapes from persistence diagrams.

    Persistence landscape λ_k(t) is the k-th largest value of the
    tent functions centered at each persistence point.

    Advantages:
    - Lives in a Hilbert space (enables statistics)
    - Stable under perturbations
    - Multiple layers capture different scales

    References:
    - Bubenik (2015): Statistical topological data analysis
    - Used in fMRI and brain connectivity analysis
    """

    name: str = "persistence_landscapes"
    description: str = (
        "Compute persistence landscapes from persistence diagrams. "
        "Landscapes embed diagrams into a Hilbert space for statistics. "
        "Input: persistence data from compute_ph tool. "
        "Output: n_layers × n_bins matrix per dimension. "
        "Enables statistical analysis and kernel methods on TDA features."
    )
    args_schema: Type[BaseModel] = PersistenceLandscapesInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        n_layers: int = 5,
        n_bins: int = 100,
        combine_dims: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute persistence landscapes.

        Args:
            persistence_data: Persistence pairs by dimension
            n_layers: Number of landscape layers
            n_bins: Resolution of each layer
            combine_dims: If True, merge all homology dims into one diagram
                          before computing landscapes (output: n_layers × n_bins).
                          If False, compute per-dim and concatenate
                          (output: n_dims × n_layers × n_bins).

        Returns:
            Dictionary with landscapes
        """
        try:
            # Find global filtration range
            all_births = []
            all_deaths = []

            for pairs in persistence_data.values():
                if not pairs or not isinstance(pairs, list):
                    continue
                for p in pairs:
                    if isinstance(p, dict) and "birth" in p and "death" in p:
                        all_births.append(p["birth"])
                        if np.isfinite(p["death"]):
                            all_deaths.append(p["death"])

            if not all_births:
                return {
                    "success": False,
                    "tool_name": self.name,
                    "error": "No valid persistence pairs found"
                }

            filt_min = min(all_births)
            filt_max = max(all_deaths) if all_deaths else max(all_births) + 1
            margin = (filt_max - filt_min) * 0.01
            filt_min -= margin
            filt_max += margin

            filtration_values = np.linspace(filt_min, filt_max, n_bins)

            if combine_dims:
                # Combine all homology dimensions into a single diagram
                all_pairs = []
                for dim_key in sorted(persistence_data.keys()):
                    pairs = persistence_data[dim_key]
                    if pairs and isinstance(pairs, list):
                        all_pairs.extend(pairs)

                if HAS_GIOTTO:
                    landscape = self._compute_giotto(all_pairs, n_layers, n_bins, filtration_values)
                else:
                    landscape = self._compute_numpy(all_pairs, n_layers, filtration_values)

                landscapes = {"combined": landscape.tolist()}

                combined_vector = []
                for layer in landscapes["combined"]:
                    combined_vector.extend(layer)

                norms = {
                    "combined": {
                        f"layer_{i}_L2": float(np.sqrt(np.sum(np.array(landscapes["combined"][i]) ** 2)))
                        for i in range(min(n_layers, len(landscapes["combined"])))
                    }
                }
            else:
                # Compute landscapes for each dimension separately
                landscapes = {}
                for dim_key in sorted(persistence_data.keys()):
                    pairs = persistence_data[dim_key]

                    if HAS_GIOTTO:
                        landscape = self._compute_giotto(pairs, n_layers, n_bins, filtration_values)
                    else:
                        landscape = self._compute_numpy(pairs, n_layers, filtration_values)

                    landscapes[dim_key] = landscape.tolist()

                # Create combined vector
                combined_vector = []
                for dim_key in sorted(landscapes.keys()):
                    for layer in landscapes[dim_key]:
                        combined_vector.extend(layer)

                # Compute norms for each dimension/layer
                norms = {}
                for dim_key, landscape in landscapes.items():
                    landscape_arr = np.array(landscape)
                    norms[dim_key] = {
                        f"layer_{i}_L2": float(np.sqrt(np.sum(landscape_arr[i] ** 2)))
                        for i in range(min(n_layers, landscape_arr.shape[0]))
                    }

            return {
                "success": True,
                "tool_name": self.name,
                "landscapes": landscapes,
                "filtration_values": filtration_values.tolist(),
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "n_layers": n_layers,
                "n_bins": n_bins,
                "combine_dims": combine_dims,
                "norms": norms,
                "interpretation": self._interpret(landscapes, norms)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_giotto(
        self,
        pairs: List[Dict],
        n_layers: int,
        n_bins: int,
        filtration_values: np.ndarray
    ) -> np.ndarray:
        """Compute landscape using giotto-tda.

        Args:
            pairs: Persistence pairs
            n_layers: Number of layers
            n_bins: Number of bins
            filtration_values: Filtration values

        Returns:
            Landscape array of shape (n_layers, n_bins)
        """
        valid_pairs = []
        for p in pairs:
            if isinstance(p, dict) and "birth" in p and "death" in p:
                birth = p["birth"]
                death = p["death"] if np.isfinite(p["death"]) else filtration_values[-1]
                valid_pairs.append([birth, death, 0])

        if not valid_pairs:
            return np.zeros((n_layers, n_bins))

        diagram = np.array(valid_pairs).reshape(1, -1, 3)

        pl = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
        pl.fit([diagram[0]])
        landscape = pl.transform([diagram[0]])[0]

        return landscape.reshape(n_layers, n_bins)

    def _compute_numpy(
        self,
        pairs: List[Dict],
        n_layers: int,
        filtration_values: np.ndarray
    ) -> np.ndarray:
        """Compute landscape using pure numpy (vectorized).

        Args:
            pairs: Persistence pairs
            n_layers: Number of layers
            filtration_values: Filtration values

        Returns:
            Landscape array
        """
        n_bins = len(filtration_values)
        landscapes = np.zeros((n_layers, n_bins))

        # Extract valid pairs
        valid_pairs = []
        for p in pairs:
            if isinstance(p, dict) and "birth" in p and "death" in p:
                birth = p["birth"]
                death = p["death"]
                if not np.isfinite(death):
                    death = filtration_values[-1]
                valid_pairs.append((birth, death))

        if not valid_pairs:
            return landscapes

        # Convert to numpy arrays for vectorized computation
        pairs_arr = np.array(valid_pairs)
        births = pairs_arr[:, 0]  # (n_points,)
        deaths = pairs_arr[:, 1]  # (n_points,)
        midpoints = (births + deaths) / 2  # (n_points,)

        # Vectorized tent function computation
        # t: (n_bins,), births/deaths/midpoints: (n_points,)
        t = filtration_values[:, np.newaxis]  # (n_bins, 1)

        # Check if t is in [birth, death] for each point
        in_range = (t >= births) & (t <= deaths)  # (n_bins, n_points)

        # Compute tent values: min(t - birth, death - t)
        rising = t - births  # (n_bins, n_points)
        falling = deaths - t  # (n_bins, n_points)
        tent_values = np.where(t <= midpoints, rising, falling)  # (n_bins, n_points)
        tent_values = np.where(in_range, tent_values, 0)  # Zero outside range

        # For each filtration value, sort and take top n_layers
        for t_idx in range(n_bins):
            values = tent_values[t_idx]
            values = values[values > 0]  # Only positive values
            if len(values) > 0:
                sorted_values = np.sort(values)[::-1]  # Descending
                n_fill = min(n_layers, len(sorted_values))
                landscapes[:n_fill, t_idx] = sorted_values[:n_fill]

        return landscapes

    def _interpret(
        self,
        landscapes: Dict[str, List[List[float]]],
        norms: Dict[str, Dict]
    ) -> str:
        """Generate interpretation of landscapes.

        Args:
            landscapes: Computed landscapes
            norms: Layer norms

        Returns:
            Human-readable interpretation
        """
        parts = []

        for dim_key in sorted(landscapes.keys()):
            dim_norms = norms.get(dim_key, {})
            if dim_norms:
                l1_norm = dim_norms.get("layer_0_L2", 0)
                parts.append(f"{dim_key} layer 0 L2-norm: {l1_norm:.3f}")

        # Check if first layer dominates
        for dim_key in sorted(landscapes.keys()):
            dim_norms = norms.get(dim_key, {})
            l0 = dim_norms.get("layer_0_L2", 0)
            l1 = dim_norms.get("layer_1_L2", 0)
            if l0 > 0 and l1 > 0:
                ratio = l0 / l1
                if ratio > 3:
                    parts.append(f"{dim_key}: dominated by single prominent feature")

        return ". ".join(parts) if parts else "Landscapes computed successfully"

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
