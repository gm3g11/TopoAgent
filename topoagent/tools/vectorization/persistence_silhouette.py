"""Persistence Silhouette Tool for TopoAgent.

Compute persistence silhouettes from persistence diagrams.
Weighted sum of tent functions - alternative to landscapes.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import giotto-tda
try:
    from gtda.diagrams import Silhouette
    HAS_GIOTTO = True
except ImportError:
    HAS_GIOTTO = False


class PersistenceSilhouetteInput(BaseModel):
    """Input schema for PersistenceSilhouetteTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    power: float = Field(
        1.0, description="Weight power (persistence^power)"
    )
    n_bins: int = Field(
        100, description="Resolution of silhouette"
    )


class PersistenceSilhouetteTool(BaseTool):
    """Compute persistence silhouettes from persistence diagrams.

    Silhouette φ(t) = Σᵢ wᵢ · Λᵢ(t) where:
    - Λᵢ(t) is the tent function for feature i
    - wᵢ = persistence_i^power is the weight

    Advantages over landscapes:
    - Single function (not multiple layers)
    - Naturally weights by persistence
    - More robust with small samples

    References:
    - Chazal et al. (2014): Stochastic convergence of silhouettes
    - Used in fMRI and pathology studies
    """

    name: str = "persistence_silhouette"
    description: str = (
        "Compute persistence silhouettes from persistence diagrams. "
        "Weighted sum of tent functions with persistence-based weights. "
        "Input: persistence data from compute_ph tool. "
        "Output: single weighted curve per dimension. "
        "Alternative to landscapes with better small-sample performance."
    )
    args_schema: Type[BaseModel] = PersistenceSilhouetteInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        power: float = 1.0,
        n_bins: int = 100,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute persistence silhouettes.

        Args:
            persistence_data: Persistence pairs by dimension
            power: Weight power
            n_bins: Resolution

        Returns:
            Dictionary with silhouettes
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

            # Compute silhouettes for each dimension
            silhouettes = {}
            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]

                if HAS_GIOTTO:
                    silhouette = self._compute_giotto(pairs, power, n_bins, filtration_values)
                else:
                    silhouette = self._compute_numpy(pairs, power, filtration_values)

                silhouettes[dim_key] = silhouette.tolist()

            # Create combined vector
            combined_vector = []
            for dim_key in sorted(silhouettes.keys()):
                combined_vector.extend(silhouettes[dim_key])

            # Compute statistics
            statistics = {}
            for dim_key, silhouette in silhouettes.items():
                sil_arr = np.array(silhouette)
                statistics[dim_key] = {
                    "max": float(np.max(sil_arr)),
                    "mean": float(np.mean(sil_arr)),
                    "integral": float(np.trapz(sil_arr, filtration_values)),
                    "peak_location": float(filtration_values[np.argmax(sil_arr)])
                }

            return {
                "success": True,
                "tool_name": self.name,
                "silhouettes": silhouettes,
                "filtration_values": filtration_values.tolist(),
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "power": power,
                "n_bins": n_bins,
                "statistics": statistics,
                "interpretation": self._interpret(statistics)
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
        power: float,
        n_bins: int,
        filtration_values: np.ndarray
    ) -> np.ndarray:
        """Compute silhouette using giotto-tda.

        Args:
            pairs: Persistence pairs
            power: Weight power
            n_bins: Number of bins
            filtration_values: Filtration values

        Returns:
            Silhouette array
        """
        valid_pairs = []
        for p in pairs:
            if isinstance(p, dict) and "birth" in p and "death" in p:
                birth = p["birth"]
                death = p["death"] if np.isfinite(p["death"]) else filtration_values[-1]
                valid_pairs.append([birth, death, 0])

        if not valid_pairs:
            return np.zeros(n_bins)

        diagram = np.array(valid_pairs).reshape(1, -1, 3)

        sil = Silhouette(power=power, n_bins=n_bins)
        sil.fit([diagram[0]])
        silhouette = sil.transform([diagram[0]])[0]

        return silhouette.flatten()

    def _compute_numpy(
        self,
        pairs: List[Dict],
        power: float,
        filtration_values: np.ndarray
    ) -> np.ndarray:
        """Compute silhouette using pure numpy (vectorized).

        Args:
            pairs: Persistence pairs
            power: Weight power
            filtration_values: Filtration values

        Returns:
            Silhouette array
        """
        n_bins = len(filtration_values)
        silhouette = np.zeros(n_bins)

        # Extract valid pairs
        valid_pairs = []
        for p in pairs:
            if isinstance(p, dict) and "birth" in p and "death" in p:
                birth = p["birth"]
                death = p["death"]
                if not np.isfinite(death):
                    death = filtration_values[-1]

                persistence = death - birth
                if persistence > 0:
                    valid_pairs.append((birth, death, persistence))

        if not valid_pairs:
            return silhouette

        # Convert to numpy arrays
        pairs_arr = np.array(valid_pairs)
        births = pairs_arr[:, 0]  # (n_points,)
        deaths = pairs_arr[:, 1]  # (n_points,)
        persistences = pairs_arr[:, 2]  # (n_points,)

        # Compute weights
        weights = persistences ** power
        total_weight = np.sum(weights)
        if total_weight == 0:
            return silhouette
        normalized_weights = weights / total_weight

        # Compute midpoints
        midpoints = (births + deaths) / 2  # (n_points,)

        # Vectorized computation
        # t: (n_bins,), births/deaths/midpoints: (n_points,)
        t = filtration_values[:, np.newaxis]  # (n_bins, 1)

        # Check if t is in [birth, death] for each point
        in_range = (t >= births) & (t <= deaths)  # (n_bins, n_points)

        # Compute tent values
        rising = t - births  # (n_bins, n_points)
        falling = deaths - t  # (n_bins, n_points)
        tent_values = np.where(t <= midpoints, rising, falling)  # (n_bins, n_points)
        tent_values = np.where(in_range, tent_values, 0)  # Zero outside range

        # Weighted sum across all points
        silhouette = np.sum(tent_values * normalized_weights, axis=1)  # (n_bins,)

        return silhouette

    def _interpret(self, statistics: Dict[str, Dict]) -> str:
        """Generate interpretation of silhouettes.

        Args:
            statistics: Computed statistics

        Returns:
            Human-readable interpretation
        """
        parts = []

        for dim_key in sorted(statistics.keys()):
            stats = statistics[dim_key]
            parts.append(
                f"{dim_key}: peak at t={stats['peak_location']:.3f}, "
                f"integral={stats['integral']:.3f}"
            )

        # Compare dimensions
        if "H0" in statistics and "H1" in statistics:
            h0_int = statistics["H0"]["integral"]
            h1_int = statistics["H1"]["integral"]
            ratio = h0_int / max(h1_int, 1e-10)
            if ratio > 2:
                parts.append("H0 dominates (component-rich)")
            elif ratio < 0.5:
                parts.append("H1 dominates (loop-rich)")

        return ". ".join(parts) if parts else "Silhouettes computed successfully"

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
