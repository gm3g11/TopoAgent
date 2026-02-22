"""Persistence Entropy Tool for TopoAgent.

Compute Shannon entropy of persistence diagrams.
Supports two modes:
- 'scalar': 2D (1 entropy value per homology dimension)
- 'vector': 200D (entropy curve at n_bins thresholds × 2 homology dimensions)

References:
- Rucco et al., "Characterisation of persistence diagrams by entropy" (2017)
- Atienza et al., "Persistent entropy for separating topological features" (2019)
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class PersistenceEntropyInput(BaseModel):
    """Input schema for PersistenceEntropyTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    mode: str = Field(
        "vector", description="'scalar' for 2D output, 'vector' for entropy curve (200D)"
    )
    n_bins: int = Field(
        100, description="Number of filtration thresholds for vector mode"
    )
    normalized: bool = Field(
        True, description="Return normalized entropy (0 to 1)"
    )


class PersistenceEntropyTool(BaseTool):
    """Compute persistence entropy from persistence diagrams.

    Supports two modes:

    1. Scalar mode (mode='scalar'): 2D output
       - One entropy value per homology dimension
       - E = -sum(p_i * log(p_i))
       - where p_i = (death_i - birth_i) / sum(death_j - birth_j)

    2. Vector/curve mode (mode='vector'): 200D output (default)
       - Entropy curve over n_bins filtration thresholds per dimension
       - At each threshold t, compute entropy of bars alive at t
       - A bar (b, d) is alive at t if b <= t <= d
       - Captures how topological complexity changes across filtration

    Properties:
    - High entropy: Many features with similar persistence (complex/disordered)
    - Low entropy: Few dominant features (simple/ordered)
    - Vector mode captures evolution of complexity across scales

    References:
    - Rucco et al., "Characterisation of persistence diagrams by entropy" (2017)
    - Atienza et al., "Persistent entropy for separating topological features" (2019)
    """

    name: str = "persistence_entropy"
    description: str = (
        "Compute persistence entropy from persistence diagrams. "
        "Supports 'scalar' mode (2D) or 'vector' mode (200D entropy curve). "
        "Vector mode: entropy at each filtration threshold, capturing complexity evolution. "
        "Input: persistence data from compute_ph tool. "
        "Output: 200D default (100 thresholds × 2 hom dims) in vector mode."
    )
    args_schema: Type[BaseModel] = PersistenceEntropyInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        mode: str = "vector",
        n_bins: int = 100,
        normalized: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute persistence entropy.

        Args:
            persistence_data: Persistence pairs by dimension
            mode: 'scalar' for 2D output, 'vector' for entropy curve
            n_bins: Number of filtration thresholds (vector mode)
            normalized: Whether to normalize to [0, 1]

        Returns:
            Dictionary with entropy values/curves
        """
        try:
            if mode == "vector":
                return self._run_vector(persistence_data, n_bins, normalized)
            else:
                return self._run_scalar(persistence_data, normalized)

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _run_scalar(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        normalized: bool,
    ) -> Dict[str, Any]:
        """Scalar mode: one entropy value per homology dimension."""
        entropies = {}

        for dim_key in sorted(persistence_data.keys()):
            pairs = persistence_data[dim_key]

            # Extract finite persistence values
            persistences = []
            for p in pairs:
                if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                    continue
                death = p["death"]
                birth = p["birth"]
                if np.isfinite(death):
                    pers = death - birth
                    if pers > 0:
                        persistences.append(pers)

            entropy = self._compute_entropy(np.array(persistences), normalized)
            entropies[dim_key] = entropy

        combined_vector = [entropies[k] for k in sorted(entropies.keys())]

        return {
            "success": True,
            "tool_name": self.name,
            "mode": "scalar",
            "entropies": entropies,
            "combined_vector": combined_vector,
            "vector_length": len(combined_vector),
            "normalized": normalized,
            "interpretation": self._interpret(entropies)
        }

    def _run_vector(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        n_bins: int,
        normalized: bool,
    ) -> Dict[str, Any]:
        """Vector mode: entropy curve over filtration thresholds."""
        all_vectors = []

        # Find global filtration range across all dimensions
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

        if not all_births or not all_deaths:
            n_dims = len(persistence_data)
            return {
                "success": True,
                "tool_name": self.name,
                "mode": "vector",
                "combined_vector": [0.0] * (n_bins * n_dims),
                "vector_length": n_bins * n_dims,
                "n_bins": n_bins,
                "note": "No valid persistence pairs found",
            }

        filt_min = min(all_births)
        filt_max = max(all_deaths)
        margin = (filt_max - filt_min) * 0.01
        thresholds = np.linspace(filt_min - margin, filt_max + margin, n_bins)

        for dim_key in sorted(persistence_data.keys()):
            pairs = persistence_data[dim_key]
            curve = self._compute_entropy_curve(pairs, thresholds, normalized)
            all_vectors.append(curve)

        combined_vector = np.concatenate(all_vectors).tolist()

        return {
            "success": True,
            "tool_name": self.name,
            "mode": "vector",
            "combined_vector": combined_vector,
            "vector_length": len(combined_vector),
            "n_bins": n_bins,
            "normalized": normalized,
        }

    def _compute_entropy_curve(
        self,
        pairs: List[Dict[str, float]],
        thresholds: np.ndarray,
        normalized: bool,
    ) -> np.ndarray:
        """Compute entropy at each filtration threshold.

        At threshold t, compute Shannon entropy of the persistence values
        of bars alive at t (i.e., bars with birth <= t <= death).

        Args:
            pairs: Persistence pairs for one homology dimension
            thresholds: Array of filtration threshold values
            normalized: Whether to normalize entropy

        Returns:
            Array of entropy values, one per threshold
        """
        n_bins = len(thresholds)
        entropy_curve = np.zeros(n_bins)

        # Extract valid (birth, death, persistence) triples
        bars = []
        for p in pairs:
            if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                continue
            b, d = p["birth"], p["death"]
            if np.isfinite(d) and d > b:
                bars.append((b, d, d - b))

        if not bars:
            return entropy_curve

        bars = np.array(bars)  # (n_bars, 3): birth, death, persistence
        births = bars[:, 0]
        deaths = bars[:, 1]
        persistences = bars[:, 2]

        for i, t in enumerate(thresholds):
            # Bars alive at threshold t: birth <= t <= death
            alive = (births <= t) & (deaths >= t)
            alive_pers = persistences[alive]

            if len(alive_pers) == 0:
                continue

            entropy_curve[i] = self._compute_entropy(alive_pers, normalized)

        return entropy_curve

    def _compute_entropy(self, persistences: np.ndarray, normalized: bool) -> float:
        """Compute Shannon entropy of persistence values.

        Args:
            persistences: Array of persistence values
            normalized: Whether to normalize

        Returns:
            Entropy value
        """
        if len(persistences) == 0:
            return 0.0

        # Normalize to probability distribution
        total = np.sum(persistences)
        if total < 1e-10:
            return 0.0

        p = persistences / total

        # Compute Shannon entropy
        entropy = -np.sum(p * np.log(p + 1e-10))

        if normalized:
            # Normalize by maximum possible entropy (uniform distribution)
            max_entropy = np.log(len(persistences))
            if max_entropy > 0:
                entropy = entropy / max_entropy

        return float(entropy)

    def _interpret(self, entropies: Dict[str, float]) -> str:
        """Generate interpretation of entropy values."""
        parts = []

        for dim_key in sorted(entropies.keys()):
            e = entropies[dim_key]
            if e < 0.3:
                desc = "low (few dominant features)"
            elif e < 0.7:
                desc = "moderate (mixed feature importance)"
            else:
                desc = "high (many similar features)"
            parts.append(f"{dim_key} entropy={e:.3f} - {desc}")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
