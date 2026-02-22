"""Betti Curves Tool for TopoAgent.

Compute Betti curves (Betti numbers as a function of filtration) from persistence diagrams.
Popular vectorization method for dermatology and medical imaging.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import giotto-tda
try:
    from gtda.diagrams import BettiCurve
    HAS_GIOTTO = True
except ImportError:
    HAS_GIOTTO = False


class BettiCurvesInput(BaseModel):
    """Input schema for BettiCurvesTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    n_bins: int = Field(
        100, description="Number of bins for the Betti curve"
    )
    normalize: bool = Field(
        True, description="Normalize curves to [0, 1]"
    )


class BettiCurvesTool(BaseTool):
    """Compute Betti curves from persistence diagrams.

    Betti curve β_k(t) counts the number of k-dimensional features
    alive at filtration value t. Provides a functional summary of
    persistence diagrams.

    Advantages:
    - Fixed-size vector output for ML
    - Captures filtration evolution
    - Interpretable: peaks indicate important scales

    References:
    - Dermatology success with Betti curves (melanoma detection)
    - Standard vectorization in TDA literature
    """

    name: str = "betti_curves"
    description: str = (
        "Compute Betti curves from persistence diagrams. "
        "Betti curve β_k(t) = number of features alive at filtration t. "
        "Input: persistence data from compute_ph tool. "
        "Output: fixed-size vectors suitable for ML classifiers. "
        "Successful in dermatology and medical image classification."
    )
    args_schema: Type[BaseModel] = BettiCurvesInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        n_bins: int = 100,
        normalize: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Betti curves.

        Args:
            persistence_data: Persistence pairs by dimension
            n_bins: Number of bins
            normalize: Whether to normalize

        Returns:
            Dictionary with Betti curves
        """
        try:
            # Find global filtration range
            all_births = []
            all_deaths = []

            for dim_key, pairs in persistence_data.items():
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

            # Determine filtration range
            filt_min = min(all_births)
            filt_max = max(all_deaths) if all_deaths else max(all_births) + 1

            # Add small margin
            margin = (filt_max - filt_min) * 0.01
            filt_min -= margin
            filt_max += margin

            filtration_values = np.linspace(filt_min, filt_max, n_bins)

            # Compute Betti curves for each dimension
            betti_curves = {}
            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]

                if HAS_GIOTTO:
                    curve = self._compute_giotto(pairs, filtration_values)
                else:
                    curve = self._compute_numpy(pairs, filtration_values)

                if normalize and np.max(curve) > 0:
                    curve = curve / np.max(curve)

                betti_curves[dim_key] = curve.tolist()

            # Create combined vector
            combined_vector = []
            for dim_key in sorted(betti_curves.keys()):
                combined_vector.extend(betti_curves[dim_key])

            # Compute summary statistics
            summaries = {}
            for dim_key, curve in betti_curves.items():
                curve_arr = np.array(curve)
                summaries[dim_key] = {
                    "max": float(np.max(curve_arr)),
                    "mean": float(np.mean(curve_arr)),
                    "peak_location": float(filtration_values[np.argmax(curve_arr)]),
                    "area_under_curve": float(np.trapz(curve_arr, filtration_values))
                }

            return {
                "success": True,
                "tool_name": self.name,
                "betti_curves": betti_curves,
                "filtration_values": filtration_values.tolist(),
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "n_bins": n_bins,
                "normalized": normalize,
                "summaries": summaries,
                "interpretation": self._interpret(betti_curves, summaries)
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
        filtration_values: np.ndarray
    ) -> np.ndarray:
        """Compute Betti curve using giotto-tda.

        Args:
            pairs: Persistence pairs
            filtration_values: Filtration values for evaluation

        Returns:
            Betti curve values
        """
        # Convert to giotto-tda format: (N, 3) array with (birth, death, dimension)
        valid_pairs = []
        for p in pairs:
            if isinstance(p, dict) and "birth" in p and "death" in p:
                birth = p["birth"]
                death = p["death"] if np.isfinite(p["death"]) else filtration_values[-1]
                valid_pairs.append([birth, death, 0])  # Dimension 0 for single-dim processing

        if not valid_pairs:
            return np.zeros(len(filtration_values))

        diagram = np.array(valid_pairs).reshape(1, -1, 3)

        bc = BettiCurve(n_bins=len(filtration_values))
        bc.fit([diagram[0]])
        curve = bc.transform([diagram[0]])[0]

        return curve.flatten()

    def _compute_numpy(
        self,
        pairs: List[Dict],
        filtration_values: np.ndarray
    ) -> np.ndarray:
        """Compute Betti curve using pure numpy.

        Args:
            pairs: Persistence pairs
            filtration_values: Filtration values for evaluation

        Returns:
            Betti curve values
        """
        curve = np.zeros(len(filtration_values))

        for p in pairs:
            if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                continue

            birth = p["birth"]
            death = p["death"]

            if not np.isfinite(death):
                death = filtration_values[-1]

            # Count this feature for all filtration values where it's alive
            alive = (filtration_values >= birth) & (filtration_values < death)
            curve += alive.astype(float)

        return curve

    def _interpret(
        self,
        betti_curves: Dict[str, List[float]],
        summaries: Dict[str, Dict]
    ) -> str:
        """Generate interpretation of Betti curves.

        Args:
            betti_curves: Computed Betti curves
            summaries: Summary statistics

        Returns:
            Human-readable interpretation
        """
        parts = []

        for dim_key in sorted(summaries.keys()):
            stats = summaries[dim_key]
            parts.append(
                f"{dim_key}: peak at t={stats['peak_location']:.3f}, "
                f"max count={stats['max']:.1f}"
            )

        # Compare H0 and H1 if both exist
        if "H0" in summaries and "H1" in summaries:
            h0_area = summaries["H0"]["area_under_curve"]
            h1_area = summaries["H1"]["area_under_curve"]
            if h0_area > 2 * h1_area:
                parts.append("Dominated by connected components (H0)")
            elif h1_area > h0_area:
                parts.append("Rich loop structure (H1 dominant)")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
