"""Betti Ratios Tool for TopoAgent.

Compute Betti number ratios and persistent Betti numbers from persistence data.
Useful for bone microstructure analysis and texture characterization.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class BettiRatiosInput(BaseModel):
    """Input schema for BettiRatiosTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    persistence_threshold: float = Field(
        0.0, description="Minimum persistence to count a feature (filters noise)"
    )
    filtration_value: Optional[float] = Field(
        None, description="Specific filtration value to compute Betti numbers at"
    )


class BettiRatiosTool(BaseTool):
    """Compute Betti number ratios from persistence data.

    Betti ratios (β₀/β₁, β₁/β₂) characterize structural topology:
    - High β₀/β₁: Many components, few loops (fragmented structure)
    - Low β₀/β₁: Connected structure with loops (trabecular bone)

    Used in bone microstructure analysis and medical imaging.

    References:
    - Betti numbers in trabecular bone analysis
    - Structural analysis in radiology
    """

    name: str = "betti_ratios"
    description: str = (
        "Compute Betti number ratios from persistence diagrams. "
        "Ratios include: β₀/β₁ (components/loops), β₁/β₂ (loops/cavities). "
        "Input: persistence data from compute_ph tool. "
        "Output: Betti numbers, ratios, and persistent Betti numbers. "
        "Useful for bone microstructure and tissue topology analysis."
    )
    args_schema: Type[BaseModel] = BettiRatiosInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        persistence_threshold: float = 0.0,
        filtration_value: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Betti ratios.

        Args:
            persistence_data: Persistence pairs by dimension
            persistence_threshold: Minimum persistence to count
            filtration_value: Optional filtration value for Betti numbers

        Returns:
            Dictionary with Betti numbers and ratios
        """
        try:
            # Count features per dimension
            betti_raw = {}
            betti_persistent = {}
            betti_at_filtration = {}

            for dim_key in ["H0", "H1", "H2"]:
                pairs = persistence_data.get(dim_key, [])

                if not pairs or not isinstance(pairs, list):
                    betti_raw[dim_key] = 0
                    betti_persistent[dim_key] = 0
                    betti_at_filtration[dim_key] = 0
                    continue

                # Extract valid pairs
                valid_pairs = [
                    p for p in pairs
                    if isinstance(p, dict) and "birth" in p and "death" in p
                ]

                # Raw Betti numbers (all features)
                betti_raw[dim_key] = len(valid_pairs)

                # Persistent Betti numbers (above threshold)
                persistent_pairs = [
                    p for p in valid_pairs
                    if abs(p["death"] - p["birth"]) >= persistence_threshold
                ]
                betti_persistent[dim_key] = len(persistent_pairs)

                # Betti at specific filtration value
                if filtration_value is not None:
                    alive_at_t = [
                        p for p in valid_pairs
                        if p["birth"] <= filtration_value < p["death"]
                    ]
                    betti_at_filtration[dim_key] = len(alive_at_t)
                else:
                    betti_at_filtration[dim_key] = None

            # Compute ratios
            b0, b1, b2 = betti_raw["H0"], betti_raw["H1"], betti_raw["H2"]
            pb0, pb1, pb2 = betti_persistent["H0"], betti_persistent["H1"], betti_persistent["H2"]

            ratios = {
                "b0_b1": float(b0 / max(b1, 1)),
                "b1_b2": float(b1 / max(b2, 1)) if b2 > 0 else None,
                "b0_b2": float(b0 / max(b2, 1)) if b2 > 0 else None,
                "persistent_b0_b1": float(pb0 / max(pb1, 1)),
                "persistent_b1_b2": float(pb1 / max(pb2, 1)) if pb2 > 0 else None,
            }

            # Euler characteristic from Betti numbers
            euler_raw = b0 - b1 + b2
            euler_persistent = pb0 - pb1 + pb2

            # Structural interpretation
            interpretation = self._interpret_ratios(betti_raw, betti_persistent, ratios)

            # Create feature vector
            feature_vector = [
                float(b0), float(b1), float(b2),
                float(pb0), float(pb1), float(pb2),
                ratios["b0_b1"], ratios["persistent_b0_b1"],
                float(euler_raw), float(euler_persistent)
            ]
            feature_names = [
                "b0", "b1", "b2",
                "persistent_b0", "persistent_b1", "persistent_b2",
                "b0_b1_ratio", "persistent_b0_b1_ratio",
                "euler_raw", "euler_persistent"
            ]

            return {
                "success": True,
                "tool_name": self.name,
                "betti_numbers": betti_raw,
                "persistent_betti_numbers": betti_persistent,
                "betti_at_filtration": betti_at_filtration if filtration_value is not None else None,
                "ratios": ratios,
                "euler_characteristic": {
                    "raw": euler_raw,
                    "persistent": euler_persistent
                },
                "persistence_threshold": persistence_threshold,
                "filtration_value": filtration_value,
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "interpretation": interpretation
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _interpret_ratios(
        self,
        betti_raw: Dict[str, int],
        betti_persistent: Dict[str, int],
        ratios: Dict[str, Optional[float]]
    ) -> str:
        """Generate interpretation of Betti ratios.

        Args:
            betti_raw: Raw Betti numbers
            betti_persistent: Persistent Betti numbers
            ratios: Computed ratios

        Returns:
            Human-readable interpretation
        """
        b0, b1 = betti_raw["H0"], betti_raw["H1"]
        ratio = ratios.get("b0_b1", 0)

        parts = []

        # Structure type based on ratio
        if b1 == 0:
            parts.append("No loops detected - simple connected components")
        elif ratio > 5:
            parts.append("High β₀/β₁ ratio: fragmented structure with few loops")
        elif ratio < 0.5:
            parts.append("Low β₀/β₁ ratio: well-connected structure with many loops")
        else:
            parts.append(f"Moderate β₀/β₁ ratio ({ratio:.2f}): mixed topology")

        # Component analysis
        if b0 == 1:
            parts.append("Single connected component")
        else:
            parts.append(f"{b0} separate components detected")

        # Loop analysis
        if b1 > 0:
            noise_ratio = 1 - (betti_persistent["H1"] / max(b1, 1))
            if noise_ratio > 0.5:
                parts.append(f"{int(noise_ratio * 100)}% of loops are low-persistence (noise)")

        # 3D structure
        if betti_raw["H2"] > 0:
            parts.append(f"{betti_raw['H2']} enclosed cavities detected")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
