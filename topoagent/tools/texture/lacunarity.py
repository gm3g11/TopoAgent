"""Lacunarity Tool for TopoAgent.

Compute lacunarity (texture heterogeneity) from binary images.
Measures "gappiness" or clustering in structures.
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class LacunarityInput(BaseModel):
    """Input schema for LacunarityTool."""
    image_array: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., description="2D or 3D binary image array"
    )
    box_sizes: Optional[List[int]] = Field(
        None, description="Box sizes to analyze (default: powers of 2)"
    )
    threshold: Optional[float] = Field(
        None, description="Binarization threshold"
    )


class LacunarityTool(BaseTool):
    """Compute lacunarity (texture heterogeneity) from binary images.

    Lacunarity Λ(r) = var(mass) / mean(mass)² + 1

    Measures spatial heterogeneity and clustering:
    - High lacunarity: Heterogeneous, clustered distribution
    - Low lacunarity: Homogeneous, uniform distribution
    - Λ = 1: Completely uniform (no variance)

    Uses gliding-box algorithm to compute mass distribution.

    References:
    - Complementary to fractal dimension
    - Used in forest canopy analysis, medical imaging
    """

    name: str = "lacunarity"
    description: str = (
        "Compute lacunarity (texture heterogeneity) from binary images. "
        "Measures spatial clustering and gappiness of structures. "
        "Input: 2D or 3D binary image array. "
        "Output: lacunarity curve across scales, mean lacunarity. "
        "Complements fractal dimension for texture analysis."
    )
    args_schema: Type[BaseModel] = LacunarityInput

    def _run(
        self,
        image_array: Union[List[List[float]], List[List[List[float]]]],
        box_sizes: Optional[List[int]] = None,
        threshold: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute lacunarity.

        Args:
            image_array: 2D or 3D image array
            box_sizes: Sizes for analysis
            threshold: Binarization threshold

        Returns:
            Dictionary with lacunarity data
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float64)

            # Binarize
            if threshold is not None:
                binary = (img > threshold).astype(np.uint8)
            elif img.max() > 1 or img.min() < 0:
                mid = (img.max() + img.min()) / 2
                binary = (img > mid).astype(np.uint8)
            else:
                binary = (img > 0.5).astype(np.uint8)

            # Determine box sizes
            if box_sizes is None:
                min_dim = min(binary.shape)
                max_size = min_dim // 4
                box_sizes = []
                size = 2
                while size <= max_size:
                    box_sizes.append(size)
                    size *= 2

            if len(box_sizes) < 2:
                box_sizes = [2, 4, 8, 16]
                box_sizes = [s for s in box_sizes if s < min(binary.shape)]

            # Compute lacunarity at each scale
            lacunarity_values = []
            for box_size in box_sizes:
                lac = self._compute_lacunarity(binary, box_size)
                lacunarity_values.append(lac)

            # Summary statistics
            mean_lacunarity = np.mean(lacunarity_values)
            max_lacunarity = np.max(lacunarity_values)

            # Lacunarity at different scales
            lacunarity_at_scales = dict(zip(box_sizes, lacunarity_values))

            # Log-log slope (lacunarity exponent)
            if len(box_sizes) >= 3:
                log_sizes = np.log(box_sizes)
                log_lac = np.log(np.array(lacunarity_values))
                coeffs = np.polyfit(log_sizes, log_lac, 1)
                lac_exponent = coeffs[0]
            else:
                lac_exponent = 0.0

            # Create feature vector
            feature_vector = [
                float(mean_lacunarity),
                float(max_lacunarity),
                float(lac_exponent),
                float(lacunarity_values[0]) if lacunarity_values else 0,  # Small scale
                float(lacunarity_values[-1]) if lacunarity_values else 0  # Large scale
            ]
            feature_names = [
                "mean_lacunarity", "max_lacunarity", "lacunarity_exponent",
                "lacunarity_small_scale", "lacunarity_large_scale"
            ]

            return {
                "success": True,
                "tool_name": self.name,
                "lacunarity_curve": lacunarity_values,
                "box_sizes": box_sizes,
                "mean_lacunarity": float(mean_lacunarity),
                "max_lacunarity": float(max_lacunarity),
                "lacunarity_exponent": float(lac_exponent),
                "lacunarity_at_scales": {int(k): float(v) for k, v in lacunarity_at_scales.items()},
                "image_shape": list(binary.shape),
                "foreground_fraction": float(np.sum(binary) / binary.size),
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "interpretation": self._interpret(mean_lacunarity, lac_exponent, lacunarity_values)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_lacunarity(self, binary: np.ndarray, box_size: int) -> float:
        """Compute lacunarity at a single scale using gliding box.

        Args:
            binary: Binary image
            box_size: Size of gliding box

        Returns:
            Lacunarity value
        """
        if binary.ndim == 2:
            return self._compute_lacunarity_2d(binary, box_size)
        else:
            return self._compute_lacunarity_3d(binary, box_size)

    def _compute_lacunarity_2d(self, binary: np.ndarray, box_size: int) -> float:
        """Compute 2D lacunarity.

        Args:
            binary: 2D binary image
            box_size: Size of gliding box

        Returns:
            Lacunarity value
        """
        h, w = binary.shape

        # Collect mass values (sum of pixels in each box position)
        masses = []
        for y in range(h - box_size + 1):
            for x in range(w - box_size + 1):
                box = binary[y:y + box_size, x:x + box_size]
                mass = np.sum(box)
                masses.append(mass)

        if len(masses) == 0 or np.mean(masses) == 0:
            return 1.0

        masses = np.array(masses)
        mean_mass = np.mean(masses)
        var_mass = np.var(masses)

        # Lacunarity formula: Λ = var / mean² + 1
        # Or equivalently: Λ = E[mass²] / E[mass]²
        lacunarity = var_mass / (mean_mass ** 2) + 1

        return float(lacunarity)

    def _compute_lacunarity_3d(self, binary: np.ndarray, box_size: int) -> float:
        """Compute 3D lacunarity.

        Args:
            binary: 3D binary image
            box_size: Size of gliding box

        Returns:
            Lacunarity value
        """
        d, h, w = binary.shape

        # Sample positions (full gliding would be too slow for large images)
        step = max(1, box_size // 2)
        masses = []

        for z in range(0, d - box_size + 1, step):
            for y in range(0, h - box_size + 1, step):
                for x in range(0, w - box_size + 1, step):
                    box = binary[z:z + box_size, y:y + box_size, x:x + box_size]
                    mass = np.sum(box)
                    masses.append(mass)

        if len(masses) == 0 or np.mean(masses) == 0:
            return 1.0

        masses = np.array(masses)
        mean_mass = np.mean(masses)
        var_mass = np.var(masses)

        lacunarity = var_mass / (mean_mass ** 2) + 1

        return float(lacunarity)

    def _interpret(
        self,
        mean_lac: float,
        lac_exp: float,
        lac_curve: List[float]
    ) -> str:
        """Generate interpretation of lacunarity results.

        Args:
            mean_lac: Mean lacunarity
            lac_exp: Lacunarity exponent
            lac_curve: Lacunarity curve

        Returns:
            Human-readable interpretation
        """
        parts = []

        # Heterogeneity interpretation
        if mean_lac < 1.1:
            parts.append(f"Low lacunarity (Λ={mean_lac:.2f}): Very homogeneous distribution")
        elif mean_lac < 1.5:
            parts.append(f"Moderate lacunarity (Λ={mean_lac:.2f}): Somewhat uniform")
        elif mean_lac < 2.0:
            parts.append(f"High lacunarity (Λ={mean_lac:.2f}): Heterogeneous/clustered")
        else:
            parts.append(f"Very high lacunarity (Λ={mean_lac:.2f}): Highly clustered/gappy")

        # Scale dependence
        if len(lac_curve) >= 2:
            if lac_curve[0] > lac_curve[-1] * 1.5:
                parts.append("More heterogeneous at fine scales")
            elif lac_curve[-1] > lac_curve[0] * 1.5:
                parts.append("More heterogeneous at coarse scales")

        # Lacunarity exponent
        if lac_exp < -0.5:
            parts.append("Rapidly decreasing lacunarity with scale")
        elif lac_exp > 0.5:
            parts.append("Increasing lacunarity with scale (unusual)")

        return ". ".join(parts) if parts else f"Mean lacunarity: {mean_lac:.2f}"

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
