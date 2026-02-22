"""Euler Characteristic Curve Tool for TopoAgent.

Compute Euler characteristic at multiple thresholds.
Output: 100D (default resolution).
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import skimage for Euler number computation
try:
    from skimage.measure import euler_number
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class EulerCharacteristicCurveInput(BaseModel):
    """Input schema for EulerCharacteristicCurveTool."""
    image_array: Union[List[List[float]], Dict] = Field(
        ..., description="Grayscale image as 2D array [0,1] or dict with 'array' key"
    )
    resolution: int = Field(
        50, description="Number of threshold values (default reduced to 50 for efficiency)"
    )
    filtration: str = Field(
        "sublevel", description="Filtration type: 'sublevel' or 'superlevel'"
    )
    max_image_size: int = Field(
        256, description="Maximum image dimension; larger images are downsampled for efficiency"
    )


class EulerCharacteristicCurveTool(BaseTool):
    """Compute Euler characteristic curve from a grayscale image.

    The Euler characteristic curve χ(t) records the Euler characteristic
    of the sublevel (or superlevel) set at each threshold t:

    For sublevel: χ(t) = euler_number({x : f(x) < t})
    For superlevel: χ(t) = euler_number({x : f(x) > t})

    where Euler characteristic χ = components - holes + cavities
    (or χ = β₀ - β₁ + β₂ in terms of Betti numbers).

    Properties:
    - Topological invariant at each threshold
    - Captures how topology evolves with intensity
    - Fast to compute (no full persistence computation needed)
    - Consistent filtration direction with persistence homology

    For 2D images, this is computed using 8-connectivity.

    References:
    - Euler Characteristic Curves in TDA literature
    - Richardson & Werman, "Efficient classification using Euler curves"
    """

    name: str = "euler_characteristic_curve"
    description: str = (
        "Compute Euler characteristic curve from grayscale image. "
        "Records χ = components - holes at each intensity threshold. "
        "Captures topological evolution across filtration. "
        "Input: grayscale image [0,1]. "
        "Output: curve of length 'resolution' (default 100)."
    )
    args_schema: Type[BaseModel] = EulerCharacteristicCurveInput

    def _run(
        self,
        image_array: Union[List[List[float]], Dict],
        resolution: int = 50,
        filtration: str = "sublevel",
        max_image_size: int = 256,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Euler characteristic curve.

        Args:
            image_array: Grayscale image [0,1]
            resolution: Number of thresholds (default 50)
            filtration: 'sublevel' or 'superlevel'
            max_image_size: Maximum image dimension for efficiency

        Returns:
            Dictionary with EC curve
        """
        try:
            # Handle input format
            if isinstance(image_array, dict):
                if 'array' in image_array:
                    image = np.array(image_array['array'], dtype=np.float32)
                else:
                    return {
                        "success": False,
                        "tool_name": self.name,
                        "error": "Dict input must have 'array' key"
                    }
            else:
                image = np.array(image_array, dtype=np.float32)

            if image.ndim != 2:
                return {
                    "success": False,
                    "tool_name": self.name,
                    "error": f"Expected 2D image, got shape {image.shape}"
                }

            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0

            # Downsample large images for efficiency
            original_shape = image.shape
            downsampled = False
            if max(image.shape) > max_image_size:
                scale = max_image_size / max(image.shape)
                new_h = int(image.shape[0] * scale)
                new_w = int(image.shape[1] * scale)
                # Use simple block reduction for speed
                block_h = image.shape[0] // new_h
                block_w = image.shape[1] // new_w
                if block_h > 0 and block_w > 0:
                    # Crop to fit blocks evenly
                    crop_h = new_h * block_h
                    crop_w = new_w * block_w
                    image = image[:crop_h, :crop_w].reshape(new_h, block_h, new_w, block_w).mean(axis=(1, 3))
                    downsampled = True

            # Compute EC curve
            thresholds = np.linspace(0, 1, resolution)
            ec_curve = self._compute_ec_curve(image, thresholds, filtration)

            # Compute summary statistics
            summaries = {
                'min_ec': float(np.min(ec_curve)),
                'max_ec': float(np.max(ec_curve)),
                'mean_ec': float(np.mean(ec_curve)),
                'std_ec': float(np.std(ec_curve)),
                'range_ec': float(np.max(ec_curve) - np.min(ec_curve)),
                'argmin_ec': float(thresholds[np.argmin(ec_curve)]),
                'argmax_ec': float(thresholds[np.argmax(ec_curve)]),
            }

            return {
                "success": True,
                "tool_name": self.name,
                "ec_curve": ec_curve.tolist(),
                "thresholds": thresholds.tolist(),
                "combined_vector": ec_curve.tolist(),
                "vector_length": len(ec_curve),
                "resolution": resolution,
                "filtration": filtration,
                "connectivity": 2,  # 8-connectivity for 2D images
                "connectivity_note": "Using 8-connectivity (connectivity=2 in skimage)",
                "original_shape": list(original_shape),
                "processed_shape": list(image.shape),
                "downsampled": downsampled,
                "summaries": summaries,
                "interpretation": self._interpret(ec_curve, thresholds, summaries)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_ec_curve(
        self,
        image: np.ndarray,
        thresholds: np.ndarray,
        filtration: str
    ) -> np.ndarray:
        """Compute EC at each threshold.

        Args:
            image: 2D grayscale image
            thresholds: Array of threshold values
            filtration: 'sublevel' or 'superlevel'

        Returns:
            Array of EC values
        """
        ec_values = np.zeros(len(thresholds))

        for i, t in enumerate(thresholds):
            if filtration == 'sublevel':
                binary = image < t
            else:  # superlevel
                binary = image > t

            if HAS_SKIMAGE:
                ec = euler_number(binary.astype(np.uint8), connectivity=2)
            else:
                ec = self._euler_number_numpy(binary)

            ec_values[i] = ec

        return ec_values.astype(np.float32)

    def _euler_number_numpy(self, binary: np.ndarray) -> int:
        """Compute Euler number using numpy (fallback).

        Uses the formula: χ = V - E + F for 2D
        where V = vertices (pixels), E = edges, F = faces (4-connectivity)

        For 8-connectivity, we count connected components and holes.
        """
        from scipy import ndimage

        # Count components (8-connectivity for foreground)
        labeled, n_components = ndimage.label(binary, structure=np.ones((3, 3)))

        # Count holes (8-connectivity for background inside foreground)
        # A hole is a background component not connected to border
        inv_binary = ~binary
        labeled_bg, n_bg = ndimage.label(inv_binary, structure=np.ones((3, 3)))

        # Count background components not touching border
        border_labels = set()
        border_labels.update(labeled_bg[0, :])
        border_labels.update(labeled_bg[-1, :])
        border_labels.update(labeled_bg[:, 0])
        border_labels.update(labeled_bg[:, -1])

        n_holes = n_bg - len(border_labels.difference({0}))

        return n_components - n_holes

    def _interpret(
        self,
        ec_curve: np.ndarray,
        thresholds: np.ndarray,
        summaries: Dict
    ) -> str:
        """Generate interpretation of EC curve."""
        parts = []

        # Overall range
        parts.append(
            f"EC ranges from {summaries['min_ec']:.0f} to {summaries['max_ec']:.0f}"
        )

        # Peak/trough locations
        parts.append(
            f"Maximum EC at threshold {summaries['argmax_ec']:.2f}, "
            f"minimum at {summaries['argmin_ec']:.2f}"
        )

        # Interpret trend
        if ec_curve[0] > ec_curve[-1]:
            parts.append("EC generally decreases (features merge/disappear)")
        elif ec_curve[0] < ec_curve[-1]:
            parts.append("EC generally increases (features emerge)")
        else:
            parts.append("EC relatively stable across filtration")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
