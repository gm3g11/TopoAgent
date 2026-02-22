"""Local Binary Pattern (LBP) Texture Tool for TopoAgent.

Baseline non-TDA texture descriptor using LBP histogram.
Output: 256D (default) histogram.
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import skimage for LBP
try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class LBPTextureInput(BaseModel):
    """Input schema for LBPTextureTool."""
    image_array: Union[List[List[float]], Dict] = Field(
        ..., description="Grayscale image as 2D array [0,1] or dict with 'array' key"
    )
    P: int = Field(
        8, description="Number of circularly symmetric neighbor points"
    )
    R: float = Field(
        1.0, description="Radius of circle for neighbors"
    )
    method: str = Field(
        "uniform", description="LBP method: 'default', 'ror', 'uniform', 'var'"
    )
    n_scales: Optional[int] = Field(
        None, description="Number of scales for multi-scale LBP. "
        "When set (>1), computes LBP at scales (P=8*(i+1), R=i+1) for i=0..n_scales-1 "
        "and concatenates histograms. E.g. n_scales=8 → 304D (uniform method)."
    )


class LBPTextureTool(BaseTool):
    """Compute Local Binary Pattern texture descriptor.

    This is a BASELINE (non-TDA) texture descriptor for comparison.
    LBP compares each pixel to its neighbors and encodes the pattern
    as a binary number, then builds a histogram.

    Algorithm:
    1. For each pixel, compare to P neighbors at radius R
    2. Encode as P-bit binary number (neighbor > center = 1)
    3. Build histogram of LBP codes across image

    Methods:
    - 'default': Standard LBP (2^P bins)
    - 'uniform': Uniform patterns only (P+2 bins, rotation invariant)
    - 'ror': Rotation invariant (fewer bins)
    - 'var': Rotation invariant with variance

    Properties:
    - Captures local texture patterns
    - Computationally efficient
    - Rotation invariant (with proper method)
    - Standard baseline in texture classification

    References:
    - Ojala et al., "Multiresolution gray-scale and rotation invariant
      texture classification with local binary patterns" (2002)
    """

    name: str = "lbp_texture"
    description: str = (
        "Compute Local Binary Pattern (LBP) texture descriptor. "
        "BASELINE (non-TDA) method for comparison. "
        "Encodes local texture patterns into histogram. "
        "Input: grayscale image [0,1]. "
        "Output: histogram (256D default, or P+2 for 'uniform' method)."
    )
    args_schema: Type[BaseModel] = LBPTextureInput

    def _run(
        self,
        image_array: Union[List[List[float]], Dict],
        P: int = 8,
        R: float = 1.0,
        method: str = "uniform",
        n_scales: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute LBP texture descriptor.

        Args:
            image_array: Grayscale image [0,1]
            P: Number of neighbor points (single-scale mode)
            R: Radius (single-scale mode)
            method: LBP method
            n_scales: If >1, multi-scale mode with auto-generated (P,R) scales

        Returns:
            Dictionary with LBP histogram
        """
        try:
            # Handle input format
            if isinstance(image_array, dict):
                if 'array' in image_array:
                    image = np.array(image_array['array'], dtype=np.float64)
                else:
                    return {
                        "success": False,
                        "tool_name": self.name,
                        "error": "Dict input must have 'array' key"
                    }
            else:
                image = np.array(image_array, dtype=np.float64)

            if image.ndim != 2:
                return {
                    "success": False,
                    "tool_name": self.name,
                    "error": f"Expected 2D image, got shape {image.shape}"
                }

            # Normalize to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0

            # Multi-scale mode
            if n_scales is not None and n_scales > 1:
                return self._run_multi_scale(image, n_scales, method)

            # Single-scale mode (original behavior)
            if HAS_SKIMAGE:
                lbp_image = local_binary_pattern(image, P=P, R=R, method=method)
            else:
                lbp_image = self._compute_lbp_numpy(image, P, R)

            # Compute histogram
            n_bins = self._get_n_bins(P, method)
            histogram, bin_edges = np.histogram(
                lbp_image.ravel(),
                bins=n_bins,
                range=(0, n_bins),
                density=True
            )

            # Compute summary statistics
            summaries = {
                'entropy': float(self._histogram_entropy(histogram)),
                'uniformity': float(np.sum(histogram ** 2)),
                'dominant_bin': int(np.argmax(histogram)),
                'dominant_value': float(np.max(histogram)),
                'n_nonzero_bins': int(np.sum(histogram > 0.001)),
            }

            return {
                "success": True,
                "tool_name": self.name,
                "histogram": histogram.tolist(),
                "combined_vector": histogram.tolist(),
                "vector_length": len(histogram),
                "P": P,
                "R": R,
                "method": method,
                "n_bins": n_bins,
                "summaries": summaries,
                "interpretation": self._interpret(histogram, summaries, method)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _run_multi_scale(
        self,
        image: np.ndarray,
        n_scales: int,
        method: str = "uniform",
    ) -> Dict[str, Any]:
        """Multi-scale LBP: compute at multiple (P, R) scales and concatenate.

        Scales: (P=8*(i+1), R=i+1) for i=0..n_scales-1
        Each scale with 'uniform' method gives P+2 bins.
        E.g. n_scales=8 → (10+18+26+34+42+50+58+66) = 304D

        Args:
            image: 2D grayscale image [0,1]
            n_scales: Number of scales
            method: LBP method

        Returns:
            Dictionary with concatenated multi-scale histogram
        """
        scales = [(8 * (i + 1), float(i + 1)) for i in range(n_scales)]
        multi_scale_vec = []
        scale_details = []
        feature_names = []

        for P_s, R_s in scales:
            if HAS_SKIMAGE:
                lbp_image = local_binary_pattern(image, P=int(P_s), R=R_s, method=method)
            else:
                lbp_image = self._compute_lbp_numpy(image, int(P_s), R_s)

            n_bins = self._get_n_bins(int(P_s), method)
            histogram, _ = np.histogram(
                lbp_image.ravel(),
                bins=n_bins,
                range=(0, n_bins),
                density=True
            )
            multi_scale_vec.extend(histogram.tolist())
            scale_details.append({"P": int(P_s), "R": R_s, "n_bins": n_bins})
            feature_names.extend([f"LBP_P{int(P_s)}_R{R_s:.0f}_bin{j}" for j in range(n_bins)])

        combined = [float(v) for v in multi_scale_vec]

        # Summary across all scales
        arr = np.array(combined)
        summaries = {
            'entropy': float(self._histogram_entropy(arr)),
            'uniformity': float(np.sum(arr ** 2)),
            'n_nonzero_bins': int(np.sum(arr > 0.001)),
            'n_scales': n_scales,
            'total_bins': len(combined),
        }

        return {
            "success": True,
            "tool_name": self.name,
            "mode": "multi_scale",
            "n_scales": n_scales,
            "scales": scale_details,
            "method": method,
            "combined_vector": combined,
            "vector_length": len(combined),
            "feature_names": feature_names,
            "summaries": summaries,
            "interpretation": (
                f"Multi-scale LBP with {n_scales} scales ({method} method). "
                f"Output dimension: {len(combined)}D. "
                f"Scales: {[(int(P), int(R)) for P, R in scales]}."
            ),
        }

    def _get_n_bins(self, P: int, method: str) -> int:
        """Get number of histogram bins for given method."""
        if method == 'uniform':
            return P + 2  # P uniform patterns + 1 non-uniform
        elif method == 'default':
            return 2 ** P
        elif method == 'ror':
            return P + 2
        elif method == 'nri_uniform':
            return P * (P - 1) + 3  # Non-rotation-invariant uniform patterns
        elif method == 'var':
            return 2 ** P
        else:
            return 256  # Default

    def _compute_lbp_numpy(
        self,
        image: np.ndarray,
        P: int,
        R: float
    ) -> np.ndarray:
        """Compute LBP using numpy (fallback).

        Simple implementation for P=8, R=1.
        """
        rows, cols = image.shape
        lbp = np.zeros_like(image, dtype=np.uint8)

        # 8 neighbors at radius 1
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1)
        ]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = image[i, j]
                code = 0
                for k, (di, dj) in enumerate(offsets):
                    if image[i + di, j + dj] >= center:
                        code |= (1 << k)
                lbp[i, j] = code

        return lbp.astype(np.float64)

    @staticmethod
    def _histogram_entropy(histogram: np.ndarray) -> float:
        """Compute entropy of histogram."""
        h = histogram[histogram > 0]
        if len(h) == 0:
            return 0.0
        return -np.sum(h * np.log(h + 1e-10))

    def _interpret(
        self,
        histogram: np.ndarray,
        summaries: Dict,
        method: str
    ) -> str:
        """Generate interpretation of LBP histogram."""
        parts = []

        parts.append(f"LBP texture with {method} method")

        entropy = summaries['entropy']
        if entropy < 2.0:
            parts.append("Low entropy (uniform texture)")
        elif entropy < 4.0:
            parts.append("Moderate entropy (mixed texture)")
        else:
            parts.append("High entropy (complex texture)")

        parts.append(
            f"Dominant pattern at bin {summaries['dominant_bin']} "
            f"({summaries['dominant_value']:.3f})"
        )

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
