"""Noise Filter Tool for TopoAgent.

Applies Gaussian/median filtering for topological denoising.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class NoiseFilterInput(BaseModel):
    """Input schema for NoiseFilterTool."""
    image_array: List[List[float]] = Field(..., description="2D image array")
    method: str = Field("gaussian", description="Filtering method: 'gaussian', 'median', 'bilateral'")
    sigma: float = Field(1.0, description="Standard deviation for Gaussian filter")
    kernel_size: int = Field(3, description="Kernel size for median filter (odd number)")


class NoiseFilterTool(BaseTool):
    """Apply noise filtering for cleaner topological analysis.

    Noise can create spurious topological features, so filtering
    is often important before computing persistent homology.
    """

    name: str = "noise_filter"
    description: str = (
        "Apply noise filtering to medical images before topological analysis. "
        "Methods: 'gaussian' (smooth edges), 'median' (preserve edges, remove salt-pepper), 'bilateral' (edge-preserving smoothing). "
        "Use this after loading images to reduce noise-induced topological artifacts. "
        "Input: image array, filter method and parameters. "
        "Output: filtered image array, filter statistics."
    )
    args_schema: Type[BaseModel] = NoiseFilterInput

    def _run(
        self,
        image_array: List[List[float]],
        method: str = "gaussian",
        sigma: float = 1.0,
        kernel_size: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Apply noise filtering to the image.

        Args:
            image_array: 2D image array
            method: Filter method
            sigma: Gaussian sigma
            kernel_size: Median kernel size

        Returns:
            Dictionary with filtered image and statistics
        """
        try:
            from scipy.ndimage import gaussian_filter, median_filter

            # Convert to numpy array
            img = np.array(image_array, dtype=np.float32)

            # Store original stats for comparison
            original_std = np.std(img)

            if method == "gaussian":
                filtered = gaussian_filter(img, sigma=sigma)
                params_used = {"sigma": sigma}
            elif method == "median":
                filtered = median_filter(img, size=kernel_size)
                params_used = {"kernel_size": kernel_size}
            elif method == "bilateral":
                filtered = self._bilateral_filter(img, sigma_spatial=sigma, sigma_range=0.1)
                params_used = {"sigma_spatial": sigma, "sigma_range": 0.1}
            else:
                return {
                    "success": False,
                    "error": f"Unknown method: {method}. Use 'gaussian', 'median', or 'bilateral'"
                }

            # Calculate noise reduction estimate
            filtered_std = np.std(filtered)
            noise_reduction = (original_std - filtered_std) / original_std if original_std > 0 else 0

            return {
                "success": True,
                "filtered_image": filtered.tolist(),
                "shape": filtered.shape,
                "method": method,
                "parameters": params_used,
                "original_std": float(original_std),
                "filtered_std": float(filtered_std),
                "noise_reduction_estimate": float(noise_reduction),
                "min_value": float(filtered.min()),
                "max_value": float(filtered.max())
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _bilateral_filter(
        self,
        img: np.ndarray,
        sigma_spatial: float = 1.0,
        sigma_range: float = 0.1
    ) -> np.ndarray:
        """Apply bilateral filtering (edge-preserving).

        Args:
            img: Input image
            sigma_spatial: Spatial sigma
            sigma_range: Range/intensity sigma

        Returns:
            Filtered image
        """
        try:
            from skimage.restoration import denoise_bilateral
            return denoise_bilateral(img, sigma_spatial=sigma_spatial, sigma_color=sigma_range)
        except ImportError:
            # Fallback to Gaussian if skimage not available
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma=sigma_spatial)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
