"""Binarization Tool for TopoAgent.

Applies adaptive thresholding for topology extraction.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class BinarizationInput(BaseModel):
    """Input schema for BinarizationTool."""
    image_array: List[List[float]] = Field(..., description="2D image array (grayscale)")
    method: str = Field("otsu", description="Thresholding method: 'otsu', 'adaptive_mean', 'adaptive_gaussian', 'manual'")
    threshold: Optional[float] = Field(None, description="Manual threshold value (0-1) if method='manual'")
    block_size: int = Field(11, description="Block size for adaptive methods (odd number)")
    c_value: float = Field(2, description="Constant subtracted from mean in adaptive methods")


class BinarizationTool(BaseTool):
    """Apply adaptive thresholding for topology extraction.

    Converts grayscale images to binary using various thresholding methods.
    This is often required before computing topological features.
    """

    name: str = "binarization"
    description: str = (
        "Convert grayscale images to binary using adaptive thresholding. "
        "Methods: 'otsu' (automatic optimal threshold), 'adaptive_mean', 'adaptive_gaussian', 'manual'. "
        "Use this after loading images and before filtration when working with binary topology. "
        "Input: grayscale image array, thresholding method. "
        "Output: binary image array, threshold value used."
    )
    args_schema: Type[BaseModel] = BinarizationInput

    def _run(
        self,
        image_array: List[List[float]],
        method: str = "otsu",
        threshold: Optional[float] = None,
        block_size: int = 11,
        c_value: float = 2,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Apply binarization to the image.

        Args:
            image_array: 2D grayscale image array
            method: Thresholding method
            threshold: Manual threshold (for method='manual')
            block_size: Block size for adaptive methods
            c_value: Constant for adaptive methods

        Returns:
            Dictionary with binary image and threshold info
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float32)

            # Ensure image is in range [0, 1]
            if img.max() > 1.0:
                img = img / 255.0

            if method == "otsu":
                binary, thresh = self._otsu_threshold(img)
            elif method == "adaptive_mean":
                binary, thresh = self._adaptive_threshold(img, "mean", block_size, c_value)
            elif method == "adaptive_gaussian":
                binary, thresh = self._adaptive_threshold(img, "gaussian", block_size, c_value)
            elif method == "manual":
                if threshold is None:
                    threshold = 0.5
                binary = (img > threshold).astype(np.float32)
                thresh = threshold
            else:
                return {
                    "success": False,
                    "error": f"Unknown method: {method}. Use 'otsu', 'adaptive_mean', 'adaptive_gaussian', or 'manual'"
                }

            # Statistics
            foreground_ratio = np.mean(binary)

            return {
                "success": True,
                "binary_image": binary.tolist(),
                "shape": binary.shape,
                "threshold_value": float(thresh),
                "method": method,
                "foreground_ratio": float(foreground_ratio),
                "background_ratio": float(1 - foreground_ratio)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _otsu_threshold(self, img: np.ndarray) -> tuple:
        """Apply Otsu's method for automatic thresholding.

        Args:
            img: Grayscale image array

        Returns:
            Tuple of (binary image, threshold value)
        """
        try:
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(img)
            binary = (img > thresh).astype(np.float32)
            return binary, thresh
        except ImportError:
            # Fallback implementation
            hist, bin_edges = np.histogram(img.flatten(), bins=256, range=(0, 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Otsu's method
            total = img.size
            sum_total = np.sum(bin_centers * hist)
            sum_b = 0
            w_b = 0
            max_variance = 0
            optimal_thresh = 0

            for i in range(256):
                w_b += hist[i]
                if w_b == 0:
                    continue
                w_f = total - w_b
                if w_f == 0:
                    break

                sum_b += bin_centers[i] * hist[i]
                mean_b = sum_b / w_b
                mean_f = (sum_total - sum_b) / w_f

                variance = w_b * w_f * (mean_b - mean_f) ** 2

                if variance > max_variance:
                    max_variance = variance
                    optimal_thresh = bin_centers[i]

            binary = (img > optimal_thresh).astype(np.float32)
            return binary, optimal_thresh

    def _adaptive_threshold(
        self,
        img: np.ndarray,
        method: str,
        block_size: int,
        c_value: float
    ) -> tuple:
        """Apply adaptive thresholding.

        Args:
            img: Grayscale image array
            method: 'mean' or 'gaussian'
            block_size: Size of neighborhood
            c_value: Constant to subtract

        Returns:
            Tuple of (binary image, average threshold)
        """
        from scipy.ndimage import uniform_filter, gaussian_filter

        if method == "mean":
            local_mean = uniform_filter(img, size=block_size)
        else:  # gaussian
            local_mean = gaussian_filter(img, sigma=block_size // 6)

        threshold_map = local_mean - c_value / 255.0
        binary = (img > threshold_map).astype(np.float32)

        return binary, np.mean(threshold_map)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
