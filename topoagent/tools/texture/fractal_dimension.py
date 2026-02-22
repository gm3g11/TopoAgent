"""Fractal Dimension Tool for TopoAgent.

Compute box-counting fractal dimension from binary images.
Complementary to TDA for texture characterization.
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class FractalDimensionInput(BaseModel):
    """Input schema for FractalDimensionTool."""
    image_array: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., description="2D or 3D binary image array"
    )
    threshold: Optional[float] = Field(
        None, description="Binarization threshold"
    )
    min_box_size: int = Field(
        2, description="Minimum box size"
    )
    max_box_size: Optional[int] = Field(
        None, description="Maximum box size (default: image_size/4)"
    )


class FractalDimensionTool(BaseTool):
    """Compute box-counting fractal dimension from binary images.

    Fractal dimension D measures self-similarity:
    D = -lim(log(N(s)) / log(s)) as s -> 0

    Where N(s) is the number of boxes of size s needed to cover the set.

    Typical values:
    - D ≈ 1.0: Line-like structure
    - D ≈ 1.5: Moderately complex boundary
    - D ≈ 2.0: Plane-filling structure

    References:
    - Used in tumor boundary analysis
    - Complementary to TDA for texture features
    """

    name: str = "fractal_dimension"
    description: str = (
        "Compute box-counting fractal dimension from binary images. "
        "Measures structural complexity and self-similarity. "
        "Input: 2D or 3D binary image array. "
        "Output: fractal dimension, regression statistics, box counts. "
        "Useful for tumor boundary analysis and texture characterization."
    )
    args_schema: Type[BaseModel] = FractalDimensionInput

    def _run(
        self,
        image_array: Union[List[List[float]], List[List[List[float]]]],
        threshold: Optional[float] = None,
        min_box_size: int = 2,
        max_box_size: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute fractal dimension.

        Args:
            image_array: 2D or 3D image array
            threshold: Binarization threshold
            min_box_size: Minimum box size
            max_box_size: Maximum box size

        Returns:
            Dictionary with fractal dimension and related data
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
            min_dim = min(binary.shape)
            if max_box_size is None:
                max_box_size = min_dim // 4

            # Use powers of 2 for box sizes
            box_sizes = []
            size = min_box_size
            while size <= max_box_size:
                box_sizes.append(size)
                size *= 2

            if len(box_sizes) < 3:
                # Fall back to linear spacing
                box_sizes = list(range(min_box_size, max(max_box_size + 1, min_box_size + 3), max(1, (max_box_size - min_box_size) // 5)))
                if len(box_sizes) < 2:
                    # If image is very small, use all possible sizes
                    box_sizes = [s for s in range(1, min_dim) if s >= 1]
                    if len(box_sizes) < 2:
                        return {
                            "success": False,
                            "tool_name": self.name,
                            "error": "Image too small for fractal dimension analysis (need at least 2x2)"
                        }

            # Count boxes at each scale
            box_counts = []
            for box_size in box_sizes:
                count = self._count_boxes(binary, box_size)
                box_counts.append(count)

            # Compute fractal dimension via linear regression
            log_sizes = np.log(box_sizes)
            log_counts = np.log(np.array(box_counts) + 1e-10)

            # Linear regression: log(N) = -D * log(s) + c
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fractal_dim = -coeffs[0]

            # Compute R-squared
            predicted = np.polyval(coeffs, log_sizes)
            ss_res = np.sum((log_counts - predicted) ** 2)
            ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
            r_squared = 1 - (ss_res / max(ss_tot, 1e-10))

            # Create feature vector
            feature_vector = [
                float(fractal_dim),
                float(r_squared),
                float(box_counts[0]),  # Count at finest scale
                float(box_counts[-1])  # Count at coarsest scale
            ]
            feature_names = [
                "fractal_dimension", "r_squared",
                "count_finest", "count_coarsest"
            ]

            return {
                "success": True,
                "tool_name": self.name,
                "fractal_dimension": float(fractal_dim),
                "box_counting_data": {
                    "box_sizes": box_sizes,
                    "box_counts": box_counts,
                    "log_sizes": log_sizes.tolist(),
                    "log_counts": log_counts.tolist()
                },
                "regression_stats": {
                    "r_squared": float(r_squared),
                    "slope": float(coeffs[0]),
                    "intercept": float(coeffs[1])
                },
                "image_shape": list(binary.shape),
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "interpretation": self._interpret(fractal_dim, r_squared, binary.ndim)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _count_boxes(self, binary: np.ndarray, box_size: int) -> int:
        """Count non-empty boxes of given size.

        Args:
            binary: Binary image
            box_size: Size of boxes

        Returns:
            Number of non-empty boxes
        """
        if binary.ndim == 2:
            return self._count_boxes_2d(binary, box_size)
        else:
            return self._count_boxes_3d(binary, box_size)

    def _count_boxes_2d(self, binary: np.ndarray, box_size: int) -> int:
        """Count non-empty boxes for 2D image.

        Args:
            binary: 2D binary image
            box_size: Size of boxes

        Returns:
            Number of non-empty boxes
        """
        h, w = binary.shape
        count = 0

        for y in range(0, h, box_size):
            for x in range(0, w, box_size):
                # Extract box
                box = binary[y:min(y + box_size, h), x:min(x + box_size, w)]
                if np.any(box):
                    count += 1

        return count

    def _count_boxes_3d(self, binary: np.ndarray, box_size: int) -> int:
        """Count non-empty boxes for 3D image.

        Args:
            binary: 3D binary image
            box_size: Size of boxes

        Returns:
            Number of non-empty boxes
        """
        d, h, w = binary.shape
        count = 0

        for z in range(0, d, box_size):
            for y in range(0, h, box_size):
                for x in range(0, w, box_size):
                    box = binary[
                        z:min(z + box_size, d),
                        y:min(y + box_size, h),
                        x:min(x + box_size, w)
                    ]
                    if np.any(box):
                        count += 1

        return count

    def _interpret(self, fd: float, r_squared: float, ndim: int) -> str:
        """Generate interpretation of fractal dimension.

        Args:
            fd: Fractal dimension
            r_squared: R-squared of regression
            ndim: Number of dimensions

        Returns:
            Human-readable interpretation
        """
        parts = []

        # Dimension interpretation
        if ndim == 2:
            if fd < 1.2:
                parts.append(f"D={fd:.2f}: Nearly linear structure")
            elif fd < 1.5:
                parts.append(f"D={fd:.2f}: Moderately complex boundary")
            elif fd < 1.8:
                parts.append(f"D={fd:.2f}: Complex, space-filling tendency")
            else:
                parts.append(f"D={fd:.2f}: Highly complex, nearly plane-filling")
        else:
            if fd < 2.2:
                parts.append(f"D={fd:.2f}: Surface-like structure")
            elif fd < 2.5:
                parts.append(f"D={fd:.2f}: Moderately complex 3D structure")
            else:
                parts.append(f"D={fd:.2f}: Highly complex, space-filling")

        # Fit quality
        if r_squared > 0.95:
            parts.append("Excellent self-similarity (R²>0.95)")
        elif r_squared > 0.9:
            parts.append("Good fractal behavior (R²>0.9)")
        elif r_squared > 0.8:
            parts.append("Moderate fractal behavior")
        else:
            parts.append(f"Weak fractal behavior (R²={r_squared:.2f})")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
