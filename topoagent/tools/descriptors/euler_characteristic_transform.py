"""Euler Characteristic Transform (ECT) Descriptor Tool for TopoAgent.

Computes the Euler Characteristic Transform over multiple directions.
Output: 640D (32 directions × 20 heights).

Custom implementation.

References:
- Turner, Mukherjee, Boyer, "Persistent homology transform for modeling shapes
  and surfaces" (Information and Inference, 2014)
- Crawford et al., "Predicting clinical outcomes in glioblastoma: an application
  of topological and functional data analysis" (JASA 2020)
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

try:
    from skimage.measure import euler_number
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class EulerCharacteristicTransformInput(BaseModel):
    """Input schema for EulerCharacteristicTransformTool."""
    image_array: Union[List[List[float]], Dict] = Field(
        ..., description="Grayscale image as 2D array [0,1] or dict with 'array' key"
    )
    n_directions: int = Field(
        32, description="Number of directions to sweep"
    )
    n_heights: int = Field(
        20, description="Number of height values per direction"
    )
    max_image_size: int = Field(
        128, description="Maximum image dimension (larger images are downsampled)"
    )


class EulerCharacteristicTransformTool(BaseTool):
    """Compute Euler Characteristic Transform from a grayscale image.

    The ECT computes the Euler characteristic of sublevel sets of the
    image projected along multiple directions. For each direction v and
    height h, compute:

        ECT(v, h) = χ({x : <x, v> ≤ h} ∩ S)

    where S is the foreground of the binarized image.

    Unlike the simple EC curve (which uses intensity thresholds), the ECT
    captures SHAPE information by sweeping directional projections. This
    makes it a Beyond-PH descriptor that is provably injective on shapes.

    Properties:
    - Sufficient statistic for shapes (injective on finite simplicial complexes)
    - Captures directional/geometric structure
    - More informative than single EC curve
    - Complementary to PH-based descriptors

    Output: n_directions × n_heights
    Default: 32 × 20 = 640D
    """

    name: str = "euler_characteristic_transform"
    description: str = (
        "Compute Euler Characteristic Transform from grayscale image. "
        "Sweeps directional projections to capture shape structure. "
        "Beyond-PH descriptor: captures geometric information PH misses. "
        "Input: grayscale image [0,1]. "
        "Output: 640D default (32 directions × 20 heights)."
    )
    args_schema: Type[BaseModel] = EulerCharacteristicTransformInput

    def _run(
        self,
        image_array: Union[List[List[float]], Dict],
        n_directions: int = 32,
        n_heights: int = 20,
        max_image_size: int = 128,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
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

            # Downsample for efficiency
            if max(image.shape) > max_image_size:
                scale = max_image_size / max(image.shape)
                new_h = max(int(image.shape[0] * scale), 2)
                new_w = max(int(image.shape[1] * scale), 2)
                # Block-mean downsampling
                block_h = image.shape[0] // new_h
                block_w = image.shape[1] // new_w
                if block_h > 0 and block_w > 0:
                    crop_h = new_h * block_h
                    crop_w = new_w * block_w
                    image = image[:crop_h, :crop_w].reshape(
                        new_h, block_h, new_w, block_w
                    ).mean(axis=(1, 3))

            # Binarize at Otsu threshold
            threshold = self._otsu_threshold(image)
            binary = image > threshold

            # Get foreground pixel coordinates
            coords = np.array(np.where(binary)).T  # (n_points, 2)

            if len(coords) < 2:
                return {
                    "success": True,
                    "tool_name": self.name,
                    "combined_vector": [0.0] * (n_directions * n_heights),
                    "vector_length": n_directions * n_heights,
                    "n_directions": n_directions,
                    "n_heights": n_heights,
                    "note": "Too few foreground pixels",
                }

            # Compute ECT
            ect = self._compute_ect(binary, coords, n_directions, n_heights)
            combined_vector = ect.flatten().tolist()

            return {
                "success": True,
                "tool_name": self.name,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "n_directions": n_directions,
                "n_heights": n_heights,
                "image_shape": list(image.shape),
                "n_foreground_pixels": len(coords),
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_ect(
        self,
        binary: np.ndarray,
        coords: np.ndarray,
        n_directions: int,
        n_heights: int
    ) -> np.ndarray:
        """Compute the ECT.

        For each direction, project foreground pixels and compute EC of
        sublevel sets at n_heights thresholds.

        Args:
            binary: Binary image
            coords: Foreground pixel coordinates (n_points, 2)
            n_directions: Number of sweep directions
            n_heights: Number of height thresholds per direction

        Returns:
            ECT matrix of shape (n_directions, n_heights)
        """
        ect = np.zeros((n_directions, n_heights))

        # Generate uniformly spaced directions on the circle
        angles = np.linspace(0, np.pi, n_directions, endpoint=False)

        # Precompute all direction vectors at once
        directions = np.column_stack([np.cos(angles), np.sin(angles)])  # (n_dir, 2)

        # Precompute all projections: (n_points, n_directions)
        all_projections = coords @ directions.T

        # Reuse a single binary buffer to avoid 640 allocations per image
        sublevel_binary = np.zeros_like(binary, dtype=np.uint8)
        rows, cols = coords[:, 0], coords[:, 1]

        for i in range(n_directions):
            projections = all_projections[:, i]

            proj_min, proj_max = projections.min(), projections.max()
            if proj_max - proj_min < 1e-10:
                continue

            heights = np.linspace(proj_min, proj_max, n_heights)

            # Sort points by projection value for incremental sublevel building
            sort_idx = np.argsort(projections)
            sorted_proj = projections[sort_idx]
            sorted_rows = rows[sort_idx]
            sorted_cols = cols[sort_idx]

            # Reset binary buffer
            sublevel_binary[:] = 0
            ptr = 0  # pointer into sorted points

            for j, h in enumerate(heights):
                # Incrementally add points up to height h
                while ptr < len(sorted_proj) and sorted_proj[ptr] <= h:
                    sublevel_binary[sorted_rows[ptr], sorted_cols[ptr]] = 1
                    ptr += 1

                if ptr == 0:
                    ect[i, j] = 0
                    continue

                # Compute Euler characteristic
                if HAS_SKIMAGE:
                    ec = euler_number(sublevel_binary, connectivity=2)
                else:
                    ec = self._euler_number_fast(sublevel_binary)
                ect[i, j] = ec

        return ect

    def _euler_number_fast(self, binary: np.ndarray) -> int:
        """Fast Euler number computation using quad-tree bit patterns."""
        from scipy import ndimage
        labeled, n_components = ndimage.label(binary, structure=np.ones((3, 3)))
        inv_binary = ~binary
        labeled_bg, n_bg = ndimage.label(inv_binary, structure=np.ones((3, 3)))
        border_labels = set()
        border_labels.update(labeled_bg[0, :])
        border_labels.update(labeled_bg[-1, :])
        border_labels.update(labeled_bg[:, 0])
        border_labels.update(labeled_bg[:, -1])
        n_holes = n_bg - len(border_labels.difference({0}))
        return n_components - n_holes

    @staticmethod
    def _otsu_threshold(image: np.ndarray) -> float:
        """Compute Otsu's threshold."""
        hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        total = hist.sum()
        if total == 0:
            return 0.5

        weight_bg = np.cumsum(hist)
        weight_fg = total - weight_bg

        mean_bg = np.cumsum(hist * bin_centers)
        mean_bg = np.divide(mean_bg, weight_bg, where=weight_bg > 0)

        mean_fg = np.cumsum(hist[::-1] * bin_centers[::-1])[::-1]
        mean_fg = np.divide(mean_fg, weight_fg, where=weight_fg > 0)

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        idx = np.argmax(variance)

        return bin_centers[idx]

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        return self._run(*args, **kwargs)
