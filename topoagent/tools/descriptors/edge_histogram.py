"""Edge Histogram Descriptor Tool for TopoAgent.

Baseline non-TDA descriptor using edge orientation histograms.
Output: 80D (8 orientations × 10 spatial cells).

Custom implementation using OpenCV Canny edge detection.

References:
- Won et al., "A Feature-based Description Scheme for MPEG-7 Visual" (2002)
- Standard MPEG-7 Edge Histogram Descriptor
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class EdgeHistogramInput(BaseModel):
    """Input schema for EdgeHistogramTool."""
    image_array: Union[List[List[float]], Dict] = Field(
        ..., description="Grayscale image as 2D array [0,1] or dict with 'array' key"
    )
    n_orientation_bins: int = Field(
        8, description="Number of edge orientation bins"
    )
    n_spatial_cells: int = Field(
        10, description="Number of spatial cells (regions) for local histograms"
    )


class EdgeHistogramTool(BaseTool):
    """Compute edge orientation histogram descriptor.

    BASELINE (non-TDA) descriptor that captures the distribution of
    edge orientations across spatial regions of the image.

    Algorithm:
    1. Detect edges using Canny edge detector
    2. Compute gradient orientation at edge pixels
    3. Divide image into spatial cells
    4. For each cell, build a histogram of edge orientations
    5. Concatenate all cell histograms

    Properties:
    - Captures structural/geometric edge information
    - Spatial layout awareness (unlike global histograms)
    - Complementary to texture (LBP) and topology (TDA) descriptors
    - Standard in computer vision (MPEG-7)

    Output: n_orientation_bins × n_spatial_cells
    Default: 8 × 10 = 80D
    """

    name: str = "edge_histogram"
    description: str = (
        "Compute edge orientation histogram descriptor. "
        "BASELINE (non-TDA) capturing edge structure across spatial regions. "
        "Input: grayscale image [0,1]. "
        "Output: 80D default (8 orientations × 10 spatial cells)."
    )
    args_schema: Type[BaseModel] = EdgeHistogramInput

    def _run(
        self,
        image_array: Union[List[List[float]], Dict],
        n_orientation_bins: int = 8,
        n_spatial_cells: int = 10,
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

            # Normalize to [0, 255] uint8 for edge detection
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)

            # Compute edges and orientations
            edges, orientations = self._compute_edges(image_uint8)

            # Compute spatial edge histograms
            histogram = self._compute_spatial_histogram(
                edges, orientations, n_orientation_bins, n_spatial_cells
            )

            combined_vector = histogram.flatten().tolist()

            return {
                "success": True,
                "tool_name": self.name,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "n_orientation_bins": n_orientation_bins,
                "n_spatial_cells": n_spatial_cells,
                "n_edge_pixels": int(np.sum(edges > 0)),
                "edge_density": float(np.mean(edges > 0)),
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_edges(self, image_uint8: np.ndarray) -> tuple:
        """Detect edges and compute gradient orientations.

        Returns:
            Tuple of (edge_mask, orientation_map)
        """
        if HAS_CV2:
            # Canny edge detection
            edges = cv2.Canny(image_uint8, 50, 150)

            # Compute gradient for orientation
            gx = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
        else:
            # Numpy fallback using Sobel-like filters
            from scipy import ndimage
            gx = ndimage.sobel(image_uint8.astype(np.float64), axis=1)
            gy = ndimage.sobel(image_uint8.astype(np.float64), axis=0)

            # Simple edge detection via gradient magnitude thresholding
            magnitude = np.sqrt(gx**2 + gy**2)
            threshold = np.percentile(magnitude[magnitude > 0], 70) if np.any(magnitude > 0) else 0
            edges = (magnitude > threshold).astype(np.uint8) * 255

        # Compute orientation (0 to pi, undirected edges)
        orientations = np.arctan2(gy, gx) % np.pi  # [0, pi)

        return edges, orientations

    def _compute_spatial_histogram(
        self,
        edges: np.ndarray,
        orientations: np.ndarray,
        n_orientation_bins: int,
        n_spatial_cells: int
    ) -> np.ndarray:
        """Compute edge orientation histograms in spatial cells.

        Divides image into a grid of cells and computes orientation
        histogram in each cell.

        Returns:
            Array of shape (n_spatial_cells, n_orientation_bins)
        """
        h, w = edges.shape

        # Determine grid layout (as close to square as possible)
        n_rows = int(np.ceil(np.sqrt(n_spatial_cells)))
        n_cols = int(np.ceil(n_spatial_cells / n_rows))
        actual_cells = n_rows * n_cols

        cell_h = h // n_rows
        cell_w = w // n_cols

        histogram = np.zeros((n_spatial_cells, n_orientation_bins))

        # Orientation bin edges
        bin_edges = np.linspace(0, np.pi, n_orientation_bins + 1)

        cell_idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if cell_idx >= n_spatial_cells:
                    break

                # Extract cell region
                r_start = i * cell_h
                r_end = (i + 1) * cell_h if i < n_rows - 1 else h
                c_start = j * cell_w
                c_end = (j + 1) * cell_w if j < n_cols - 1 else w

                cell_edges = edges[r_start:r_end, c_start:c_end]
                cell_orient = orientations[r_start:r_end, c_start:c_end]

                # Get orientations at edge pixels
                edge_mask = cell_edges > 0
                if np.any(edge_mask):
                    edge_orientations = cell_orient[edge_mask]
                    # Compute histogram
                    hist, _ = np.histogram(
                        edge_orientations,
                        bins=bin_edges,
                        density=True
                    )
                    histogram[cell_idx] = hist

                cell_idx += 1

        return histogram

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        return self._run(*args, **kwargs)
