"""Weighted Euler Characteristic Transform (ECT) Tool for TopoAgent.

Compute the ECT for 3D shape analysis. Provides complete shape descriptor.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import scipy for euler number
try:
    from scipy import ndimage
    HAS_SCIPY = True
    HAS_EULER_NUMBER = hasattr(ndimage, 'euler_number')
except ImportError:
    HAS_SCIPY = False
    HAS_EULER_NUMBER = False


class WeightedECTInput(BaseModel):
    """Input schema for WeightedECTTool."""
    image_array: List[List[List[float]]] = Field(
        ..., description="3D image array (shape data)"
    )
    n_directions: int = Field(
        100, description="Number of directions to sample"
    )
    n_thresholds: int = Field(
        50, description="Number of threshold values"
    )
    weights: Optional[List[float]] = Field(
        None, description="Optional vertex weights"
    )


class WeightedECTTool(BaseTool):
    """Compute Weighted Euler Characteristic Transform for 3D shapes.

    ECT: For each direction v, compute χ(X ∩ H_t) for all thresholds t.
    The weighted version incorporates vertex/voxel weights.

    Properties:
    - Complete shape descriptor (injective for generic shapes)
    - Captures both local and global topology
    - Applicable to 3D medical imaging

    References:
    - Turner et al. (2014): Persistent homology transform
    - Used in protein shape analysis and tumor characterization
    """

    name: str = "weighted_ect"
    description: str = (
        "Compute Weighted Euler Characteristic Transform for 3D shape analysis. "
        "Provides complete shape descriptor by computing χ along multiple directions. "
        "Input: 3D image array. "
        "Output: ECT matrix (directions × thresholds), flattened vector. "
        "Cutting-edge method for 3D medical shape characterization."
    )
    args_schema: Type[BaseModel] = WeightedECTInput

    def _run(
        self,
        image_array: List[List[List[float]]],
        n_directions: int = 100,
        n_thresholds: int = 50,
        weights: Optional[List[float]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Weighted ECT.

        Args:
            image_array: 3D image array
            n_directions: Number of directions
            n_thresholds: Number of thresholds
            weights: Optional voxel weights

        Returns:
            Dictionary with ECT data
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float64)

            if img.ndim != 3:
                return {
                    "success": False,
                    "tool_name": self.name,
                    "error": "Input must be a 3D array"
                }

            # Generate uniformly distributed directions on sphere
            directions = self._sample_directions(n_directions)

            # Compute height function range
            max_height = np.sqrt(np.sum(np.array(img.shape) ** 2))
            thresholds = np.linspace(-max_height, max_height, n_thresholds)

            # Compute ECT matrix
            ect_matrix = np.zeros((n_directions, n_thresholds))

            for i, direction in enumerate(directions):
                # Compute height function along this direction
                height_function = self._compute_height_function(img, direction)

                # Apply weights if provided
                if weights is not None:
                    weights_arr = np.array(weights).reshape(img.shape)
                else:
                    weights_arr = None

                # Compute Euler characteristic at each threshold
                for j, t in enumerate(thresholds):
                    chi = self._compute_euler_at_threshold(
                        img, height_function, t, weights_arr
                    )
                    ect_matrix[i, j] = chi

            # Flatten for ML
            flattened_vector = ect_matrix.flatten().tolist()

            # Compute summary statistics
            summaries = {
                "mean_chi": float(np.mean(ect_matrix)),
                "std_chi": float(np.std(ect_matrix)),
                "max_chi": float(np.max(ect_matrix)),
                "min_chi": float(np.min(ect_matrix)),
                "total_variation": float(np.sum(np.abs(np.diff(ect_matrix, axis=1))))
            }

            # Signature: average ECT over directions
            signature = np.mean(ect_matrix, axis=0).tolist()

            return {
                "success": True,
                "tool_name": self.name,
                "ect_matrix": ect_matrix.tolist(),
                "directions": directions.tolist(),
                "thresholds": thresholds.tolist(),
                "flattened_vector": flattened_vector,
                "vector_length": len(flattened_vector),
                "signature": signature,
                "summaries": summaries,
                "n_directions": n_directions,
                "n_thresholds": n_thresholds,
                "interpretation": self._interpret(summaries, ect_matrix)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _sample_directions(self, n: int) -> np.ndarray:
        """Sample uniformly distributed directions on unit sphere.

        Uses Fibonacci spiral method for approximately uniform distribution.

        Args:
            n: Number of directions

        Returns:
            Array of shape (n, 3) with unit vectors
        """
        indices = np.arange(n, dtype=float)
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

        y = 1 - 2 * indices / (n - 1) if n > 1 else np.array([0.0])
        radius = np.sqrt(1 - y ** 2)
        theta = phi * indices

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        return np.column_stack([x, y, z])

    def _compute_height_function(
        self,
        img: np.ndarray,
        direction: np.ndarray
    ) -> np.ndarray:
        """Compute height function (projection onto direction).

        Args:
            img: 3D image
            direction: Unit direction vector

        Returns:
            Height values for each voxel
        """
        d, h, w = img.shape
        z_grid, y_grid, x_grid = np.mgrid[0:d, 0:h, 0:w]

        # Center the grid
        z_centered = z_grid - d / 2
        y_centered = y_grid - h / 2
        x_centered = x_grid - w / 2

        # Project onto direction
        height = (direction[0] * x_centered +
                  direction[1] * y_centered +
                  direction[2] * z_centered)

        return height

    def _compute_euler_at_threshold(
        self,
        img: np.ndarray,
        height_function: np.ndarray,
        threshold: float,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute Euler characteristic at a threshold level.

        Args:
            img: 3D image (used as mask)
            height_function: Height values
            threshold: Threshold level
            weights: Optional voxel weights

        Returns:
            Euler characteristic
        """
        # Create sublevel set: voxels with height <= threshold AND in foreground
        # Binarize image if not already
        if img.max() > 1 or img.min() < 0:
            binary_img = (img > (img.max() + img.min()) / 2)
        else:
            binary_img = (img > 0.5)

        sublevel = binary_img & (height_function <= threshold)

        if not np.any(sublevel):
            return 0.0

        # Compute Euler characteristic
        if HAS_EULER_NUMBER:
            chi = ndimage.euler_number(sublevel.astype(np.uint8), connectivity=1)
        else:
            chi = self._compute_euler_fallback(sublevel)

        # Apply weights if provided
        if weights is not None:
            weighted_chi = chi * np.mean(weights[sublevel])
            return float(weighted_chi)

        return float(chi)

    def _compute_euler_fallback(self, binary: np.ndarray) -> int:
        """Fallback Euler characteristic computation.

        Args:
            binary: Binary 3D image

        Returns:
            Euler characteristic
        """
        n0 = np.sum(binary)
        n1 = (np.sum(binary[:-1, :, :] & binary[1:, :, :]) +
              np.sum(binary[:, :-1, :] & binary[:, 1:, :]) +
              np.sum(binary[:, :, :-1] & binary[:, :, 1:]))

        n2_xy = np.sum(binary[:-1, :-1, :] & binary[:-1, 1:, :] &
                       binary[1:, :-1, :] & binary[1:, 1:, :])
        n2_xz = np.sum(binary[:-1, :, :-1] & binary[:-1, :, 1:] &
                       binary[1:, :, :-1] & binary[1:, :, 1:])
        n2_yz = np.sum(binary[:, :-1, :-1] & binary[:, :-1, 1:] &
                       binary[:, 1:, :-1] & binary[:, 1:, 1:])
        n2 = n2_xy + n2_xz + n2_yz

        n3 = np.sum(binary[:-1, :-1, :-1] & binary[:-1, :-1, 1:] &
                    binary[:-1, 1:, :-1] & binary[:-1, 1:, 1:] &
                    binary[1:, :-1, :-1] & binary[1:, :-1, 1:] &
                    binary[1:, 1:, :-1] & binary[1:, 1:, 1:])

        return int(n0 - n1 + n2 - n3)

    def _interpret(self, summaries: Dict, ect_matrix: np.ndarray) -> str:
        """Generate interpretation of ECT results.

        Args:
            summaries: Summary statistics
            ect_matrix: ECT matrix

        Returns:
            Human-readable interpretation
        """
        parts = []

        mean_chi = summaries["mean_chi"]
        variation = summaries["total_variation"]

        # Shape complexity
        if np.abs(mean_chi) < 0.5:
            parts.append("Average topology near neutral (χ≈0)")
        elif mean_chi > 0:
            parts.append(f"Positive mean χ ({mean_chi:.1f}): component-dominated")
        else:
            parts.append(f"Negative mean χ ({mean_chi:.1f}): hole/cavity-dominated")

        # Anisotropy from direction variance
        direction_variance = np.var(ect_matrix, axis=1)
        mean_dir_var = np.mean(direction_variance)
        if mean_dir_var > 1:
            parts.append("Highly direction-dependent (anisotropic shape)")
        elif mean_dir_var < 0.1:
            parts.append("Nearly isotropic shape")

        # Total variation
        if variation > ect_matrix.size:
            parts.append("Complex topology variation with scale")

        return ". ".join(parts) if parts else "ECT computed successfully"

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
