"""Minkowski Functionals Tool for TopoAgent.

Compute Minkowski functionals (intrinsic volumes) from binary images.
Standard morphological descriptors for radiomics and medical imaging.
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import scipy for morphological operations
try:
    from scipy import ndimage
    from scipy.ndimage import binary_erosion, generate_binary_structure
    HAS_SCIPY = True
    # Check if euler_number function exists (added in scipy 1.4.0)
    HAS_EULER_NUMBER = hasattr(ndimage, 'euler_number')
except ImportError:
    HAS_SCIPY = False
    HAS_EULER_NUMBER = False


class MinkowskiFunctionalsInput(BaseModel):
    """Input schema for MinkowskiFunctionalsTool."""
    image_array: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., description="2D or 3D binary image array"
    )
    threshold: Optional[float] = Field(
        None, description="Binarization threshold (default: auto)"
    )
    pixel_size: float = Field(
        1.0, description="Physical size of pixel/voxel for scaled measurements"
    )
    n_thresholds: Optional[int] = Field(
        None, description="Number of thresholds for multi-threshold mode. "
        "When set (>1), computes MF at evenly-spaced thresholds and "
        "concatenates [volume, surface, euler] per threshold → 3*n_thresholds dims."
    )
    adaptive: bool = Field(
        False, description="If True with n_thresholds, use percentile-based thresholds "
        "instead of linearly-spaced thresholds."
    )


class MinkowskiFunctionalsTool(BaseTool):
    """Compute Minkowski functionals from binary images.

    Minkowski functionals are fundamental morphological descriptors:
    - 2D: Volume (area), Surface (perimeter), Euler characteristic
    - 3D: Volume, Surface area, Mean breadth, Euler characteristic

    Also known as intrinsic volumes or quermassintegrals.

    References:
    - Hadwiger's theorem: Minkowski functionals form a complete set
    - Used extensively in radiomics for tumor characterization
    """

    name: str = "minkowski_functionals"
    description: str = (
        "Compute Minkowski functionals from binary images. "
        "For 2D: area, perimeter, Euler characteristic. "
        "For 3D: volume, surface area, mean breadth, Euler characteristic. "
        "Input: 2D or 3D binary image array. "
        "Output: Minkowski functionals with normalized versions. "
        "Essential for radiomics and morphological analysis."
    )
    args_schema: Type[BaseModel] = MinkowskiFunctionalsInput

    def _run(
        self,
        image_array: Union[List[List[float]], List[List[List[float]]]],
        threshold: Optional[float] = None,
        pixel_size: float = 1.0,
        n_thresholds: Optional[int] = None,
        adaptive: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Minkowski functionals.

        Args:
            image_array: 2D or 3D image array
            threshold: Binarization threshold (single-threshold mode)
            pixel_size: Physical pixel/voxel size
            n_thresholds: If >1, multi-threshold mode: compute MF at multiple
                thresholds and concatenate → 3*n_thresholds dims
            adaptive: If True with n_thresholds, use percentile-based thresholds

        Returns:
            Dictionary with Minkowski functionals
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float64)

            # Multi-threshold mode
            if n_thresholds is not None and n_thresholds > 1:
                return self._run_multi_threshold(img, n_thresholds, adaptive, pixel_size)

            # Single-threshold mode (original behavior)
            if threshold is not None:
                binary = (img > threshold).astype(np.uint8)
            elif img.max() > 1 or img.min() < 0:
                mid = (img.max() + img.min()) / 2
                binary = (img > mid).astype(np.uint8)
            else:
                binary = (img > 0.5).astype(np.uint8)

            is_3d = binary.ndim == 3

            if is_3d:
                mf = self._compute_3d_minkowski(binary, pixel_size)
            else:
                mf = self._compute_2d_minkowski(binary, pixel_size)

            # Compute normalized versions
            total_size = binary.size * (pixel_size ** binary.ndim)
            normalized = self._normalize_functionals(mf, total_size, is_3d)

            # Create feature vector
            feature_vector = list(mf.values()) + list(normalized.values())
            feature_names = [f"MF_{k}" for k in mf.keys()] + [f"MF_{k}_normalized" for k in normalized.keys()]

            return {
                "success": True,
                "tool_name": self.name,
                "minkowski_functionals": mf,
                "normalized": normalized,
                "image_shape": list(img.shape),
                "pixel_size": pixel_size,
                "is_3d": is_3d,
                "foreground_fraction": float(np.sum(binary) / binary.size),
                "combined_vector": feature_vector,  # Standardized key name
                "feature_vector": feature_vector,   # Keep for backwards compatibility
                "feature_names": feature_names,
                "vector_length": len(feature_vector),
                "interpretation": self._interpret(mf, normalized, is_3d)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _run_multi_threshold(
        self,
        img: np.ndarray,
        n_thresholds: int,
        adaptive: bool,
        pixel_size: float,
    ) -> Dict[str, Any]:
        """Multi-threshold Minkowski functionals.

        Computes MF at n_thresholds evenly-spaced thresholds and concatenates
        raw [volume, surface, euler] per threshold → 3*n_thresholds dimensions.

        Args:
            img: Image array (2D or 3D)
            n_thresholds: Number of thresholds
            adaptive: If True, use percentile-based thresholds
            pixel_size: Physical pixel size

        Returns:
            Dictionary with concatenated multi-threshold feature vector
        """
        is_3d = img.ndim == 3

        if adaptive:
            percentiles = np.linspace(10, 90, n_thresholds)
            thresholds = np.percentile(img, percentiles)
        else:
            thresholds = np.linspace(0.05, 0.95, n_thresholds)

        curve = []
        feature_names = []
        for i, t in enumerate(thresholds):
            binary = (img > t).astype(np.uint8)
            if is_3d:
                mf = self._compute_3d_minkowski(binary, pixel_size)
            else:
                mf = self._compute_2d_minkowski(binary, pixel_size)

            curve.extend([
                mf.get('volume', 0.0),
                mf.get('surface', 0.0),
                mf.get('euler_characteristic', 0.0),
            ])
            feature_names.extend([
                f"MF_volume_t{i}", f"MF_surface_t{i}", f"MF_euler_t{i}"
            ])

        feature_vector = [float(v) for v in curve]

        return {
            "success": True,
            "tool_name": self.name,
            "mode": "multi_threshold",
            "n_thresholds": n_thresholds,
            "adaptive": adaptive,
            "thresholds": thresholds.tolist(),
            "image_shape": list(img.shape),
            "pixel_size": pixel_size,
            "is_3d": is_3d,
            "combined_vector": feature_vector,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "vector_length": len(feature_vector),
            "interpretation": (
                f"Multi-threshold Minkowski functionals at {n_thresholds} thresholds "
                f"({'adaptive' if adaptive else 'linear'} spacing). "
                f"Output dimension: {len(feature_vector)}D (3 per threshold)."
            ),
        }

    def _compute_2d_minkowski(self, binary: np.ndarray, pixel_size: float) -> Dict[str, float]:
        """Compute 2D Minkowski functionals.

        Args:
            binary: 2D binary image
            pixel_size: Physical pixel size

        Returns:
            Dictionary with V0 (area), V1 (perimeter), V2 (Euler)
        """
        # V0: Area (number of foreground pixels)
        area = int(np.sum(binary)) * (pixel_size ** 2)

        # V1: Perimeter (boundary length)
        perimeter = self._compute_perimeter_2d(binary) * pixel_size

        # V2: Euler characteristic
        if HAS_EULER_NUMBER:
            euler = ndimage.euler_number(binary, connectivity=1)
        else:
            euler = self._compute_euler_2d(binary)

        return {
            "volume": float(area),
            "surface": float(perimeter),
            "euler_characteristic": float(euler)
        }

    def _compute_3d_minkowski(self, binary: np.ndarray, pixel_size: float) -> Dict[str, float]:
        """Compute 3D Minkowski functionals.

        Args:
            binary: 3D binary image
            pixel_size: Physical voxel size

        Returns:
            Dictionary with V0 (volume), V1 (surface), V2 (mean breadth), V3 (Euler)
        """
        # V0: Volume (number of foreground voxels)
        volume = int(np.sum(binary)) * (pixel_size ** 3)

        # V1: Surface area
        surface_area = self._compute_surface_area_3d(binary) * (pixel_size ** 2)

        # V2: Mean breadth (integral mean curvature)
        mean_breadth = self._compute_mean_breadth_3d(binary) * pixel_size

        # V3: Euler characteristic
        if HAS_EULER_NUMBER:
            euler = ndimage.euler_number(binary, connectivity=1)
        else:
            euler = self._compute_euler_3d(binary)

        return {
            "volume": float(volume),
            "surface": float(surface_area),
            "mean_breadth": float(mean_breadth),
            "euler_characteristic": float(euler)
        }

    def _compute_perimeter_2d(self, binary: np.ndarray) -> float:
        """Compute perimeter of 2D binary image.

        Args:
            binary: 2D binary image

        Returns:
            Perimeter length in pixels
        """
        # Count boundary pixels (foreground with at least one background neighbor)
        if HAS_SCIPY:
            struct = generate_binary_structure(2, 1)
            eroded = binary_erosion(binary, structure=struct)
            boundary = binary.astype(int) - eroded.astype(int)
            perimeter = np.sum(boundary)

            # Multiply by average edge length (4-connected)
            return perimeter * 1.0
        else:
            # Simple boundary counting
            padded = np.pad(binary, 1, mode='constant', constant_values=0)
            boundary = 0
            boundary += np.sum(binary & ~padded[:-2, 1:-1])  # top
            boundary += np.sum(binary & ~padded[2:, 1:-1])   # bottom
            boundary += np.sum(binary & ~padded[1:-1, :-2])  # left
            boundary += np.sum(binary & ~padded[1:-1, 2:])   # right
            return float(boundary)

    def _compute_surface_area_3d(self, binary: np.ndarray) -> float:
        """Compute surface area of 3D binary image using face counting.

        Args:
            binary: 3D binary image

        Returns:
            Surface area in voxel faces
        """
        # Count exposed faces
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        surface = 0

        # Each direction contributes faces where foreground meets background
        surface += np.sum(binary & ~padded[:-2, 1:-1, 1:-1])  # -x
        surface += np.sum(binary & ~padded[2:, 1:-1, 1:-1])   # +x
        surface += np.sum(binary & ~padded[1:-1, :-2, 1:-1])  # -y
        surface += np.sum(binary & ~padded[1:-1, 2:, 1:-1])   # +y
        surface += np.sum(binary & ~padded[1:-1, 1:-1, :-2])  # -z
        surface += np.sum(binary & ~padded[1:-1, 1:-1, 2:])   # +z

        return float(surface)

    def _compute_mean_breadth_3d(self, binary: np.ndarray) -> float:
        """Estimate mean breadth (integral mean curvature) for 3D.

        This is a simplified estimation based on edge counting.

        Args:
            binary: 3D binary image

        Returns:
            Mean breadth estimate
        """
        # Count edges (approximation based on local curvature)
        # Mean breadth is related to the edge count in the boundary
        padded = np.pad(binary, 1, mode='constant', constant_values=0)

        # Count edges along each axis
        edges_x = np.sum(binary[:-1, :, :] != binary[1:, :, :])
        edges_y = np.sum(binary[:, :-1, :] != binary[:, 1:, :])
        edges_z = np.sum(binary[:, :, :-1] != binary[:, :, 1:])

        # Mean breadth approximation (scaled by pi/4 for proper normalization)
        return float((edges_x + edges_y + edges_z) * np.pi / 4)

    def _compute_euler_2d(self, binary: np.ndarray) -> int:
        """Compute 2D Euler characteristic without scipy.

        Uses Python ints to avoid numpy unsigned integer overflow.

        Args:
            binary: 2D binary image

        Returns:
            Euler characteristic
        """
        V = int(np.sum(binary))
        E_h = int(np.sum(binary[:, :-1] & binary[:, 1:]))
        E_v = int(np.sum(binary[:-1, :] & binary[1:, :]))
        F = int(np.sum(binary[:-1, :-1] & binary[:-1, 1:] & binary[1:, :-1] & binary[1:, 1:]))
        return V - E_h - E_v + F

    def _compute_euler_3d(self, binary: np.ndarray) -> int:
        """Compute 3D Euler characteristic without scipy.

        Uses Python ints to avoid numpy unsigned integer overflow.

        Args:
            binary: 3D binary image

        Returns:
            Euler characteristic
        """
        n0 = int(np.sum(binary))
        n1 = (int(np.sum(binary[:-1, :, :] & binary[1:, :, :])) +
              int(np.sum(binary[:, :-1, :] & binary[:, 1:, :])) +
              int(np.sum(binary[:, :, :-1] & binary[:, :, 1:])))

        n2_xy = int(np.sum(binary[:-1, :-1, :] & binary[:-1, 1:, :] &
                       binary[1:, :-1, :] & binary[1:, 1:, :]))
        n2_xz = int(np.sum(binary[:-1, :, :-1] & binary[:-1, :, 1:] &
                       binary[1:, :, :-1] & binary[1:, :, 1:]))
        n2_yz = int(np.sum(binary[:, :-1, :-1] & binary[:, :-1, 1:] &
                       binary[:, 1:, :-1] & binary[:, 1:, 1:]))
        n2 = n2_xy + n2_xz + n2_yz

        n3 = int(np.sum(binary[:-1, :-1, :-1] & binary[:-1, :-1, 1:] &
                    binary[:-1, 1:, :-1] & binary[:-1, 1:, 1:] &
                    binary[1:, :-1, :-1] & binary[1:, :-1, 1:] &
                    binary[1:, 1:, :-1] & binary[1:, 1:, 1:]))

        return n0 - n1 + n2 - n3

    def _normalize_functionals(
        self,
        mf: Dict[str, float],
        total_size: float,
        is_3d: bool
    ) -> Dict[str, float]:
        """Normalize Minkowski functionals.

        Args:
            mf: Raw Minkowski functionals
            total_size: Total image size
            is_3d: Whether image is 3D

        Returns:
            Normalized functionals
        """
        normalized = {
            "volume_fraction": mf["volume"] / max(total_size, 1e-10),
            "surface_density": mf["surface"] / max(total_size, 1e-10),
            "euler_density": mf["euler_characteristic"] / max(total_size, 1e-10)
        }

        if is_3d and "mean_breadth" in mf:
            normalized["breadth_density"] = mf["mean_breadth"] / max(total_size, 1e-10)

        return normalized

    def _interpret(
        self,
        mf: Dict[str, float],
        normalized: Dict[str, float],
        is_3d: bool
    ) -> str:
        """Generate interpretation of Minkowski functionals.

        Args:
            mf: Minkowski functionals
            normalized: Normalized values
            is_3d: Whether image is 3D

        Returns:
            Human-readable interpretation
        """
        vf = normalized["volume_fraction"]
        euler = mf["euler_characteristic"]

        parts = []

        # Volume fraction interpretation
        if vf < 0.1:
            parts.append(f"Sparse structure ({vf*100:.1f}% foreground)")
        elif vf > 0.9:
            parts.append(f"Dense structure ({vf*100:.1f}% foreground)")
        else:
            parts.append(f"Moderate density ({vf*100:.1f}% foreground)")

        # Euler characteristic interpretation
        if euler > 0:
            parts.append(f"Euler χ={int(euler)}: more components than holes")
        elif euler < 0:
            parts.append(f"Euler χ={int(euler)}: more holes than components (porous)")
        else:
            parts.append("Euler χ=0: balanced topology")

        # Surface-to-volume ratio
        if mf["volume"] > 0:
            sv_ratio = mf["surface"] / mf["volume"]
            if sv_ratio > 1:
                parts.append(f"High surface-to-volume ratio ({sv_ratio:.2f}): complex boundary")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
