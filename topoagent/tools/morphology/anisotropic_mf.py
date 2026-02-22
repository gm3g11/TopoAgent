"""Anisotropic Minkowski Functionals Tool for TopoAgent.

Compute directional Minkowski functionals for 3D structural analysis.
Specialized for bone microarchitecture and material science applications.
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import scipy for linear algebra
try:
    from scipy import ndimage
    from scipy.linalg import eigh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AnisotropicMFInput(BaseModel):
    """Input schema for AnisotropicMFTool."""
    image_array: List[List[List[float]]] = Field(
        ..., description="3D binary image array"
    )
    n_directions: int = Field(
        13, description="Number of directions for analysis (13 or 26)"
    )
    threshold: Optional[float] = Field(
        None, description="Binarization threshold"
    )


class AnisotropicMFTool(BaseTool):
    """Compute anisotropic Minkowski functionals for 3D analysis.

    Analyzes structural directionality by computing Minkowski functionals
    along multiple directions. Used for:
    - Bone microarchitecture analysis (trabecular orientation)
    - Material science (fiber orientation)
    - Tissue structure analysis

    Outputs fabric tensor and degree of anisotropy (DA).

    References:
    - Trabecular bone analysis using anisotropic MFs
    - Fabric tensor computation for structural materials
    """

    name: str = "anisotropic_mf"
    description: str = (
        "Compute anisotropic Minkowski functionals for 3D structural analysis. "
        "Analyzes directionality via fabric tensor and degree of anisotropy. "
        "Input: 3D binary image array. "
        "Output: directional MFs, fabric tensor, principal directions, DA. "
        "Specialized for bone microstructure and material analysis."
    )
    args_schema: Type[BaseModel] = AnisotropicMFInput

    def _run(
        self,
        image_array: List[List[List[float]]],
        n_directions: int = 13,
        threshold: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute anisotropic Minkowski functionals.

        Args:
            image_array: 3D image array
            n_directions: Number of analysis directions
            threshold: Binarization threshold

        Returns:
            Dictionary with anisotropic MFs and fabric tensor
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

            # Binarize
            if threshold is not None:
                binary = (img > threshold).astype(np.uint8)
            elif img.max() > 1 or img.min() < 0:
                mid = (img.max() + img.min()) / 2
                binary = (img > mid).astype(np.uint8)
            else:
                binary = (img > 0.5).astype(np.uint8)

            # Get direction vectors
            directions = self._get_directions(n_directions)

            # Compute MF for each direction
            directional_mf = []
            for direction in directions:
                mf = self._compute_directional_mf(binary, direction)
                directional_mf.append({
                    "direction": direction.tolist(),
                    "surface_area": mf["surface"],
                    "mean_intercept_length": mf["mil"]
                })

            # Compute fabric tensor
            fabric_tensor = self._compute_fabric_tensor(directional_mf)

            # Eigenanalysis
            if HAS_SCIPY:
                eigenvalues, eigenvectors = eigh(fabric_tensor)
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(fabric_tensor)

            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Degree of anisotropy
            da = 1 - (eigenvalues[2] / max(eigenvalues[0], 1e-10))

            # Fractional anisotropy (FA)
            mean_ev = np.mean(eigenvalues)
            if mean_ev > 0:
                fa = np.sqrt(3/2) * np.sqrt(
                    np.sum((eigenvalues - mean_ev) ** 2) /
                    np.sum(eigenvalues ** 2)
                )
            else:
                fa = 0.0

            # Create feature vector
            feature_vector = [
                float(da),
                float(fa),
                float(eigenvalues[0]),
                float(eigenvalues[1]),
                float(eigenvalues[2]),
                float(eigenvalues[0] / max(eigenvalues[2], 1e-10))  # eigenvalue ratio
            ]
            feature_names = [
                "degree_of_anisotropy", "fractional_anisotropy",
                "eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
                "eigenvalue_ratio"
            ]

            return {
                "success": True,
                "tool_name": self.name,
                "directional_mf": directional_mf,
                "fabric_tensor": fabric_tensor.tolist(),
                "eigenvalues": eigenvalues.tolist(),
                "principal_directions": eigenvectors.T.tolist(),  # Rows are eigenvectors
                "degree_of_anisotropy": float(da),
                "fractional_anisotropy": float(fa),
                "n_directions": n_directions,
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "interpretation": self._interpret(da, fa, eigenvectors)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _get_directions(self, n_directions: int) -> np.ndarray:
        """Get uniformly distributed direction vectors.

        Args:
            n_directions: Number of directions (13 or 26)

        Returns:
            Array of unit direction vectors
        """
        if n_directions == 13:
            # 13 unique directions (half of 26-connectivity neighbors)
            directions = [
                [1, 0, 0], [0, 1, 0], [0, 0, 1],  # axis-aligned
                [1, 1, 0], [1, 0, 1], [0, 1, 1],  # face diagonals
                [1, -1, 0], [1, 0, -1], [0, 1, -1],
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]  # space diagonals
            ]
        else:
            # 26-connectivity: all neighbors of a central voxel
            directions = []
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    for z in [-1, 0, 1]:
                        if x != 0 or y != 0 or z != 0:
                            directions.append([x, y, z])

        directions = np.array(directions, dtype=np.float64)
        # Normalize
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        return directions / norms

    def _compute_directional_mf(
        self,
        binary: np.ndarray,
        direction: np.ndarray
    ) -> Dict[str, float]:
        """Compute Minkowski functionals along a direction.

        Args:
            binary: 3D binary image
            direction: Unit direction vector

        Returns:
            Dictionary with directional surface and MIL
        """
        # Shift the image in the given direction
        shift = np.round(direction).astype(int)

        # Compute boundary along this direction
        # Boundary = foreground voxels with background neighbor in this direction
        if shift[0] != 0:
            if shift[0] > 0:
                shifted = np.pad(binary[1:, :, :], ((0, 1), (0, 0), (0, 0)), constant_values=0)
            else:
                shifted = np.pad(binary[:-1, :, :], ((1, 0), (0, 0), (0, 0)), constant_values=0)
        else:
            shifted = binary.copy()

        if shift[1] != 0:
            if shift[1] > 0:
                temp = np.pad(shifted[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=0)
            else:
                temp = np.pad(shifted[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=0)
            shifted = temp

        if shift[2] != 0:
            if shift[2] > 0:
                temp = np.pad(shifted[:, :, 1:], ((0, 0), (0, 0), (0, 1)), constant_values=0)
            else:
                temp = np.pad(shifted[:, :, :-1], ((0, 0), (0, 0), (1, 0)), constant_values=0)
            shifted = temp

        # Boundary in this direction
        boundary = binary.astype(int) - (binary & shifted).astype(int)
        surface = np.sum(boundary > 0)

        # Mean Intercept Length (MIL) approximation
        # MIL = 2 * volume / surface_area
        volume = np.sum(binary)
        mil = 2 * volume / max(surface, 1) if surface > 0 else 0

        return {
            "surface": float(surface),
            "mil": float(mil)
        }

    def _compute_fabric_tensor(
        self,
        directional_mf: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Compute fabric tensor from directional measurements.

        Args:
            directional_mf: List of directional MF results

        Returns:
            3x3 fabric tensor
        """
        # Initialize fabric tensor
        H = np.zeros((3, 3))

        for item in directional_mf:
            direction = np.array(item["direction"])
            mil = item["mean_intercept_length"]

            # Add outer product weighted by MIL^2
            H += (mil ** 2) * np.outer(direction, direction)

        # Normalize
        H /= len(directional_mf)

        return H

    def _interpret(
        self,
        da: float,
        fa: float,
        eigenvectors: np.ndarray
    ) -> str:
        """Generate interpretation of anisotropy results.

        Args:
            da: Degree of anisotropy
            fa: Fractional anisotropy
            eigenvectors: Principal direction eigenvectors

        Returns:
            Human-readable interpretation
        """
        parts = []

        # Anisotropy interpretation
        if da < 0.2:
            parts.append(f"Nearly isotropic structure (DA={da:.2f})")
        elif da < 0.5:
            parts.append(f"Moderately anisotropic (DA={da:.2f})")
        elif da < 0.8:
            parts.append(f"Strongly anisotropic (DA={da:.2f})")
        else:
            parts.append(f"Highly anisotropic structure (DA={da:.2f})")

        # Principal direction
        principal = eigenvectors[:, 0]
        abs_principal = np.abs(principal)
        max_idx = np.argmax(abs_principal)
        axis_names = ["X", "Y", "Z"]

        if abs_principal[max_idx] > 0.9:
            parts.append(f"Primarily aligned along {axis_names[max_idx]}-axis")
        else:
            parts.append(f"Principal direction: [{principal[0]:.2f}, {principal[1]:.2f}, {principal[2]:.2f}]")

        # FA interpretation (for comparison with DTI)
        if fa > 0:
            parts.append(f"Fractional anisotropy FA={fa:.2f}")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
