"""Euler Characteristic Tool for TopoAgent.

Compute the Euler characteristic from binary images.
χ = V - E + F (2D) or χ = β₀ - β₁ + β₂ (3D)
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import scipy.ndimage for euler_number
try:
    from scipy import ndimage
    # Check if euler_number function exists (added in scipy 1.4.0)
    HAS_EULER_NUMBER = hasattr(ndimage, 'euler_number')
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    HAS_EULER_NUMBER = False


class EulerCharacteristicInput(BaseModel):
    """Input schema for EulerCharacteristicTool."""
    image_array: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., description="2D or 3D binary image array"
    )
    connectivity: int = Field(
        4, description="Connectivity for 2D (4 or 8) or 3D (6, 18, or 26)"
    )
    threshold: Optional[float] = Field(
        None, description="Binarization threshold if image is grayscale (default: 0.5)"
    )


class EulerCharacteristicTool(BaseTool):
    """Compute Euler characteristic from binary images.

    The Euler characteristic (χ) is a fundamental topological invariant:
    - 2D: χ = V - E + F (vertices - edges + faces)
    - 3D: χ = β₀ - β₁ + β₂ (alternating sum of Betti numbers)

    For a simply connected region: χ = 1
    For a region with n holes: χ = 1 - n
    """

    name: str = "euler_characteristic"
    description: str = (
        "Compute Euler characteristic (χ) from a binary image. "
        "This topological invariant measures: χ = components - holes + cavities. "
        "Input: 2D or 3D binary image array. "
        "Output: Euler characteristic value and interpretation. "
        "Useful for quantifying image topology and texture complexity."
    )
    args_schema: Type[BaseModel] = EulerCharacteristicInput

    def _run(
        self,
        image_array: Union[List[List[float]], List[List[List[float]]]],
        connectivity: int = 4,
        threshold: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute Euler characteristic.

        Args:
            image_array: 2D or 3D image array
            connectivity: Neighborhood connectivity (4, 8 for 2D; 6, 18, 26 for 3D)
            threshold: Binarization threshold

        Returns:
            Dictionary with Euler characteristic and related metrics
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float64)

            # Binarize if needed
            if threshold is not None:
                img = (img > threshold).astype(np.uint8)
            elif img.max() > 1 or img.min() < 0:
                # Auto-binarize using 0.5 * (max + min)
                mid = (img.max() + img.min()) / 2
                img = (img > mid).astype(np.uint8)
            else:
                img = (img > 0.5).astype(np.uint8)

            ndim = img.ndim
            is_3d = ndim == 3

            # Validate connectivity
            if is_3d:
                if connectivity not in [6, 18, 26]:
                    connectivity = 6  # Default for 3D
            else:
                if connectivity not in [4, 8]:
                    connectivity = 4  # Default for 2D

            # Compute Euler characteristic
            if HAS_EULER_NUMBER:
                euler = self._compute_scipy(img, connectivity)
            else:
                euler = self._compute_fallback(img, connectivity)

            # Compute Betti numbers for interpretation
            betti = self._estimate_betti_numbers(img, connectivity)

            # Generate interpretation
            interpretation = self._interpret_euler(euler, betti, is_3d)

            return {
                "success": True,
                "tool_name": self.name,
                "euler_characteristic": int(euler),
                "betti_numbers": betti,
                "alternating_sum": f"β₀ - β₁ + β₂ = {betti['H0']} - {betti['H1']} + {betti.get('H2', 0)} = {euler}",
                "connectivity": connectivity,
                "image_shape": list(img.shape),
                "foreground_fraction": float(np.sum(img) / img.size),
                "interpretation": interpretation
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_scipy(self, img: np.ndarray, connectivity: int) -> int:
        """Compute Euler characteristic using scipy.ndimage.

        Args:
            img: Binary image array
            connectivity: Neighborhood connectivity

        Returns:
            Euler characteristic value
        """
        return ndimage.euler_number(img, connectivity=connectivity)

    def _compute_fallback(self, img: np.ndarray, connectivity: int) -> int:
        """Fallback computation without scipy.

        Uses the formula: χ = Σ(-1)^k * n_k where n_k is the number of k-cubes.

        Args:
            img: Binary image array
            connectivity: Neighborhood connectivity

        Returns:
            Euler characteristic value
        """
        if img.ndim == 2:
            return self._compute_2d_euler(img, connectivity)
        else:
            return self._compute_3d_euler(img, connectivity)

    def _compute_2d_euler(self, img: np.ndarray, connectivity: int) -> int:
        """Compute Euler characteristic for 2D image.

        For 4-connectivity: χ = V - E + F where
        - V = number of foreground pixels
        - E = number of horizontal + vertical adjacencies
        - F = number of 2x2 foreground squares

        Args:
            img: 2D binary image
            connectivity: 4 or 8

        Returns:
            Euler characteristic
        """
        # Count vertices (foreground pixels)
        V = np.sum(img)

        # Count edges (adjacencies)
        E_h = np.sum(img[:, :-1] & img[:, 1:])  # Horizontal
        E_v = np.sum(img[:-1, :] & img[1:, :])  # Vertical
        E = E_h + E_v

        # Count faces (2x2 squares)
        F = np.sum(img[:-1, :-1] & img[:-1, 1:] & img[1:, :-1] & img[1:, 1:])

        if connectivity == 4:
            return int(V - E + F)
        else:
            # 8-connectivity: also count diagonal edges
            E_d1 = np.sum(img[:-1, :-1] & img[1:, 1:])
            E_d2 = np.sum(img[:-1, 1:] & img[1:, :-1])
            return int(V - E - E_d1 - E_d2 + F)

    def _compute_3d_euler(self, img: np.ndarray, connectivity: int) -> int:
        """Compute Euler characteristic for 3D image.

        Args:
            img: 3D binary image
            connectivity: 6, 18, or 26

        Returns:
            Euler characteristic
        """
        # For 3D, use the formula χ = n0 - n1 + n2 - n3
        # where n_k is number of k-dimensional elements

        # n0: voxels
        n0 = np.sum(img)

        # n1: edges (6 directions for 6-connectivity)
        n1 = (np.sum(img[:-1, :, :] & img[1:, :, :]) +  # x edges
              np.sum(img[:, :-1, :] & img[:, 1:, :]) +  # y edges
              np.sum(img[:, :, :-1] & img[:, :, 1:]))   # z edges

        # n2: faces (squares)
        n2_xy = np.sum(img[:-1, :-1, :] & img[:-1, 1:, :] &
                       img[1:, :-1, :] & img[1:, 1:, :])
        n2_xz = np.sum(img[:-1, :, :-1] & img[:-1, :, 1:] &
                       img[1:, :, :-1] & img[1:, :, 1:])
        n2_yz = np.sum(img[:, :-1, :-1] & img[:, :-1, 1:] &
                       img[:, 1:, :-1] & img[:, 1:, 1:])
        n2 = n2_xy + n2_xz + n2_yz

        # n3: cubes (2x2x2)
        n3 = np.sum(img[:-1, :-1, :-1] & img[:-1, :-1, 1:] &
                    img[:-1, 1:, :-1] & img[:-1, 1:, 1:] &
                    img[1:, :-1, :-1] & img[1:, :-1, 1:] &
                    img[1:, 1:, :-1] & img[1:, 1:, 1:])

        return int(n0 - n1 + n2 - n3)

    def _estimate_betti_numbers(self, img: np.ndarray, connectivity: int) -> Dict[str, int]:
        """Estimate Betti numbers from the image.

        Args:
            img: Binary image array
            connectivity: Neighborhood connectivity

        Returns:
            Dictionary with estimated Betti numbers
        """
        # β₀: number of connected components
        if HAS_SCIPY:
            try:
                if img.ndim == 2:
                    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
                else:
                    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
                labeled, num_components = ndimage.label(img, structure=structure)
                b0 = num_components
            except Exception:
                b0 = self._count_components_fallback(img)
        else:
            b0 = self._count_components_fallback(img)

        betti = {"H0": int(b0)}

        if img.ndim == 2:
            # β₁: number of holes (estimate from Euler characteristic)
            euler = self._compute_fallback(img, connectivity)
            b1 = b0 - euler  # χ = β₀ - β₁, so β₁ = β₀ - χ
            betti["H1"] = int(max(0, b1))
        else:
            # 3D case: estimate β₁ (tunnels) and β₂ (cavities)
            euler = self._compute_fallback(img, connectivity)
            # χ = β₀ - β₁ + β₂
            # Count cavities by inverting and counting components
            try:
                if HAS_SCIPY:
                    _, b2 = ndimage.label(1 - img)
                    b2 = max(0, b2 - 1)  # Subtract background
                else:
                    b2 = 0
            except Exception:
                b2 = 0
            b1 = b0 + b2 - euler
            betti["H1"] = int(max(0, b1))
            betti["H2"] = int(b2)

        return betti

    def _count_components_fallback(self, img: np.ndarray) -> int:
        """Simple connected components count without scipy.

        Args:
            img: Binary image array

        Returns:
            Number of connected components
        """
        # Simple flood-fill approach for fallback
        visited = np.zeros_like(img, dtype=bool)
        count = 0

        def flood_fill(start):
            stack = [start]
            while stack:
                pos = stack.pop()
                if visited[pos]:
                    continue
                if not img[pos]:
                    continue
                visited[pos] = True
                # Add neighbors
                for d in range(img.ndim):
                    for delta in [-1, 1]:
                        neighbor = list(pos)
                        neighbor[d] += delta
                        neighbor = tuple(neighbor)
                        if all(0 <= neighbor[i] < img.shape[i] for i in range(img.ndim)):
                            if not visited[neighbor] and img[neighbor]:
                                stack.append(neighbor)

        for idx in np.ndindex(img.shape):
            if img[idx] and not visited[idx]:
                flood_fill(idx)
                count += 1

        return count

    def _interpret_euler(self, euler: int, betti: Dict[str, int], is_3d: bool) -> str:
        """Generate interpretation of Euler characteristic.

        Args:
            euler: Euler characteristic value
            betti: Estimated Betti numbers
            is_3d: Whether image is 3D

        Returns:
            Human-readable interpretation
        """
        b0 = betti["H0"]
        b1 = betti.get("H1", 0)

        if is_3d:
            b2 = betti.get("H2", 0)
            if euler == 1 and b0 == 1 and b1 == 0 and b2 == 0:
                return "Simply connected solid region (topologically equivalent to a ball)"
            elif b0 == 1 and b1 > 0:
                return f"Single connected region with {b1} tunnel(s)/handle(s)"
            elif b0 > 1:
                return f"{b0} disconnected components"
            else:
                return f"Complex topology: {b0} component(s), {b1} tunnel(s), {b2} cavity/cavities"
        else:
            if euler == 1 and b0 == 1:
                return "Simply connected region (no holes)"
            elif euler == 0 and b0 == 1:
                return "Single region with 1 hole (like a ring/annulus)"
            elif b0 == 1 and b1 > 1:
                return f"Single connected region with {b1} holes"
            elif b0 > 1:
                return f"{b0} disconnected components with {b1} total holes"
            else:
                return f"Euler characteristic χ = {euler}: {b0} components, {b1} holes"

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
