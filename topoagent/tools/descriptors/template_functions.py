"""Template Functions (Tent Functions) Descriptor Tool for TopoAgent.

Computes tent function evaluations on persistence diagrams.
Output: 50D (25 templates × 2 homology dims).

Custom implementation based on Perea, Munch, Khasawneh (2022).

References:
- Perea, Munch, Khasawneh, "Approximating Continuous Functions on
  Persistence Diagrams Using Template Functions" (FoCM 2022)
"""

from typing import Any, Dict, List, Optional, Type
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class TemplateFunctionsInput(BaseModel):
    """Input schema for TemplateFunctionsTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    n_templates: int = Field(
        25, description="Number of template (tent) functions (grid_x × grid_y)"
    )
    template_type: str = Field(
        "tent", description="Template type: 'tent' (piecewise linear) or 'gaussian'"
    )


class TemplateFunctionsTool(BaseTool):
    """Compute template function features from persistence diagrams.

    Places a grid of template functions on the birth-persistence plane.
    For each diagram point, evaluates all template functions and sums
    contributions. This provides a universal approximation framework.

    Tent functions:
    - Piecewise linear, compactly supported on one grid cell
    - Each point contributes only to its local cell
    - Similar to linear interpolation on a mesh

    The framework is universal: any continuous function on the space of
    persistence diagrams can be approximated by template function sums.

    Properties:
    - Universal approximation guarantee (strongest theoretical property)
    - Piecewise linear (tent) or smooth (Gaussian) basis
    - Local support → sparse, interpretable
    - Grid resolution controls expressiveness vs dimension

    Output: n_templates × n_homology_dims
    Default: 25 × 2 = 50D (5×5 grid per dimension)
    """

    name: str = "template_functions"
    description: str = (
        "Compute template function (tent) features from persistence diagrams. "
        "Universal approximation on diagram space via grid of basis functions. "
        "Input: persistence data from compute_ph tool. "
        "Output: 50D default (25 templates × 2 hom dims)."
    )
    args_schema: Type[BaseModel] = TemplateFunctionsInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        n_templates: int = 25,
        template_type: str = "tent",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            all_vectors = []

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]
                vec = self._compute_templates(pairs, n_templates, template_type)
                all_vectors.append(vec)

            combined_vector = np.concatenate(all_vectors).tolist()

            return {
                "success": True,
                "tool_name": self.name,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "n_templates": n_templates,
                "template_type": template_type,
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _compute_templates(
        self,
        pairs: List[Dict[str, float]],
        n_templates: int,
        template_type: str
    ) -> np.ndarray:
        """Compute template function values for one homology dimension.

        Works in birth-persistence coordinates (x=birth, y=death-birth).
        """
        # Extract valid pairs in (birth, persistence) coordinates
        points = []
        for p in pairs:
            if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                continue
            b, d = p["birth"], p["death"]
            if np.isfinite(d) and d > b:
                points.append((b, d - b))  # (birth, persistence)

        if not points:
            return np.zeros(n_templates)

        points = np.array(points)
        births, persistences = points[:, 0], points[:, 1]

        # Determine grid dimensions (as close to square as possible)
        grid_size = int(np.ceil(np.sqrt(n_templates)))
        actual_n = grid_size * grid_size

        # Define grid on the birth-persistence plane
        b_min, b_max = births.min(), births.max()
        p_min, p_max = 0, persistences.max()

        # Add small margins
        b_margin = max((b_max - b_min) * 0.1, 1e-6)
        p_margin = max((p_max - p_min) * 0.1, 1e-6)
        b_min -= b_margin
        b_max += b_margin
        p_max += p_margin

        # Grid centers
        b_centers = np.linspace(b_min, b_max, grid_size)
        p_centers = np.linspace(p_min, p_max, grid_size)

        # Grid spacing
        db = (b_max - b_min) / max(grid_size - 1, 1)
        dp = (p_max - p_min) / max(grid_size - 1, 1)

        # Compute template function values
        result = np.zeros(actual_n)

        for birth, pers in zip(births, persistences):
            for i, bc in enumerate(b_centers):
                for j, pc in enumerate(p_centers):
                    idx = i * grid_size + j
                    if template_type == "tent":
                        val = self._tent(birth, pers, bc, pc, db, dp)
                    else:  # gaussian
                        val = self._gaussian(birth, pers, bc, pc, db, dp)
                    result[idx] += val

        # Truncate to requested n_templates
        return result[:n_templates]

    @staticmethod
    def _tent(x: float, y: float, cx: float, cy: float, dx: float, dy: float) -> float:
        """Evaluate tent function centered at (cx, cy) with support (dx, dy)."""
        if dx < 1e-10 or dy < 1e-10:
            return 0.0
        tx = max(0, 1 - abs(x - cx) / dx)
        ty = max(0, 1 - abs(y - cy) / dy)
        return tx * ty

    @staticmethod
    def _gaussian(x: float, y: float, cx: float, cy: float, dx: float, dy: float) -> float:
        """Evaluate Gaussian function centered at (cx, cy)."""
        sigma_x = dx / 2
        sigma_y = dy / 2
        if sigma_x < 1e-10 or sigma_y < 1e-10:
            return 0.0
        return np.exp(-((x - cx)**2 / (2 * sigma_x**2) + (y - cy)**2 / (2 * sigma_y**2)))

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        return self._run(*args, **kwargs)
