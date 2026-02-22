"""Sublevel Filtration Tool for TopoAgent.

Creates sublevel set filtration for bright features on dark background.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class SublevelFiltrationInput(BaseModel):
    """Input schema for SublevelFiltrationTool."""
    image_array: List[List[float]] = Field(..., description="2D image array (grayscale, normalized)")
    num_thresholds: int = Field(100, description="Number of filtration levels")


class SublevelFiltrationTool(BaseTool):
    """Create sublevel set filtration for topological analysis.

    Sublevel sets capture regions where pixel values are below a threshold.
    Best for detecting bright features on dark background (lesions, nodules).
    """

    name: str = "sublevel_filtration"
    description: str = (
        "Create sublevel set filtration for persistent homology computation. "
        "BEST FOR: Bright features on dark background (skin lesions, lung nodules, tumors). "
        "The filtration tracks how connected components merge as the threshold increases. "
        "Input: normalized grayscale image array (0-1). "
        "Output: filtration data structure ready for persistent homology computation."
    )
    args_schema: Type[BaseModel] = SublevelFiltrationInput

    def _run(
        self,
        image_array: List[List[float]],
        num_thresholds: int = 100,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Create sublevel set filtration.

        Args:
            image_array: 2D grayscale image array
            num_thresholds: Number of filtration levels

        Returns:
            Dictionary with filtration data
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float32)

            # Ensure values are in [0, 1]
            if img.max() > 1.0:
                img = img / img.max()

            # Generate thresholds
            thresholds = np.linspace(0, 1, num_thresholds)

            # Compute sublevel sets at each threshold
            # Store as birth/death times for components
            filtration_values = img.flatten()

            # Basic statistics about the filtration
            stats = {
                "min_value": float(img.min()),
                "max_value": float(img.max()),
                "mean_value": float(img.mean()),
                "std_value": float(img.std()),
                "num_pixels": int(img.size),
                "shape": list(img.shape)
            }

            # Compute level set sizes at various thresholds
            level_set_sizes = []
            for t in thresholds[::10]:  # Sample every 10th threshold
                size = np.sum(img <= t) / img.size
                level_set_sizes.append({"threshold": float(t), "relative_size": float(size)})

            return {
                "success": True,
                "filtration_type": "sublevel",
                "filtration_values": filtration_values.tolist(),
                "image_shape": list(img.shape),
                "num_thresholds": num_thresholds,
                "statistics": stats,
                "level_set_growth": level_set_sizes,
                "description": "Sublevel filtration captures bright features by tracking regions below threshold."
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
