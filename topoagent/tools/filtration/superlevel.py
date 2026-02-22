"""Superlevel Filtration Tool for TopoAgent.

Creates superlevel set filtration for dark features on bright background.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class SuperlevelFiltrationInput(BaseModel):
    """Input schema for SuperlevelFiltrationTool."""
    image_array: List[List[float]] = Field(..., description="2D image array (grayscale, normalized)")
    num_thresholds: int = Field(100, description="Number of filtration levels")


class SuperlevelFiltrationTool(BaseTool):
    """Create superlevel set filtration for topological analysis.

    Superlevel sets capture regions where pixel values are above a threshold.
    Best for detecting dark features on bright background (vessels, cavities).
    """

    name: str = "superlevel_filtration"
    description: str = (
        "Create superlevel set filtration for persistent homology computation. "
        "BEST FOR: Dark features on bright background (blood vessels, cavities, airways). "
        "The filtration tracks how connected components merge as the threshold decreases. "
        "Input: normalized grayscale image array (0-1). "
        "Output: filtration data structure ready for persistent homology computation."
    )
    args_schema: Type[BaseModel] = SuperlevelFiltrationInput

    def _run(
        self,
        image_array: List[List[float]],
        num_thresholds: int = 100,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Create superlevel set filtration.

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

            # For superlevel filtration, we invert the image
            # This converts it to a sublevel filtration problem
            inverted_img = 1.0 - img

            # Generate thresholds (decreasing for superlevel)
            thresholds = np.linspace(1, 0, num_thresholds)

            # Filtration values are the inverted image values
            filtration_values = inverted_img.flatten()

            # Basic statistics
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
                size = np.sum(img >= t) / img.size
                level_set_sizes.append({"threshold": float(t), "relative_size": float(size)})

            return {
                "success": True,
                "filtration_type": "superlevel",
                "filtration_values": filtration_values.tolist(),
                "image_shape": list(img.shape),
                "num_thresholds": num_thresholds,
                "statistics": stats,
                "level_set_growth": level_set_sizes,
                "description": "Superlevel filtration captures dark features by tracking regions above threshold."
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
