"""Cubical Complex Tool for TopoAgent.

Builds cubical complex for 2D/3D image data.
"""

from typing import Any, Dict, Optional, Type, List, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class CubicalComplexInput(BaseModel):
    """Input schema for CubicalComplexTool."""
    image_array: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., description="2D or 3D image array"
    )
    filtration_type: str = Field(
        "sublevel", description="Type of filtration: 'sublevel' or 'superlevel'"
    )


class CubicalComplexTool(BaseTool):
    """Build cubical complex from 2D/3D image data.

    Cubical complexes are the natural choice for structured grid data
    like medical images. They directly use the pixel/voxel structure.
    """

    name: str = "cubical_complex"
    description: str = (
        "Build cubical complex from 2D or 3D medical images. "
        "Cubical complexes are ideal for structured grid data (images, CT scans, MRI). "
        "They preserve the natural pixel/voxel structure of the data. "
        "Input: 2D/3D image array, filtration type. "
        "Output: cubical complex ready for persistent homology computation."
    )
    args_schema: Type[BaseModel] = CubicalComplexInput

    def _run(
        self,
        image_array: Union[List[List[float]], List[List[List[float]]]],
        filtration_type: str = "sublevel",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Build cubical complex from image.

        Args:
            image_array: 2D or 3D image array
            filtration_type: 'sublevel' or 'superlevel'

        Returns:
            Dictionary with cubical complex data
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float32)

            # Normalize to [0, 1]
            if img.max() > 1.0:
                img = img / img.max()

            # For superlevel, negate the values
            if filtration_type == "superlevel":
                img = -img

            # Determine dimensionality
            ndim = img.ndim
            shape = img.shape

            # Build cubical complex using GUDHI if available
            try:
                import gudhi
                cubical = gudhi.CubicalComplex(dimensions=list(shape), top_dimensional_cells=img.flatten())

                # Get persistence pairs
                persistence = cubical.persistence()

                # Format persistence pairs
                persistence_pairs = []
                for dim, (birth, death) in persistence:
                    if death == float('inf'):
                        death = 1.0 if filtration_type == "sublevel" else 0.0
                    persistence_pairs.append({
                        "dimension": int(dim),
                        "birth": float(abs(birth)),
                        "death": float(abs(death)),
                        "persistence": float(abs(death - birth))
                    })

                # Betti numbers at various thresholds
                betti_numbers = cubical.betti_numbers()

                return {
                    "success": True,
                    "complex_type": "cubical",
                    "dimensions": ndim,
                    "shape": list(shape),
                    "num_cells": int(img.size),
                    "filtration_type": filtration_type,
                    "persistence_pairs": persistence_pairs,
                    "betti_numbers": list(betti_numbers),
                    "num_features": {
                        f"H{i}": len([p for p in persistence_pairs if p["dimension"] == i])
                        for i in range(ndim)
                    }
                }

            except ImportError:
                # Fallback without GUDHI - provide basic structure
                return {
                    "success": True,
                    "complex_type": "cubical",
                    "dimensions": ndim,
                    "shape": list(shape),
                    "num_cells": int(img.size),
                    "filtration_type": filtration_type,
                    "filtration_values": img.flatten().tolist(),
                    "note": "GUDHI not available. Use compute_ph tool for full persistence computation.",
                    "statistics": {
                        "min": float(img.min()),
                        "max": float(img.max()),
                        "mean": float(img.mean()),
                        "std": float(img.std())
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
