"""Persistence Image Tool for TopoAgent.

Convert persistence diagrams to fixed-size vector representations.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class PersistenceImageInput(BaseModel):
    """Input schema for PersistenceImageTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    resolution: int = Field(20, description="Resolution of the persistence image (NxN grid)")
    sigma: float = Field(0.1, description="Gaussian kernel bandwidth")
    weight_function: str = Field("linear", description="Weight function: 'linear', 'squared', 'const'")


class PersistenceImageTool(BaseTool):
    """Convert persistence diagrams to fixed-size vector representations.

    Persistence images are stable vectorizations of persistence diagrams
    that can be used directly with machine learning classifiers.
    """

    name: str = "persistence_image"
    description: str = (
        "Convert persistence diagrams to fixed-size feature vectors. "
        "Persistence images are stable vectorizations suitable for ML classifiers. "
        "Each persistence diagram dimension produces an NxN image/vector. "
        "Input: persistence data, resolution, kernel bandwidth. "
        "Output: flattened feature vectors for each homology dimension."
    )
    args_schema: Type[BaseModel] = PersistenceImageInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        resolution: int = 20,
        sigma: float = 0.1,
        weight_function: str = "linear",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Generate persistence images.

        Args:
            persistence_data: Persistence pairs by dimension
            resolution: Image resolution (NxN)
            sigma: Gaussian kernel bandwidth
            weight_function: Weight function type

        Returns:
            Dictionary with persistence images and feature vectors
        """
        try:
            images = {}
            feature_vectors = {}

            for dim_key, pairs in persistence_data.items():
                if not pairs or not isinstance(pairs, list):
                    continue

                # Skip if not proper persistence pairs
                if not all(isinstance(p, dict) and "birth" in p and "death" in p for p in pairs):
                    continue

                # Convert to birth/persistence coordinates
                points = []
                for p in pairs:
                    birth = p["birth"]
                    death = p["death"]
                    persistence = death - birth
                    if persistence > 0:
                        points.append((birth, persistence))

                if not points:
                    # Empty diagram
                    images[dim_key] = np.zeros((resolution, resolution))
                    feature_vectors[dim_key] = [0.0] * (resolution * resolution)
                    continue

                # Compute persistence image
                pi = self._compute_persistence_image(
                    points, resolution, sigma, weight_function
                )

                images[dim_key] = pi
                feature_vectors[dim_key] = pi.flatten().tolist()

            # Combine all dimensions into single feature vector
            # Always include H0 and H1 (pad with zeros if missing) for consistent 800D output
            combined_vector = []
            for dim in ['H0', 'H1']:
                if dim in feature_vectors:
                    combined_vector.extend(feature_vectors[dim])
                else:
                    # Pad with zeros for missing dimension
                    combined_vector.extend([0.0] * (resolution * resolution))

            return {
                "success": True,
                "images": {k: v.tolist() for k, v in images.items()},
                "feature_vectors": feature_vectors,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "resolution": resolution,
                "sigma": sigma,
                "weight_function": weight_function,
                "dimensions_processed": list(images.keys())
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _compute_persistence_image(
        self,
        points: List[tuple],
        resolution: int,
        sigma: float,
        weight_function: str
    ) -> np.ndarray:
        """Compute persistence image from birth-persistence points.

        Uses vectorized numpy operations for efficiency.

        Args:
            points: List of (birth, persistence) tuples
            resolution: Grid resolution
            sigma: Gaussian bandwidth
            weight_function: Weight function type

        Returns:
            2D persistence image array
        """
        if not points:
            return np.zeros((resolution, resolution))

        # Convert to numpy arrays
        points_arr = np.array(points)
        births = points_arr[:, 0]
        persistences = points_arr[:, 1]

        birth_min, birth_max = births.min(), births.max()
        pers_max = persistences.max()

        # Add padding
        birth_range = birth_max - birth_min if birth_max > birth_min else 1
        pers_range = pers_max if pers_max > 0 else 1

        # Create grid
        birth_bins = np.linspace(birth_min - 0.1 * birth_range, birth_max + 0.1 * birth_range, resolution)
        pers_bins = np.linspace(0, pers_max + 0.1 * pers_range, resolution)

        # Create meshgrid for vectorized computation
        birth_grid, pers_grid = np.meshgrid(birth_bins, pers_bins)  # Both (resolution, resolution)

        # Compute weights based on persistence
        if weight_function == "linear":
            weights = persistences
        elif weight_function == "squared":
            weights = persistences ** 2
        else:  # const
            weights = np.ones(len(persistences))

        # Vectorized Gaussian computation
        # For each point, compute contribution to entire grid at once
        image = np.zeros((resolution, resolution))
        sigma_sq_2 = 2 * sigma ** 2

        for i in range(len(births)):
            dist_sq = (birth_grid - births[i]) ** 2 + (pers_grid - persistences[i]) ** 2
            image += weights[i] * np.exp(-dist_sq / sigma_sq_2)

        # Normalize
        if image.max() > 0:
            image = image / image.max()

        return image

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
