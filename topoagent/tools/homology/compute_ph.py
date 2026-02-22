"""Compute Persistent Homology Tool for TopoAgent.

Computes persistent homology (H0, H1, H2) from image data.
"""

from typing import Any, Dict, Optional, Type, List, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class ComputePHInput(BaseModel):
    """Input schema for ComputePHTool."""
    image_array: Union[List[List[float]], List[List[List[float]]]] = Field(
        ..., description="2D or 3D image array (or filtration values)"
    )
    filtration_type: str = Field(
        "sublevel", description="Filtration type: 'sublevel' or 'superlevel'"
    )
    max_dimension: int = Field(1, description="Maximum homology dimension to compute (0, 1, or 2)")


class ComputePHTool(BaseTool):
    """Compute persistent homology from image data.

    Persistent homology captures topological features across scales:
    - H0: Connected components (regions, objects)
    - H1: Loops/holes (ring structures, boundaries)
    - H2: Voids (cavities, 3D only)
    """

    name: str = "compute_ph"
    description: str = (
        "Compute persistent homology (H0, H1, H2) from medical images. "
        "H0 captures connected components (number of distinct regions). "
        "H1 captures loops/holes (ring structures, boundaries). "
        "H2 captures voids (cavities in 3D data). "
        "Input: image array, filtration type, max dimension. "
        "Output: persistence pairs (birth, death) for each dimension."
    )
    args_schema: Type[BaseModel] = ComputePHInput

    def _run(
        self,
        image_array: Union[List[List[float]], List[List[List[float]]]],
        filtration_type: str = "sublevel",
        max_dimension: int = 1,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute persistent homology.

        Args:
            image_array: 2D or 3D image array
            filtration_type: 'sublevel' or 'superlevel'
            max_dimension: Max homology dimension

        Returns:
            Dictionary with persistence data
        """
        try:
            # Robustly convert to numpy array (handles corrupted JSON lists)
            if isinstance(image_array, np.ndarray):
                img = image_array.astype(np.float32)
            elif isinstance(image_array, list):
                # Validate structure before conversion
                try:
                    img = np.array(image_array, dtype=np.float32)
                except ValueError:
                    # Inhomogeneous list — try to fix by ensuring uniform row lengths
                    if all(isinstance(row, list) for row in image_array):
                        min_len = min(len(row) for row in image_array)
                        img = np.array([row[:min_len] for row in image_array], dtype=np.float32)
                    else:
                        raise
            else:
                img = np.array(image_array, dtype=np.float32)

            # Handle RGB images: convert to grayscale for PH computation.
            # Design note: PH is always computed on the grayscale-converted image.
            # Color mode (per_channel vs grayscale) affects DOWNSTREAM vectorization
            # only: per_channel mode computes the descriptor on R, G, B channels
            # separately and concatenates for 3x feature dimensions. This matches
            # the benchmark4 pipeline design.
            if img.ndim == 3 and img.shape[2] in (3, 4):
                img = np.mean(img[:, :, :3], axis=2).astype(np.float32)

            # Normalize to [0, 1]
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())

            # For superlevel, use monotone transform (1.0 - img) instead of negation
            # This keeps birth < death valid, avoiding the abs() bug
            if filtration_type == "superlevel":
                img = 1.0 - img

            # Try to use GUDHI
            try:
                import gudhi

                # Create cubical complex
                cubical = gudhi.CubicalComplex(
                    dimensions=list(img.shape),
                    top_dimensional_cells=img.flatten()
                )

                # Compute persistence
                cubical.compute_persistence()
                persistence = cubical.persistence_intervals_in_dimension

                # Extract persistence pairs by dimension
                persistence_by_dim = {}
                for dim in range(max_dimension + 1):
                    intervals = persistence(dim)
                    pairs = []
                    for birth, death in intervals:
                        if death == float('inf'):
                            death = img.max()  # Max value in the (possibly transformed) image
                        # No abs() needed - with monotone transform, birth < death is always valid
                        pairs.append({
                            "birth": float(birth),
                            "death": float(death),
                            "persistence": float(death - birth)
                        })
                    # Sort by persistence
                    pairs.sort(key=lambda x: x["persistence"], reverse=True)
                    persistence_by_dim[f"H{dim}"] = pairs

                # Compute summary statistics
                stats = self._compute_statistics(persistence_by_dim)

                return {
                    "success": True,
                    "persistence": persistence_by_dim,
                    "statistics": stats,
                    "filtration_type": filtration_type,
                    "image_shape": list(img.shape),
                    "max_dimension": max_dimension
                }

            except ImportError:
                # Fallback: use giotto-tda
                try:
                    from gtda.homology import CubicalPersistence

                    # Reshape for giotto-tda (needs batch dimension)
                    img_batch = img.reshape(1, *img.shape)

                    # Compute persistence
                    cp = CubicalPersistence(
                        homology_dimensions=list(range(max_dimension + 1)),
                        coeff=2
                    )
                    diagrams = cp.fit_transform(img_batch)

                    # Parse results
                    persistence_by_dim = {}
                    for dim in range(max_dimension + 1):
                        pairs = []
                        dim_diag = diagrams[0][diagrams[0][:, 2] == dim]
                        for birth, death, _ in dim_diag:
                            if not np.isinf(death):
                                pairs.append({
                                    "birth": float(birth),
                                    "death": float(death),
                                    "persistence": float(death - birth)
                                })
                        pairs.sort(key=lambda x: x["persistence"], reverse=True)
                        persistence_by_dim[f"H{dim}"] = pairs

                    stats = self._compute_statistics(persistence_by_dim)

                    return {
                        "success": True,
                        "persistence": persistence_by_dim,
                        "statistics": stats,
                        "filtration_type": filtration_type,
                        "image_shape": list(img.shape),
                        "max_dimension": max_dimension,
                        "library": "giotto-tda"
                    }

                except ImportError:
                    # Neither library available - compute basic approximation
                    return self._basic_persistence(img, filtration_type, max_dimension)

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _compute_statistics(self, persistence_by_dim: Dict) -> Dict[str, Any]:
        """Compute summary statistics from persistence data.

        Args:
            persistence_by_dim: Persistence pairs by dimension

        Returns:
            Statistics dictionary
        """
        stats = {}
        for dim, pairs in persistence_by_dim.items():
            if pairs:
                persistences = [p["persistence"] for p in pairs]
                stats[dim] = {
                    "count": len(pairs),
                    "total_persistence": float(sum(persistences)),
                    "max_persistence": float(max(persistences)),
                    "mean_persistence": float(np.mean(persistences)),
                    "std_persistence": float(np.std(persistences))
                }
            else:
                stats[dim] = {
                    "count": 0,
                    "total_persistence": 0.0,
                    "max_persistence": 0.0,
                    "mean_persistence": 0.0,
                    "std_persistence": 0.0
                }
        return stats

    def _basic_persistence(
        self,
        img: np.ndarray,
        filtration_type: str,
        max_dimension: int
    ) -> Dict[str, Any]:
        """Basic persistence approximation without TDA libraries.

        Args:
            img: Image array
            filtration_type: Filtration type
            max_dimension: Max dimension

        Returns:
            Approximate persistence data
        """
        from scipy.ndimage import label

        persistence_by_dim = {}

        # H0: Connected components at various thresholds
        thresholds = np.linspace(img.min(), img.max(), 20)
        h0_pairs = []

        for t in thresholds:
            if filtration_type == "sublevel":
                binary = img <= t
            else:
                binary = img >= t
            labeled, num_components = label(binary)
            h0_pairs.append({
                "threshold": float(t),
                "num_components": int(num_components)
            })

        persistence_by_dim["H0"] = h0_pairs

        # Basic H1 estimation using Euler characteristic
        # χ = V - E + F (vertices - edges + faces)
        # For binary images, approximate number of holes
        if max_dimension >= 1:
            mid_threshold = (img.min() + img.max()) / 2
            if filtration_type == "sublevel":
                binary = img <= mid_threshold
            else:
                binary = img >= mid_threshold

            # Estimate holes using Euler characteristic
            labeled, num_components = label(binary)
            labeled_inv, num_holes = label(~binary)

            persistence_by_dim["H1"] = [{
                "note": "Approximate H1 from connected components",
                "num_components": int(num_components),
                "estimated_holes": int(max(0, num_holes - 1))
            }]

        return {
            "success": True,
            "persistence": persistence_by_dim,
            "filtration_type": filtration_type,
            "image_shape": list(img.shape),
            "max_dimension": max_dimension,
            "note": "Approximation - install GUDHI or giotto-tda for full computation"
        }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
