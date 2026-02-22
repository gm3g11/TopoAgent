"""ATOL (Automatic Topologically-Oriented Learning) Tool - Benchmark3 fixed copy.

Fix: In _compute_numpy_atol, changed sigma = np.std(points) to
sigma = np.mean(np.std(points, axis=0)) to use mean of per-dimension std
rather than flattened scalar std for 2D arrays.
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import GUDHI's ATOL implementation
try:
    from gudhi.representations import Atol
    HAS_GUDHI_ATOL = True
except ImportError:
    HAS_GUDHI_ATOL = False


class ATOLInput(BaseModel):
    """Input schema for ATOLTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    n_centers: int = Field(
        4, description="Number of cluster centers for ATOL"
    )
    quantiser: str = Field(
        "KMeans", description="Quantiser type: 'KMeans' or 'MiniBatchKMeans'"
    )


class ATOLTool(BaseTool):
    """Compute ATOL vectorization from persistence diagrams.

    ATOL (Automatic Topologically-Oriented Learning) represents a
    persistence diagram using distances to learned cluster centers.

    IMPORTANT: For proper use in ML pipelines, call fit() on training
    diagrams first, then use transform() or _run() on test diagrams.
    If used without fitting, falls back to per-image fitting (not ideal).

    References:
    - Royer et al., "ATOL: Measure Vectorization for Automatic
      Topologically-Oriented Learning" (2019)
    """

    name: str = "atol"
    description: str = (
        "Compute ATOL vectorization from persistence diagrams. "
        "Uses k-means clustering to learn representative centers. "
        "Each diagram is represented by weighted distances to centers. "
        "Input: persistence data from compute_ph tool. "
        "Output: n_centers values per homology dimension (default 4, giving 8D total). "
        "NOTE: For proper ML use, call fit() on training data first."
    )
    args_schema: Type[BaseModel] = ATOLInput

    # Store fitted ATOL models per dimension
    _fitted_models: Dict[str, Any] = {}
    _fitted_n_centers: int = 0
    _is_fitted: bool = False

    def fit(
        self,
        training_data: List[Dict[str, List[Dict[str, float]]]],
        n_centers: int = 4,
        quantiser: str = "KMeans"
    ) -> "ATOLTool":
        """Fit ATOL on training persistence diagrams."""
        from sklearn.cluster import KMeans, MiniBatchKMeans

        self._fitted_models = {}
        self._fitted_n_centers = n_centers

        # Collect all dimension keys
        all_dims = set()
        for data in training_data:
            all_dims.update(data.keys())

        # Fit separate ATOL for each dimension
        for dim_key in sorted(all_dims):
            all_points = []
            for data in training_data:
                if dim_key in data:
                    points = self._pairs_to_points(data[dim_key])
                    if len(points) > 0:
                        all_points.append(points)

            if not all_points:
                self._fitted_models[dim_key] = None
                continue

            all_points_array = np.vstack(all_points)

            # Filter out any remaining NaN/inf values
            valid_mask = np.all(np.isfinite(all_points_array), axis=1)
            all_points_array = all_points_array[valid_mask]

            if len(all_points_array) == 0:
                self._fitted_models[dim_key] = None
                continue

            # Create quantiser
            actual_k = min(n_centers, len(all_points_array))
            if quantiser == "MiniBatchKMeans":
                quantiser_obj = MiniBatchKMeans(
                    n_clusters=actual_k, random_state=42, n_init=3
                )
            else:
                quantiser_obj = KMeans(
                    n_clusters=actual_k, random_state=42, n_init=3
                )

            if HAS_GUDHI_ATOL:
                atol = Atol(quantiser=quantiser_obj)
                atol.fit([all_points_array])
                self._fitted_models[dim_key] = atol
            else:
                # Store fitted k-means for numpy fallback
                quantiser_obj.fit(all_points_array)
                # FIX: Use per-dimension std instead of flattened scalar std
                sigma_val = np.mean(np.std(all_points_array, axis=0))
                self._fitted_models[dim_key] = {
                    'centers': quantiser_obj.cluster_centers_,
                    'sigma': sigma_val if sigma_val > 1e-10 else 1.0
                }

        self._is_fitted = True
        return self

    def is_fitted(self) -> bool:
        """Check if ATOL has been fitted on training data."""
        return self._is_fitted

    def transform(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]]
    ) -> np.ndarray:
        """Transform a single persistence diagram using fitted ATOL."""
        if not self._is_fitted:
            raise RuntimeError(
                "ATOL must be fitted on training data first! "
                "Call fit() with training diagrams before transform()."
            )

        all_vectors = []
        for dim_key in sorted(self._fitted_models.keys()):
            points = self._pairs_to_points(persistence_data.get(dim_key, []))

            if self._fitted_models[dim_key] is None:
                all_vectors.append(np.zeros(self._fitted_n_centers))
                continue

            if len(points) == 0:
                all_vectors.append(np.zeros(self._fitted_n_centers))
                continue

            if HAS_GUDHI_ATOL:
                atol = self._fitted_models[dim_key]
                vector = atol.transform([points])[0]
            else:
                model = self._fitted_models[dim_key]
                vector = self._compute_atol_vector(
                    points, model['centers'], model['sigma']
                )

            # Pad if needed
            if len(vector) < self._fitted_n_centers:
                vector = np.pad(vector, (0, self._fitted_n_centers - len(vector)))

            all_vectors.append(vector)

        return np.concatenate(all_vectors)

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        n_centers: int = 4,
        quantiser: str = "KMeans",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute ATOL vectorization."""
        try:
            # Validate input
            if not isinstance(persistence_data, dict):
                return {
                    "success": False,
                    "tool_name": self.name,
                    "error": f"Expected dict, got {type(persistence_data)}"
                }

            # If fitted, use transform
            if self._is_fitted:
                combined_vector = self.transform(persistence_data)
                return {
                    "success": True,
                    "tool_name": self.name,
                    "combined_vector": combined_vector.tolist(),
                    "vector_length": len(combined_vector),
                    "n_centers": self._fitted_n_centers,
                    "using_gudhi": HAS_GUDHI_ATOL,
                    "fitted": True,
                    "interpretation": f"ATOL with {self._fitted_n_centers} centers (pre-fitted)"
                }

            # Fallback: per-image fitting (not ideal, but functional)
            all_vectors = {}

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]
                points = self._pairs_to_points(pairs)

                if len(points) == 0:
                    all_vectors[dim_key] = {
                        'vector': np.zeros(n_centers).tolist(),
                        'n_points': 0
                    }
                    continue

                # Compute ATOL vectorization (per-image fitting)
                if HAS_GUDHI_ATOL:
                    vector = self._compute_gudhi_atol(points, n_centers, quantiser)
                else:
                    vector = self._compute_numpy_atol(points, n_centers)

                all_vectors[dim_key] = {
                    'vector': vector.tolist(),
                    'n_points': len(points)
                }

            # Create combined vector
            combined_vector = []
            for dim_key in sorted(all_vectors.keys()):
                combined_vector.extend(all_vectors[dim_key]['vector'])

            return {
                "success": True,
                "tool_name": self.name,
                "atol_vectors": all_vectors,
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "n_centers": n_centers,
                "using_gudhi": HAS_GUDHI_ATOL,
                "fitted": False,
                "warning": "ATOL not pre-fitted. For proper ML use, call fit() on training data first.",
                "interpretation": self._interpret(all_vectors)
            }

        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    def _pairs_to_points(self, pairs: List[Dict]) -> np.ndarray:
        """Convert persistence pairs to (birth, persistence) points."""
        points = []
        for p in pairs:
            if not isinstance(p, dict) or "birth" not in p or "death" not in p:
                continue

            birth = p["birth"]
            death = p["death"]

            # Skip non-finite values (infinity, NaN)
            if not np.isfinite(birth) or not np.isfinite(death):
                continue

            pers = death - birth
            if pers > 0 and np.isfinite(pers):
                points.append([birth, pers])

        return np.array(points) if points else np.zeros((0, 2))

    def _compute_gudhi_atol(
        self,
        points: np.ndarray,
        n_centers: int,
        quantiser: str
    ) -> np.ndarray:
        """Compute ATOL using GUDHI (per-image fallback)."""
        from sklearn.cluster import KMeans, MiniBatchKMeans

        actual_k = min(n_centers, len(points))

        if quantiser == "MiniBatchKMeans":
            quantiser_obj = MiniBatchKMeans(n_clusters=actual_k, random_state=42, n_init=3)
        else:
            quantiser_obj = KMeans(n_clusters=actual_k, random_state=42, n_init=3)

        atol = Atol(quantiser=quantiser_obj)
        atol.fit([points])
        vector = atol.transform([points])[0]

        # Pad if needed
        if len(vector) < n_centers:
            vector = np.pad(vector, (0, n_centers - len(vector)))

        return vector

    def _compute_numpy_atol(
        self,
        points: np.ndarray,
        n_centers: int
    ) -> np.ndarray:
        """Compute ATOL using numpy k-means (per-image fallback)."""
        n_points = len(points)

        if n_points < n_centers:
            vector = np.zeros(n_centers)
            for i, p in enumerate(points):
                vector[i] = np.sum(p ** 2)
            return vector

        centers = self._simple_kmeans(points, n_centers)
        # FIX: Use mean of per-dimension std instead of flattened scalar std
        sigma = np.mean(np.std(points, axis=0))
        if sigma < 1e-10:
            sigma = 1.0

        return self._compute_atol_vector(points, centers, sigma)

    def _compute_atol_vector(
        self,
        points: np.ndarray,
        centers: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """Compute ATOL vector given fitted centers (vectorized)."""
        # Vectorized: compute all distances at once
        diff = points[:, None, :] - centers[None, :, :]  # (n_points, n_centers, 2)
        dists = np.linalg.norm(diff, axis=2)  # (n_points, n_centers)
        weights = points[:, 1:2]  # (n_points, 1) — persistence column
        gaussians = np.exp(-dists ** 2 / (2 * sigma ** 2))  # (n_points, n_centers)
        vector = np.sum(weights * gaussians, axis=0)  # (n_centers,)
        return vector

    def _simple_kmeans(
        self,
        points: np.ndarray,
        n_centers: int,
        max_iter: int = 100
    ) -> np.ndarray:
        """Simple k-means clustering."""
        n_points = len(points)

        rng = np.random.default_rng(42)
        idx = rng.choice(n_points, size=min(n_centers, n_points), replace=False)
        centers = points[idx].copy()

        if len(centers) < n_centers:
            padding = np.zeros((n_centers - len(centers), 2))
            centers = np.vstack([centers, padding])

        for _ in range(max_iter):
            dists = np.zeros((n_points, n_centers))
            for i, c in enumerate(centers):
                dists[:, i] = np.linalg.norm(points - c, axis=1)

            assignments = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for i in range(n_centers):
                mask = assignments == i
                if np.sum(mask) > 0:
                    new_centers[i] = np.mean(points[mask], axis=0)
                else:
                    new_centers[i] = centers[i]

            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        return centers

    def _interpret(self, all_vectors: Dict) -> str:
        """Generate interpretation of ATOL vectors."""
        parts = []

        for dim_key in sorted(all_vectors.keys()):
            vec = all_vectors[dim_key]
            n_points = vec['n_points']
            vector = np.array(vec['vector'])

            if n_points > 0:
                dom_idx = np.argmax(vector)
                parts.append(
                    f"{dim_key}: {n_points} points, "
                    f"dominant center={dom_idx}, max weight={np.max(vector):.4f}"
                )
            else:
                parts.append(f"{dim_key}: no valid points")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
