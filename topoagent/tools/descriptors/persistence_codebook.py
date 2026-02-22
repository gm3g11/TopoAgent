"""Persistence Codebook Tool for TopoAgent.

Bag-of-words representation using k-means clustering of persistence points.
Output: K (codebook size, default 50).

IMPORTANT: Codebook requires fitting on training data before transform.
For single-image use without pre-fitting, falls back to per-image fitting
(not ideal but functional for backwards compatibility).
"""

from typing import Any, Dict, List, Optional, Type, Tuple
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Try to import sklearn for MiniBatchKMeans
try:
    from sklearn.cluster import MiniBatchKMeans, KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PersistenceCodebookInput(BaseModel):
    """Input schema for PersistenceCodebookTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool"
    )
    codebook_size: int = Field(
        50, description="Size of the codebook (number of clusters)"
    )
    normalize: bool = Field(
        True, description="Normalize histogram to sum to 1"
    )


class PersistenceCodebookTool(BaseTool):
    """Compute bag-of-words representation from persistence diagrams.

    This tool creates a codebook by clustering persistence diagram
    points (birth, persistence) using k-means, then represents each
    diagram as a histogram of cluster assignments.

    IMPORTANT: For proper use in ML pipelines, call fit() on training
    diagrams first, then use transform() or _run() on test diagrams.
    If used without fitting, falls back to per-image fitting (not ideal).

    Algorithm:
    1. FIT (on training data): Convert all persistence pairs to 2D points
       and apply k-means clustering to create codebook of K centers
    2. TRANSFORM: Assign each point to nearest center, create histogram

    Properties:
    - Fixed-size output (K dimensions)
    - Captures distribution of topological features
    - Similar to bag-of-visual-words in computer vision
    - Useful for classification when feature positions vary

    References:
    - Inspired by bag-of-words models in NLP and computer vision
    - Applicable to TDA for summarizing persistence diagrams
    """

    name: str = "persistence_codebook"
    description: str = (
        "Compute bag-of-words representation from persistence diagrams. "
        "Clusters (birth, persistence) points using k-means. "
        "Returns histogram of cluster assignments. "
        "Input: persistence data from compute_ph tool. "
        "Output: K-dimensional histogram (default K=50). "
        "NOTE: For proper ML use, call fit() on training data first."
    )
    args_schema: Type[BaseModel] = PersistenceCodebookInput

    # Store fitted codebook
    _fitted_centers: Optional[np.ndarray] = None
    _fitted_codebook_size: int = 0
    _is_fitted: bool = False

    def fit(
        self,
        training_data: List[Dict[str, List[Dict[str, float]]]],
        codebook_size: int = 50
    ) -> "PersistenceCodebookTool":
        """Fit codebook on training persistence diagrams.

        MUST be called before transform/run for proper ML pipeline use.

        Args:
            training_data: List of persistence data dicts from compute_ph
            codebook_size: Number of codebook clusters

        Returns:
            self (for chaining)
        """
        self._fitted_codebook_size = codebook_size

        # Collect all points from all dimensions from all training diagrams
        all_points = []
        for data in training_data:
            for dim_key, pairs in data.items():
                points = self._pairs_to_points(pairs)
                if len(points) > 0:
                    all_points.append(points)

        if not all_points:
            # No valid points - create dummy centers
            self._fitted_centers = np.zeros((codebook_size, 2))
            self._is_fitted = True
            return self

        all_points_array = np.vstack(all_points)
        actual_k = min(codebook_size, len(all_points_array))

        # Fit k-means
        self._fitted_centers = self._fit_codebook(all_points_array, actual_k)

        # Pad if needed
        if len(self._fitted_centers) < codebook_size:
            padding = np.zeros((codebook_size - len(self._fitted_centers), 2))
            self._fitted_centers = np.vstack([self._fitted_centers, padding])

        self._is_fitted = True
        return self

    def is_fitted(self) -> bool:
        """Check if codebook has been fitted on training data."""
        return self._is_fitted

    def transform(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        normalize: bool = True
    ) -> np.ndarray:
        """Transform a single persistence diagram using fitted codebook.

        Args:
            persistence_data: Persistence data from compute_ph
            normalize: Whether to normalize histogram

        Returns:
            Combined histogram vector
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Codebook must be fitted on training data first! "
                "Call fit() with training diagrams before transform()."
            )

        # Compute histogram for each dimension
        histograms = []
        for dim_key in sorted(persistence_data.keys()):
            pairs = persistence_data.get(dim_key, [])
            points = self._pairs_to_points(pairs)

            if len(points) > 0:
                hist = self._compute_histogram(
                    points, self._fitted_centers, self._fitted_codebook_size, normalize
                )
            else:
                hist = np.zeros(self._fitted_codebook_size)

            histograms.append(hist)

        return np.concatenate(histograms)

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        codebook_size: int = 50,
        normalize: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Compute persistence codebook.

        If fitted, uses pre-fitted codebook. Otherwise, falls back to
        per-image fitting (with warning).

        Args:
            persistence_data: Persistence pairs by dimension
            codebook_size: Number of clusters K
            normalize: Whether to normalize histogram

        Returns:
            Dictionary with codebook histograms
        """
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
                combined_vector = self.transform(persistence_data, normalize)
                return {
                    "success": True,
                    "tool_name": self.name,
                    "combined_vector": combined_vector.tolist(),
                    "vector_length": len(combined_vector),
                    "codebook_size": self._fitted_codebook_size,
                    "normalized": normalize,
                    "fitted": True,
                    "interpretation": f"Codebook with {self._fitted_codebook_size} centers (pre-fitted)"
                }

            # Fallback: per-image fitting (not ideal)
            # Collect all points across dimensions
            all_points = []
            dim_points = {}

            for dim_key in sorted(persistence_data.keys()):
                pairs = persistence_data[dim_key]
                points = self._pairs_to_points(pairs)
                dim_points[dim_key] = points
                if len(points) > 0:
                    all_points.append(points)

            if not all_points:
                # No valid points
                return {
                    "success": True,
                    "tool_name": self.name,
                    "histograms": {k: np.zeros(codebook_size).tolist() for k in dim_points.keys()},
                    "combined_vector": np.zeros(codebook_size).tolist(),
                    "vector_length": codebook_size,
                    "codebook_size": codebook_size,
                    "fitted": False,
                    "interpretation": "No valid persistence points found"
                }

            # Stack all points and fit codebook
            all_points_array = np.vstack(all_points)
            actual_k = min(codebook_size, len(all_points_array))
            centers = self._fit_codebook(all_points_array, actual_k)

            # Compute histogram for each dimension
            histograms = {}
            for dim_key, points in dim_points.items():
                if len(points) > 0:
                    hist = self._compute_histogram(points, centers, codebook_size, normalize)
                else:
                    hist = np.zeros(codebook_size)
                histograms[dim_key] = hist.tolist()

            # Combined histogram (stacked per-dimension)
            combined_vector = []
            for dim_key in sorted(histograms.keys()):
                combined_vector.extend(histograms[dim_key])

            # Also compute joint histogram (all dimensions together)
            joint_hist = self._compute_histogram(
                all_points_array, centers, codebook_size, normalize
            )

            return {
                "success": True,
                "tool_name": self.name,
                "histograms": histograms,
                "joint_histogram": joint_hist.tolist(),
                "combined_vector": combined_vector,
                "vector_length": len(combined_vector),
                "codebook_size": codebook_size,
                "actual_k": actual_k,
                "n_total_points": len(all_points_array),
                "normalized": normalize,
                "fitted": False,
                "warning": "Codebook not pre-fitted. For proper ML use, call fit() on training data first.",
                "interpretation": self._interpret(histograms, actual_k)
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

            if not np.isfinite(death):
                continue

            pers = death - birth
            if pers > 0:
                points.append([birth, pers])

        return np.array(points) if points else np.zeros((0, 2))

    def _fit_codebook(self, points: np.ndarray, k: int) -> np.ndarray:
        """Fit k-means codebook.

        Args:
            points: (N, 2) array of points
            k: Number of clusters

        Returns:
            (k, 2) array of cluster centers
        """
        if HAS_SKLEARN:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                batch_size=min(1024, len(points)),
                n_init=3
            )
            kmeans.fit(points)
            return kmeans.cluster_centers_
        else:
            # Fallback to simple k-means
            return self._simple_kmeans(points, k)

    def _simple_kmeans(self, points: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple k-means when sklearn unavailable."""
        n_points = len(points)

        # Initialize
        rng = np.random.default_rng(42)
        idx = rng.choice(n_points, size=min(k, n_points), replace=False)
        centers = points[idx].copy()

        if len(centers) < k:
            padding = np.zeros((k - len(centers), 2))
            centers = np.vstack([centers, padding])

        for _ in range(max_iter):
            # Assign
            dists = np.zeros((n_points, k))
            for i, c in enumerate(centers):
                dists[:, i] = np.linalg.norm(points - c, axis=1)
            assignments = np.argmin(dists, axis=1)

            # Update
            new_centers = np.zeros_like(centers)
            for i in range(k):
                mask = assignments == i
                if np.sum(mask) > 0:
                    new_centers[i] = np.mean(points[mask], axis=0)
                else:
                    new_centers[i] = centers[i]

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        return centers

    def _compute_histogram(
        self,
        points: np.ndarray,
        centers: np.ndarray,
        k: int,
        normalize: bool
    ) -> np.ndarray:
        """Compute histogram of cluster assignments.

        Args:
            points: (N, 2) array of points
            centers: (k', 2) array of centers (k' <= k)
            k: Full codebook size (for output shape)
            normalize: Whether to normalize

        Returns:
            Histogram of length k
        """
        histogram = np.zeros(k)

        if len(points) == 0 or len(centers) == 0:
            return histogram

        # Vectorized assignment: compute all pairwise distances at once
        # dists shape: (n_points, n_centers)
        diff = points[:, None, :] - centers[None, :, :]  # (N, k', 2)
        dists = np.linalg.norm(diff, axis=2)              # (N, k')
        nearest = np.argmin(dists, axis=1)                 # (N,)
        valid = nearest < k
        counts = np.bincount(nearest[valid], minlength=k)
        histogram[:len(counts)] = counts[:k]

        if normalize:
            total = np.sum(histogram)
            if total > 0:
                histogram = histogram / total

        return histogram

    def _interpret(self, histograms: Dict, actual_k: int) -> str:
        """Generate interpretation."""
        parts = []

        for dim_key in sorted(histograms.keys()):
            hist = np.array(histograms[dim_key])
            if np.sum(hist) > 0:
                top_clusters = np.argsort(-hist)[:3]
                parts.append(
                    f"{dim_key}: dominant clusters={list(top_clusters)}, "
                    f"max weight={np.max(hist):.3f}"
                )
            else:
                parts.append(f"{dim_key}: empty histogram")

        return ". ".join(parts)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
