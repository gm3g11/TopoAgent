"""Fast PH Computation Utilities.

Optimized persistent homology computation with caching and parallelization.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path
import hashlib
import pickle

# Try to import fast PH libraries
try:
    import cripser
    HAS_CRIPSER = True
except ImportError:
    HAS_CRIPSER = False

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

try:
    from gtda.homology import CubicalPersistence
    HAS_GIOTTO = True
except ImportError:
    HAS_GIOTTO = False


class PHCache:
    """Cache for persistent homology computations.

    Supports both in-memory and disk-based caching.
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_memory_items: int = 1000):
        """Initialize PH cache.

        Args:
            cache_dir: Directory for disk cache (None = memory only)
            max_memory_items: Maximum items in memory cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self._memory_cache: Dict[str, Dict] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, image: np.ndarray, filtration: str) -> str:
        """Compute cache key for image + filtration."""
        # Use hash of image data + filtration
        img_bytes = image.tobytes()
        key_data = img_bytes + filtration.encode()
        return hashlib.md5(key_data).hexdigest()

    def get(self, image: np.ndarray, filtration: str) -> Optional[Dict]:
        """Get cached PH result."""
        key = self._compute_key(image, filtration)

        # Check memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                # Also store in memory
                self._add_to_memory(key, result)
                return result

        return None

    def put(self, image: np.ndarray, filtration: str, result: Dict):
        """Store PH result in cache."""
        key = self._compute_key(image, filtration)

        # Add to memory cache
        self._add_to_memory(key, result)

        # Add to disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

    def _add_to_memory(self, key: str, result: Dict):
        """Add to memory cache with eviction."""
        if len(self._memory_cache) >= self.max_memory_items:
            # Simple eviction: remove first item (FIFO)
            first_key = next(iter(self._memory_cache))
            del self._memory_cache[first_key]

        self._memory_cache[key] = result

    def clear(self):
        """Clear all caches."""
        self._memory_cache.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()

    def __len__(self) -> int:
        """Number of items in memory cache."""
        return len(self._memory_cache)


def compute_ph_fast(
    image: np.ndarray,
    filtration: str = "sublevel",
    max_dimension: int = 1,
    cache: Optional[PHCache] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """Compute persistent homology using fastest available library.

    Tries libraries in order: cripser > GUDHI > giotto-tda > fallback

    Args:
        image: 2D grayscale image [0, 1]
        filtration: 'sublevel' or 'superlevel'
        max_dimension: Maximum homology dimension
        cache: Optional PHCache for caching results

    Returns:
        Dictionary with H0, H1, ... keys containing persistence pairs
    """
    # Check cache first
    if cache is not None:
        cached = cache.get(image, filtration)
        if cached is not None:
            return cached

    # Normalize image
    if image.max() > 1.0:
        image = image.astype(np.float64) / 255.0
    else:
        image = image.astype(np.float64)

    # Compute PH using best available library
    if HAS_CRIPSER:
        result = _compute_ph_cripser(image, filtration, max_dimension)
    elif HAS_GUDHI:
        result = _compute_ph_gudhi(image, filtration, max_dimension)
    elif HAS_GIOTTO:
        result = _compute_ph_giotto(image, filtration, max_dimension)
    else:
        result = _compute_ph_fallback(image, filtration, max_dimension)

    # Store in cache
    if cache is not None:
        cache.put(image, filtration, result)

    return result


def _compute_ph_cripser(
    image: np.ndarray,
    filtration: str,
    max_dimension: int
) -> Dict[str, List[Dict[str, float]]]:
    """Compute PH using cripser (fastest for images)."""
    # Prepare image for cripser
    if filtration == "superlevel":
        img = 1.0 - image
    else:
        img = image

    # Compute persistence
    # cripser expects (height, width) image
    result = cripser.computePH(img.astype(np.float64), maxdim=max_dimension)

    # Parse results
    # cripser returns: [(dim, birth, death), ...] format
    # dim is FIRST, then birth, then death
    persistence_by_dim = {}
    for dim in range(max_dimension + 1):
        pairs = []
        for entry in result:
            if len(entry) >= 3:
                dim_val = int(entry[0])  # Dimension is FIRST
                birth = float(entry[1])   # Birth is SECOND
                death = float(entry[2])   # Death is THIRD
                if dim_val == dim and np.isfinite(death) and death > birth:
                    pairs.append({
                        "birth": birth,
                        "death": death,
                        "persistence": death - birth
                    })
        pairs.sort(key=lambda x: -x["persistence"])
        persistence_by_dim[f"H{dim}"] = pairs

    return persistence_by_dim


def _compute_ph_gudhi(
    image: np.ndarray,
    filtration: str,
    max_dimension: int
) -> Dict[str, List[Dict[str, float]]]:
    """Compute PH using GUDHI."""
    if filtration == "superlevel":
        img = 1.0 - image
    else:
        img = image

    # Create cubical complex
    cubical = gudhi.CubicalComplex(
        dimensions=list(img.shape),
        top_dimensional_cells=img.flatten()
    )

    # Compute persistence
    cubical.compute_persistence()

    # Parse results
    persistence_by_dim = {}
    for dim in range(max_dimension + 1):
        intervals = cubical.persistence_intervals_in_dimension(dim)
        pairs = []
        for birth, death in intervals:
            if death == float('inf'):
                death = img.max()
            if death > birth:
                pairs.append({
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth)
                })
        pairs.sort(key=lambda x: -x["persistence"])
        persistence_by_dim[f"H{dim}"] = pairs

    return persistence_by_dim


def _compute_ph_giotto(
    image: np.ndarray,
    filtration: str,
    max_dimension: int
) -> Dict[str, List[Dict[str, float]]]:
    """Compute PH using giotto-tda."""
    if filtration == "superlevel":
        img = 1.0 - image
    else:
        img = image

    # Reshape for giotto-tda
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
            if not np.isinf(death) and death > birth:
                pairs.append({
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth)
                })
        pairs.sort(key=lambda x: -x["persistence"])
        persistence_by_dim[f"H{dim}"] = pairs

    return persistence_by_dim


def _compute_ph_fallback(
    image: np.ndarray,
    filtration: str,
    max_dimension: int
) -> Dict[str, List[Dict[str, float]]]:
    """Simple fallback PH computation using scipy."""
    from scipy.ndimage import label

    if filtration == "superlevel":
        img = 1.0 - image
    else:
        img = image

    # Very basic approximation using thresholding
    thresholds = np.linspace(img.min(), img.max(), 50)

    # Track component births/deaths
    prev_components = {}
    h0_pairs = []

    for t in thresholds:
        binary = img <= t
        labeled, n_components = label(binary)

        # Simple birth/death tracking
        curr_components = set(range(1, n_components + 1))

        # New components (births)
        for c in curr_components:
            if c not in prev_components:
                prev_components[c] = t

        # Merged components (deaths)
        if n_components < len(prev_components):
            # Some components merged
            deaths_needed = len(prev_components) - n_components
            oldest = sorted(prev_components.items(), key=lambda x: x[1])[:deaths_needed]
            for c, birth in oldest:
                h0_pairs.append({
                    "birth": float(birth),
                    "death": float(t),
                    "persistence": float(t - birth)
                })
                del prev_components[c]

    # Remaining components live forever
    for c, birth in prev_components.items():
        h0_pairs.append({
            "birth": float(birth),
            "death": float(img.max()),
            "persistence": float(img.max() - birth)
        })

    h0_pairs.sort(key=lambda x: -x["persistence"])

    result = {"H0": h0_pairs}

    # Basic H1 (empty for fallback)
    if max_dimension >= 1:
        result["H1"] = []

    return result


def compute_ph_batch(
    images: np.ndarray,
    filtration: str = "sublevel",
    max_dimension: int = 1,
    n_jobs: int = 1,
    cache: Optional[PHCache] = None,
) -> List[Dict[str, List[Dict[str, float]]]]:
    """Compute PH for multiple images.

    Args:
        images: (N, H, W) array of images
        filtration: Filtration type
        max_dimension: Max homology dimension
        n_jobs: Number of parallel jobs
        cache: Optional cache

    Returns:
        List of persistence dictionaries
    """
    if n_jobs == 1:
        return [
            compute_ph_fast(img, filtration, max_dimension, cache)
            for img in images
        ]
    else:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_ph_fast)(img, filtration, max_dimension, None)
                for img in images
            )
            return results
        except ImportError:
            return compute_ph_batch(images, filtration, max_dimension, n_jobs=1, cache=cache)
