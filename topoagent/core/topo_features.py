"""TopoAgent Core: Deterministic Topology Feature Extraction.

This module provides a deterministic, auditable API for topology feature
extraction that does NOT depend on an LLM. This is the "trustworthy" core
that can be validated independently.

Key Design Principles:
1. Deterministic: Same input → Same output (no LLM randomness)
2. Auditable: Full metadata trail for every computation
3. Validated: QC checks for common failure modes
4. Reproducible: All parameters and library versions logged

Example:
    >>> from topoagent.core import extract_topo_features, TopoConfig
    >>> config = TopoConfig(filtration_type="sublevel", pi_resolution=20)
    >>> result = extract_topo_features(image_array, config)
    >>> print(result.vector)  # 800D feature vector
    >>> print(result.metadata)  # Full audit trail
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np


@dataclass
class TopoConfig:
    """Configuration for topology feature extraction.

    All parameters are explicit for reproducibility.
    """
    # Filtration parameters
    filtration_type: str = "sublevel"  # "sublevel" or "superlevel"
    max_dimension: int = 1  # Max homology dimension (0, 1, or 2)

    # Persistence Image parameters
    pi_resolution: int = 20  # NxN grid resolution
    pi_sigma: float = 0.1  # Gaussian kernel bandwidth
    pi_weight_fn: str = "linear"  # "linear", "squared", or "const"

    # Preprocessing
    normalize: bool = True  # Normalize image to [0, 1]
    grayscale: bool = True  # Convert to grayscale if color

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "filtration_type": self.filtration_type,
            "max_dimension": self.max_dimension,
            "pi_resolution": self.pi_resolution,
            "pi_sigma": self.pi_sigma,
            "pi_weight_fn": self.pi_weight_fn,
            "normalize": self.normalize,
            "grayscale": self.grayscale,
        }


@dataclass
class TopoFeatureResult:
    """Result from topology feature extraction.

    Contains the feature vector plus full audit trail.
    """
    # Primary output
    vector: np.ndarray  # Combined feature vector (e.g., 800D for 20x20 PI)
    success: bool = True
    error: Optional[str] = None

    # Metadata for audit trail
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Quality control flags
    qc: Dict[str, Any] = field(default_factory=dict)

    # Human-readable summary
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "vector": self.vector.tolist() if self.vector is not None else None,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "qc": self.qc,
            "summary": self.summary,
        }


def extract_topo_features(
    image: Union[np.ndarray, List],
    config: Optional[TopoConfig] = None
) -> TopoFeatureResult:
    """Extract topology features from an image.

    This is the main deterministic API. No LLM is involved.

    Args:
        image: 2D image array (grayscale) or 3D (color, will be converted)
        config: Configuration parameters (uses defaults if None)

    Returns:
        TopoFeatureResult with feature vector, metadata, QC flags, and summary

    Example:
        >>> result = extract_topo_features(image)
        >>> classifier.predict(result.vector)
    """
    start_time = time.time()
    config = config or TopoConfig()

    # Initialize result with empty vector
    result = TopoFeatureResult(
        vector=np.array([]),
        metadata={
            "config": config.to_dict(),
            "library_versions": _get_library_versions(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        qc={},
        summary={},
    )

    try:
        # Step 1: Preprocess image
        img = _preprocess_image(image, config)
        result.metadata["image_shape"] = list(img.shape)
        result.metadata["image_stats"] = {
            "min": float(img.min()),
            "max": float(img.max()),
            "mean": float(img.mean()),
            "std": float(img.std()),
        }

        # Step 2: Compute persistent homology
        persistence_data, ph_metadata = _compute_persistent_homology(img, config)
        result.metadata["ph"] = ph_metadata

        # QC: Check for empty diagrams
        h0_count = len(persistence_data.get("H0", []))
        h1_count = len(persistence_data.get("H1", []))
        result.qc["diagram_empty"] = (h0_count == 0 and h1_count == 0)
        result.qc["h0_count"] = h0_count
        result.qc["h1_count"] = h1_count

        # Step 3: Compute persistence image
        vector, pi_metadata = _compute_persistence_image(persistence_data, config)
        result.vector = vector
        result.metadata["pi"] = pi_metadata

        # QC: Check for degenerate features
        nonzero_ratio = np.count_nonzero(vector) / len(vector) if len(vector) > 0 else 0
        result.qc["nonzero_ratio"] = nonzero_ratio
        result.qc["low_persistence_warning"] = nonzero_ratio < 0.01

        # Compute summary statistics
        result.summary = {
            "h0_count": h0_count,
            "h1_count": h1_count,
            "vector_dim": len(vector),
            "nonzero_ratio": f"{nonzero_ratio:.1%}",
            "filtration_type": config.filtration_type,
        }

        # Add dominant scale if we have persistence data
        all_persistences = []
        for dim_key in ["H0", "H1"]:
            for pair in persistence_data.get(dim_key, []):
                if "persistence" in pair:
                    all_persistences.append(pair["persistence"])
        if all_persistences:
            result.summary["dominant_scale"] = float(max(all_persistences))
            result.summary["mean_persistence"] = float(np.mean(all_persistences))

    except Exception as e:
        result.success = False
        result.error = str(e)
        result.vector = np.zeros(config.pi_resolution ** 2 * 2)  # Default size

    # Record runtime
    result.metadata["runtime_ms"] = (time.time() - start_time) * 1000

    return result


def _preprocess_image(
    image: Union[np.ndarray, List],
    config: TopoConfig
) -> np.ndarray:
    """Preprocess image for topology computation.

    Args:
        image: Input image (array or list)
        config: Configuration

    Returns:
        Preprocessed numpy array
    """
    # Convert to numpy
    img = np.array(image, dtype=np.float32)

    # Convert to grayscale if needed
    if config.grayscale and len(img.shape) == 3:
        img = np.mean(img, axis=2)

    # Normalize to [0, 1]
    if config.normalize:
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img)

    return img


def _compute_persistent_homology(
    img: np.ndarray,
    config: TopoConfig
) -> Tuple[Dict[str, List], Dict[str, Any]]:
    """Compute persistent homology using GUDHI or fallback.

    Args:
        img: Preprocessed image
        config: Configuration

    Returns:
        Tuple of (persistence_data, metadata)
    """
    metadata = {
        "filtration_type": config.filtration_type,
        "max_dimension": config.max_dimension,
    }

    # Apply filtration transform
    if config.filtration_type == "superlevel":
        # Use monotone transform (1.0 - img) to avoid abs() bug
        img = 1.0 - img

    # Try GUDHI first
    try:
        import gudhi
        metadata["library"] = f"gudhi=={gudhi.__version__}"

        cubical = gudhi.CubicalComplex(
            dimensions=list(img.shape),
            top_dimensional_cells=img.flatten()
        )
        cubical.compute_persistence()
        persistence = cubical.persistence_intervals_in_dimension

        persistence_by_dim = {}
        for dim in range(config.max_dimension + 1):
            intervals = persistence(dim)
            pairs = []
            for birth, death in intervals:
                if death == float('inf'):
                    death = img.max()
                # No abs() needed with monotone transform
                pairs.append({
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth)
                })
            pairs.sort(key=lambda x: x["persistence"], reverse=True)
            persistence_by_dim[f"H{dim}"] = pairs

        return persistence_by_dim, metadata

    except ImportError:
        pass

    # Try giotto-tda fallback
    try:
        from gtda.homology import CubicalPersistence
        import gtda
        metadata["library"] = f"giotto-tda=={gtda.__version__}"

        img_batch = img.reshape(1, *img.shape)
        cp = CubicalPersistence(
            homology_dimensions=list(range(config.max_dimension + 1)),
            coeff=2
        )
        diagrams = cp.fit_transform(img_batch)

        persistence_by_dim = {}
        for dim in range(config.max_dimension + 1):
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

        return persistence_by_dim, metadata

    except ImportError:
        pass

    # Basic fallback using scipy
    from scipy.ndimage import label
    metadata["library"] = "scipy (approximate)"

    persistence_by_dim = {"H0": [], "H1": []}

    # Approximate H0 from connected components
    thresholds = np.linspace(img.min(), img.max(), 20)
    for i, t in enumerate(thresholds[:-1]):
        binary = img <= t
        _, num_components = label(binary)
        if num_components > 0:
            persistence_by_dim["H0"].append({
                "birth": float(t),
                "death": float(thresholds[i + 1]),
                "persistence": float(thresholds[i + 1] - t)
            })

    return persistence_by_dim, metadata


def _compute_persistence_image(
    persistence_data: Dict[str, List],
    config: TopoConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute persistence image vectorization.

    Args:
        persistence_data: Persistence pairs by dimension
        config: Configuration

    Returns:
        Tuple of (feature_vector, metadata)
    """
    resolution = config.pi_resolution
    sigma = config.pi_sigma
    weight_fn = config.pi_weight_fn

    metadata = {
        "resolution": resolution,
        "sigma": sigma,
        "weight_fn": weight_fn,
    }

    feature_vectors = {}

    for dim_key, pairs in persistence_data.items():
        if not pairs:
            feature_vectors[dim_key] = np.zeros(resolution * resolution)
            continue

        # Convert to birth/persistence coordinates
        points = []
        for p in pairs:
            birth = p["birth"]
            persistence = p["persistence"]
            if persistence > 0:
                points.append((birth, persistence))

        if not points:
            feature_vectors[dim_key] = np.zeros(resolution * resolution)
            continue

        # Compute persistence image
        pi = _compute_pi_grid(points, resolution, sigma, weight_fn)
        feature_vectors[dim_key] = pi.flatten()

    # Combine dimensions
    combined = []
    for dim in sorted(feature_vectors.keys()):
        combined.extend(feature_vectors[dim])

    metadata["dimensions_processed"] = list(feature_vectors.keys())
    metadata["vector_length"] = len(combined)

    return np.array(combined), metadata


def _compute_pi_grid(
    points: List[Tuple[float, float]],
    resolution: int,
    sigma: float,
    weight_fn: str
) -> np.ndarray:
    """Compute persistence image grid.

    Args:
        points: List of (birth, persistence) tuples
        resolution: Grid resolution
        sigma: Gaussian bandwidth
        weight_fn: Weight function type

    Returns:
        2D persistence image array
    """
    if not points:
        return np.zeros((resolution, resolution))

    # Get data range
    births = [p[0] for p in points]
    persistences = [p[1] for p in points]

    birth_min, birth_max = min(births), max(births)
    pers_max = max(persistences)

    # Add padding
    birth_range = birth_max - birth_min if birth_max > birth_min else 1
    pers_range = pers_max if pers_max > 0 else 1

    # Create grid
    birth_bins = np.linspace(
        birth_min - 0.1 * birth_range,
        birth_max + 0.1 * birth_range,
        resolution
    )
    pers_bins = np.linspace(0, pers_max + 0.1 * pers_range, resolution)

    # Initialize image
    image = np.zeros((resolution, resolution))

    # Add Gaussian for each point
    for birth, persistence in points:
        # Weight based on persistence
        if weight_fn == "linear":
            weight = persistence
        elif weight_fn == "squared":
            weight = persistence ** 2
        else:  # const
            weight = 1.0

        # Add weighted Gaussian (vectorized for efficiency)
        for i, b in enumerate(birth_bins):
            for j, p in enumerate(pers_bins):
                dist_sq = (b - birth) ** 2 + (p - persistence) ** 2
                gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
                image[j, i] += weight * gaussian

    # Normalize
    if image.max() > 0:
        image = image / image.max()

    return image


def _get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries for reproducibility."""
    versions = {}

    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass

    try:
        import gudhi
        versions["gudhi"] = gudhi.__version__
    except ImportError:
        versions["gudhi"] = "not installed"

    try:
        import gtda
        versions["giotto-tda"] = gtda.__version__
    except ImportError:
        versions["giotto-tda"] = "not installed"

    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass

    return versions


def validate_synthetic(
    shape_name: str,
    expected_h0: int,
    expected_h1: int,
    tolerance: int = 1
) -> Dict[str, Any]:
    """Validate persistence computation on synthetic shapes.

    For correctness testing - verifies Betti numbers match expectations.

    Args:
        shape_name: One of "single_disk", "two_disks", "annulus", "nested_rings"
        expected_h0: Expected H0 Betti number
        expected_h1: Expected H1 Betti number
        tolerance: Allowed deviation

    Returns:
        Validation result dictionary
    """
    # Generate synthetic shape
    if shape_name == "single_disk":
        img = _create_disk(28, 28, center=(14, 14), radius=8)
    elif shape_name == "two_disks":
        img = _create_disk(28, 28, center=(8, 14), radius=5)
        img += _create_disk(28, 28, center=(20, 14), radius=5)
    elif shape_name == "annulus":
        outer = _create_disk(28, 28, center=(14, 14), radius=10)
        inner = _create_disk(28, 28, center=(14, 14), radius=5)
        img = outer - inner
    elif shape_name == "nested_rings":
        outer = _create_disk(28, 28, center=(14, 14), radius=12)
        mid = _create_disk(28, 28, center=(14, 14), radius=8)
        inner = _create_disk(28, 28, center=(14, 14), radius=4)
        img = outer - mid + inner
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

    # Extract features
    config = TopoConfig(filtration_type="sublevel", max_dimension=1)
    result = extract_topo_features(img, config)

    # Compare
    actual_h0 = result.qc.get("h0_count", 0)
    actual_h1 = result.qc.get("h1_count", 0)

    h0_match = abs(actual_h0 - expected_h0) <= tolerance
    h1_match = abs(actual_h1 - expected_h1) <= tolerance

    return {
        "shape_name": shape_name,
        "expected": {"h0": expected_h0, "h1": expected_h1},
        "actual": {"h0": actual_h0, "h1": actual_h1},
        "h0_match": h0_match,
        "h1_match": h1_match,
        "passed": h0_match and h1_match,
    }


def _create_disk(height: int, width: int, center: Tuple[int, int], radius: float) -> np.ndarray:
    """Create a binary disk image."""
    y, x = np.ogrid[:height, :width]
    cy, cx = center
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return (r <= radius).astype(np.float32)
