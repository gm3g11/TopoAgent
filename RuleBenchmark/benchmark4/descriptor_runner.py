"""
Benchmark4: Unified Descriptor Extraction.

Extracts feature vectors for all 15 descriptors using optimal params from exp4.
Handles both PH-based and image-based descriptors, plus per-channel RGB mode.

Usage:
    from RuleBenchmark.benchmark4.descriptor_runner import extract_features

    # PH-based descriptor
    features = extract_features('persistence_image', diagrams=diags, params=params, expected_dim=200)

    # Image-based descriptor
    features = extract_features('minkowski_functionals', images=images, params=params, expected_dim=30)

    # Per-channel mode (concatenates R, G, B features)
    features = extract_features_per_channel(
        'persistence_image', diags_R=dr, diags_G=dg, diags_B=db,
        params=params, expected_dim_per_channel=200)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# diagrams_to_dict_list removed — replaced by per-image _convert_one_diagram()
# to avoid OOM on large PH caches (e.g. ISIC2019 12GB → 140GB as Python dicts)


# =============================================================================
# Tool Imports (bug-fixed from benchmark3 + originals from topoagent)
# =============================================================================

def _get_ph_tool(desc_name: str):
    """Get the tool class for a PH-based descriptor."""
    if desc_name == 'persistence_image':
        from RuleBenchmark.benchmark3.descriptors import PersistenceImageTool
        return PersistenceImageTool
    elif desc_name == 'betti_curves':
        from RuleBenchmark.benchmark3.descriptors import BettiCurvesTool
        return BettiCurvesTool
    elif desc_name == 'tropical_coordinates':
        from RuleBenchmark.benchmark3.descriptors import TropicalCoordinatesTool
        return TropicalCoordinatesTool
    elif desc_name == 'ATOL':
        from RuleBenchmark.benchmark3.descriptors import ATOLTool
        return ATOLTool
    elif desc_name == 'persistence_statistics':
        from RuleBenchmark.benchmark3.descriptors import PersistenceStatisticsTool
        return PersistenceStatisticsTool
    elif desc_name == 'persistence_landscapes':
        from topoagent.tools.vectorization import PersistenceLandscapesTool
        return PersistenceLandscapesTool
    elif desc_name == 'persistence_silhouette':
        from topoagent.tools.vectorization import PersistenceSilhouetteTool
        return PersistenceSilhouetteTool
    elif desc_name == 'persistence_entropy':
        from topoagent.tools.descriptors import PersistenceEntropyTool
        return PersistenceEntropyTool
    elif desc_name == 'persistence_codebook':
        from topoagent.tools.descriptors import PersistenceCodebookTool
        return PersistenceCodebookTool
    elif desc_name == 'template_functions':
        from topoagent.tools.descriptors import TemplateFunctionsTool
        return TemplateFunctionsTool
    else:
        raise ValueError(f"Unknown PH descriptor: {desc_name}")


def _get_image_tool(desc_name: str):
    """Get the tool class for an image-based descriptor."""
    if desc_name == 'minkowski_functionals':
        from topoagent.tools.morphology import MinkowskiFunctionalsTool
        return MinkowskiFunctionalsTool
    elif desc_name == 'euler_characteristic_curve':
        from topoagent.tools.descriptors import EulerCharacteristicCurveTool
        return EulerCharacteristicCurveTool
    elif desc_name == 'euler_characteristic_transform':
        from topoagent.tools.descriptors import EulerCharacteristicTransformTool
        return EulerCharacteristicTransformTool
    elif desc_name == 'edge_histogram':
        from topoagent.tools.descriptors import EdgeHistogramTool
        return EdgeHistogramTool
    elif desc_name == 'lbp_texture':
        from topoagent.tools.descriptors import LBPTextureTool
        return LBPTextureTool
    else:
        raise ValueError(f"Unknown image descriptor: {desc_name}")


# =============================================================================
# Parameter Mapping (config params -> tool _run() params)
# =============================================================================

PH_BASED = {
    'persistence_image', 'persistence_landscapes', 'betti_curves',
    'persistence_silhouette', 'persistence_entropy', 'persistence_statistics',
    'tropical_coordinates', 'persistence_codebook', 'ATOL', 'template_functions',
}
IMAGE_BASED = {
    'minkowski_functionals', 'euler_characteristic_curve',
    'euler_characteristic_transform', 'edge_histogram', 'lbp_texture',
}
LEARNED = {'ATOL', 'persistence_codebook'}


def _map_params(desc_name: str, params: Dict) -> Dict:
    """Map config params to tool-specific _run() params."""
    if desc_name == 'persistence_image':
        return {
            'resolution': params.get('resolution', 20),
            'sigma': params.get('sigma', 0.1),
            'weight_function': params.get('weight_function', 'linear'),
        }
    elif desc_name == 'persistence_landscapes':
        return {
            'n_layers': params.get('n_layers', 4),
            'n_bins': params.get('n_bins', 100),
            'combine_dims': params.get('combine_dims', True),
        }
    elif desc_name == 'persistence_silhouette':
        return {
            'n_bins': params.get('n_bins', 100),
            'power': params.get('power', 1.0),
        }
    elif desc_name == 'betti_curves':
        return {
            'n_bins': params.get('n_bins', 100),
            'normalize': params.get('normalize', False),
        }
    elif desc_name == 'persistence_entropy':
        return {
            'mode': params.get('mode', 'vector'),
            'n_bins': params.get('n_bins', 100),
        }
    elif desc_name == 'persistence_codebook':
        return {'codebook_size': params.get('codebook_size', 32)}
    elif desc_name == 'ATOL':
        return {'n_centers': params.get('n_centers', 16)}
    elif desc_name == 'tropical_coordinates':
        return {'max_terms': params.get('max_terms', 5)}
    elif desc_name == 'template_functions':
        return {
            'n_templates': params.get('n_templates', 25),
            'template_type': params.get('template_type', 'tent'),
        }
    elif desc_name == 'persistence_statistics':
        return {'subset': params.get('subset', 'basic')}
    elif desc_name == 'euler_characteristic_curve':
        return {
            'resolution': params.get('resolution', 100),
            'max_image_size': 1024,  # same rule as PH: original if ≤1024
        }
    elif desc_name == 'euler_characteristic_transform':
        return {
            'n_directions': params.get('n_directions', 32),
            'n_heights': params.get('n_heights', 20),
            'max_image_size': 512,  # cap at 512 — ECT is O(n_pixels × n_dir × n_heights)
        }
    elif desc_name == 'edge_histogram':
        return {
            'n_orientation_bins': params.get('n_orientation_bins', 8),
            'n_spatial_cells': params.get('n_spatial_cells', 10),
        }
    elif desc_name == 'lbp_texture':
        return {
            'P': params.get('P', 8),
            'R': params.get('R', 1.0),
            'method': params.get('method', 'uniform'),
            'scales': params.get('scales', [(8, 1.0), (16, 2.0), (24, 3.0)]),
        }
    elif desc_name == 'minkowski_functionals':
        return {
            'n_thresholds': params.get('n_thresholds', 10),
            'adaptive': params.get('adaptive', False),
        }
    else:
        return params


# =============================================================================
# PH Descriptor Extraction
# =============================================================================

def _convert_one_diagram(diag_np: Dict) -> Dict:
    """Convert a single numpy-format diagram to dict-list format.

    Input:  {'H0': ndarray(n,2), 'H1': ndarray(m,2)}
    Output: {'H0': [{'birth': b, 'death': d}, ...], 'H1': [...]}

    Memory-efficient: converts one diagram at a time instead of all at once.
    For large PH caches (e.g. ISIC2019 at 12GB), bulk conversion via
    diagrams_to_dict_list() would create ~140GB of Python dicts.
    """
    d = {}
    for key in ['H0', 'H1']:
        arr = diag_np.get(key)
        if arr is not None and len(arr) > 0:
            d[key] = [{'birth': float(b), 'death': float(d_)}
                      for b, d_ in arr]
        else:
            d[key] = []
    return d


def _extract_ph_features(
    desc_name: str,
    diagrams: List[Dict],  # numpy format: {'H0': ndarray(n,2), 'H1': ndarray(m,2)}
    params: Dict,
    expected_dim: int,
) -> np.ndarray:
    """Extract features from PH-based descriptors.

    Args:
        desc_name: Descriptor name
        diagrams: PH diagrams in numpy format
        params: Optimal parameters
        expected_dim: Expected output dimension

    Returns:
        Feature matrix (N, expected_dim)
    """
    tool_cls = _get_ph_tool(desc_name)
    tool = tool_cls()
    run_params = _map_params(desc_name, params)

    feat_list = []
    for i, diag_np in enumerate(diagrams):
        # Convert one diagram at a time to avoid massive memory allocation
        diag = _convert_one_diagram(diag_np)
        result = tool._run(persistence_data=diag, **run_params)
        if result.get('success', False):
            vec = np.array(result['combined_vector'], dtype=np.float32)
        else:
            vec = np.zeros(expected_dim, dtype=np.float32)

        # Pad/truncate to expected dimension
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

        if (i + 1) % 1000 == 0:
            print(f"      {desc_name}: {i+1}/{len(diagrams)} images extracted")

    return np.array(feat_list, dtype=np.float32)


# =============================================================================
# Learned Descriptor Extraction (ATOL, persistence_codebook)
# =============================================================================

def _numpy_diag_to_points(diag_np: Dict, dim_key: str) -> np.ndarray:
    """Extract (birth, persistence) points directly from numpy diagram.

    No dict conversion — avoids 12x memory amplification.
    Input:  diag_np = {'H0': ndarray(n,2), 'H1': ndarray(m,2)} with [birth, death]
    Output: ndarray(k,2) with [birth, persistence] where persistence = death - birth
    """
    arr = diag_np.get(dim_key)
    if arr is None or len(arr) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Filter: finite values and death > birth
    valid = (np.isfinite(arr).all(axis=1) &
             (arr[:, 1] > arr[:, 0]) &
             (np.abs(arr).max(axis=1) < 1e6))
    arr = arr[valid]
    if len(arr) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([arr[:, 0], arr[:, 1] - arr[:, 0]]).astype(np.float32)


def _collect_all_points(diagrams: List[Dict], dim_key: str) -> np.ndarray:
    """Pool (birth, persistence) points from all diagrams for one dimension.

    Streams one diagram at a time — constant memory overhead.
    """
    chunks = []
    for diag_np in diagrams:
        pts = _numpy_diag_to_points(diag_np, dim_key)
        if len(pts) > 0:
            chunks.append(pts)
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(chunks)


def extract_learned_features(
    desc_name: str,
    train_diagrams: List[Dict],
    test_diagrams: List[Dict],
    params: Dict,
    expected_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for learned descriptors with proper fit/transform.

    Works directly with numpy PH arrays — no dict conversion, no OOM.

    Args:
        desc_name: 'ATOL' or 'persistence_codebook'
        train_diagrams: Training PH diagrams (numpy format)
        test_diagrams: Test PH diagrams (numpy format)
        params: Optimal parameters
        expected_dim: Expected feature dimension

    Returns:
        (X_train, X_test) feature matrices
    """
    from sklearn.cluster import MiniBatchKMeans

    run_params = _map_params(desc_name, params)

    # Detect dimensions present in diagrams
    dim_keys = sorted(set(
        k for d in train_diagrams[:10] for k in d.keys()
    ))  # e.g. ['H0', 'H1']

    # ---- Phase 1: Fit (collect points → MiniBatchKMeans) per dimension ----
    fitted_models = {}  # dim_key → model info

    for dim_key in dim_keys:
        all_points = _collect_all_points(train_diagrams, dim_key)
        print(f"      {desc_name} fit {dim_key}: {len(all_points)} points pooled from {len(train_diagrams)} diagrams")

        if len(all_points) == 0:
            fitted_models[dim_key] = None
            continue

        if desc_name == 'ATOL':
            n_centers = run_params.get('n_centers', 16)
            actual_k = min(n_centers, len(all_points))
            kmeans = MiniBatchKMeans(n_clusters=actual_k, random_state=42, n_init=3)
            kmeans.fit(all_points)
            sigma_val = np.mean(np.std(all_points, axis=0))
            if sigma_val < 1e-10:
                sigma_val = 1.0
            fitted_models[dim_key] = {
                'type': 'atol',
                'centers': kmeans.cluster_centers_,
                'sigma': sigma_val,
                'n_centers': n_centers,
            }

        elif desc_name == 'persistence_codebook':
            codebook_size = run_params.get('codebook_size', 32)
            actual_k = min(codebook_size, len(all_points))
            kmeans = MiniBatchKMeans(
                n_clusters=actual_k, random_state=42,
                batch_size=min(1024, len(all_points)), n_init=3)
            kmeans.fit(all_points)
            fitted_models[dim_key] = {
                'type': 'codebook',
                'centers': kmeans.cluster_centers_,
                'codebook_size': codebook_size,
            }

    # ---- Phase 2: Transform (vectorize each diagram) ----

    def _transform_one(diag_np):
        """Vectorize a single diagram using fitted models."""
        all_vectors = []
        for dim_key in dim_keys:
            model = fitted_models.get(dim_key)
            points = _numpy_diag_to_points(diag_np, dim_key)

            if model is None or len(points) == 0:
                if desc_name == 'ATOL':
                    n_centers = run_params.get('n_centers', 16)
                    all_vectors.append(np.zeros(n_centers, dtype=np.float32))
                else:
                    codebook_size = run_params.get('codebook_size', 32)
                    all_vectors.append(np.zeros(codebook_size, dtype=np.float32))
                continue

            if model['type'] == 'atol':
                centers = model['centers']
                sigma = model['sigma']
                n_centers = model['n_centers']
                # ATOL: weighted Gaussian kernel distances
                diff = points[:, None, :] - centers[None, :, :]  # (N, K, 2)
                dists = np.linalg.norm(diff, axis=2)  # (N, K)
                weights = points[:, 1:2]  # persistence column
                gaussians = np.exp(-dists ** 2 / (2 * sigma ** 2))
                vec = np.sum(weights * gaussians, axis=0).astype(np.float32)
                # Pad if actual_k < n_centers
                if len(vec) < n_centers:
                    vec = np.pad(vec, (0, n_centers - len(vec)))
                all_vectors.append(vec)

            elif model['type'] == 'codebook':
                centers = model['centers']
                codebook_size = model['codebook_size']
                # Codebook: histogram of nearest-center assignments
                diff = points[:, None, :] - centers[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                nearest = np.argmin(dists, axis=1)
                hist = np.bincount(nearest, minlength=len(centers)).astype(np.float32)
                total = hist.sum()
                if total > 0:
                    hist = hist / total  # normalize
                # Pad to codebook_size
                if len(hist) < codebook_size:
                    hist = np.pad(hist, (0, codebook_size - len(hist)))
                elif len(hist) > codebook_size:
                    hist = hist[:codebook_size]
                all_vectors.append(hist)

        return np.concatenate(all_vectors)

    def _extract_batch(diagrams_np, label=""):
        feat_list = []
        for i, diag_np in enumerate(diagrams_np):
            try:
                vec = _transform_one(diag_np).astype(np.float32)
            except Exception:
                vec = np.zeros(expected_dim, dtype=np.float32)
            if len(vec) < expected_dim:
                vec = np.pad(vec, (0, expected_dim - len(vec)))
            elif len(vec) > expected_dim:
                vec = vec[:expected_dim]
            feat_list.append(vec)
            if (i + 1) % 1000 == 0:
                print(f"        {label}: {i+1}/{len(diagrams_np)}")
        return np.array(feat_list, dtype=np.float32)

    X_train = _extract_batch(train_diagrams, label=f"{desc_name} train")
    X_test = _extract_batch(test_diagrams, label=f"{desc_name} test")
    return X_train, X_test


# =============================================================================
# Image Descriptor Extraction
# =============================================================================

def _extract_image_features(
    desc_name: str,
    images: np.ndarray,  # (N, H, W) grayscale float [0,1]
    params: Dict,
    expected_dim: int,
) -> np.ndarray:
    """Extract features from image-based descriptors."""
    if desc_name == 'minkowski_functionals':
        return _extract_minkowski(images, params, expected_dim)
    if desc_name == 'lbp_texture':
        return _extract_lbp(images, params, expected_dim)

    tool_cls = _get_image_tool(desc_name)
    tool = tool_cls()
    run_params = _map_params(desc_name, params)
    # Remove non-tool params
    run_params.pop('scales', None)

    feat_list = []
    for img in images:
        result = tool._run(image_array=img.tolist(), **run_params)
        if result.get('success', False):
            vec = np.array(result['combined_vector'], dtype=np.float32)
        else:
            vec = np.zeros(expected_dim, dtype=np.float32)

        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


def _extract_minkowski(images: np.ndarray, params: Dict, expected_dim: int) -> np.ndarray:
    """Extract Minkowski functionals at multiple thresholds."""
    from topoagent.tools.morphology import MinkowskiFunctionalsTool

    n_thresholds = params.get('n_thresholds', 10)
    adaptive = params.get('adaptive', False)
    tool = MinkowskiFunctionalsTool()
    percentiles = np.linspace(10, 90, n_thresholds)

    feat_list = []
    for img in images:
        if adaptive:
            thresholds = np.percentile(img, percentiles)
        else:
            thresholds = np.linspace(0.05, 0.95, n_thresholds)

        curve = []
        for t in thresholds:
            result = tool._run(image_array=img.tolist(), threshold=float(t))
            if result.get('success', False):
                mf = result.get('minkowski_functionals', {})
                curve.extend([
                    mf.get('volume', 0.0),
                    mf.get('surface', 0.0),
                    mf.get('euler_characteristic', 0.0),
                ])
            else:
                curve.extend([0.0, 0.0, 0.0])

        vec = np.array(curve, dtype=np.float32)
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


def _extract_lbp(images: np.ndarray, params: Dict, expected_dim: int) -> np.ndarray:
    """Extract multi-scale LBP."""
    from topoagent.tools.descriptors import LBPTextureTool

    scales = params.get('scales', [(8, 1.0), (16, 2.0), (24, 3.0)])
    tool = LBPTextureTool()

    feat_list = []
    for img in images:
        multi_scale_vec = []
        for P, R in scales:
            result = tool._run(image_array=img.tolist(), P=P, R=R, method='uniform')
            if result.get('success', False):
                multi_scale_vec.extend(result['combined_vector'])
            else:
                multi_scale_vec.extend([0.0] * (P + 2))

        vec = np.array(multi_scale_vec, dtype=np.float32)
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        elif len(vec) > expected_dim:
            vec = vec[:expected_dim]
        feat_list.append(vec)

    return np.array(feat_list, dtype=np.float32)


# =============================================================================
# Unified Extraction API
# =============================================================================

def extract_features(
    desc_name: str,
    diagrams: Optional[List[Dict]] = None,
    images: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    expected_dim: int = 100,
) -> np.ndarray:
    """Extract features for a single descriptor.

    Args:
        desc_name: Descriptor name (one of 15)
        diagrams: PH diagrams (for PH-based descriptors), numpy format
        images: Grayscale images (N, H, W) for image-based descriptors
        params: Descriptor parameters (from optimal_rules)
        expected_dim: Expected feature dimension D*

    Returns:
        Feature matrix (N, expected_dim)
    """
    if params is None:
        params = {}

    if desc_name in PH_BASED:
        if diagrams is None:
            raise ValueError(f"PH-based descriptor '{desc_name}' requires diagrams")
        return _extract_ph_features(desc_name, diagrams, params, expected_dim)
    elif desc_name in IMAGE_BASED:
        if images is None:
            raise ValueError(f"Image-based descriptor '{desc_name}' requires images")
        return _extract_image_features(desc_name, images, params, expected_dim)
    else:
        raise ValueError(f"Unknown descriptor: {desc_name}")


def extract_features_per_channel(
    desc_name: str,
    diags_R: Optional[List[Dict]] = None,
    diags_G: Optional[List[Dict]] = None,
    diags_B: Optional[List[Dict]] = None,
    images_R: Optional[np.ndarray] = None,
    images_G: Optional[np.ndarray] = None,
    images_B: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    expected_dim_per_channel: int = 100,
) -> np.ndarray:
    """Extract per-channel RGB features and concatenate.

    For PH-based: pass diags_R, diags_G, diags_B
    For image-based: pass images_R, images_G, images_B
    Total dim = 3 * expected_dim_per_channel.

    Returns:
        Feature matrix (N, 3 * expected_dim_per_channel)
    """
    if params is None:
        params = {}

    if desc_name in PH_BASED:
        feat_R = extract_features(desc_name, diagrams=diags_R, params=params,
                                  expected_dim=expected_dim_per_channel)
        feat_G = extract_features(desc_name, diagrams=diags_G, params=params,
                                  expected_dim=expected_dim_per_channel)
        feat_B = extract_features(desc_name, diagrams=diags_B, params=params,
                                  expected_dim=expected_dim_per_channel)
    elif desc_name in IMAGE_BASED:
        feat_R = extract_features(desc_name, images=images_R, params=params,
                                  expected_dim=expected_dim_per_channel)
        feat_G = extract_features(desc_name, images=images_G, params=params,
                                  expected_dim=expected_dim_per_channel)
        feat_B = extract_features(desc_name, images=images_B, params=params,
                                  expected_dim=expected_dim_per_channel)
    else:
        raise ValueError(f"Unknown descriptor: {desc_name}")

    return np.concatenate([feat_R, feat_G, feat_B], axis=1)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    from RuleBenchmark.benchmark4.precompute_ph import load_cached_ph
    from RuleBenchmark.benchmark4.optimal_rules import OptimalRules

    rules = OptimalRules()

    # Test with BloodMNIST cache (if available)
    try:
        data = load_cached_ph('BloodMNIST', n_samples=100)
    except FileNotFoundError:
        print("No BloodMNIST cache found. Run precompute_ph.py first.")
        sys.exit(0)

    diags = data['diagrams_gray_sublevel']
    print(f"Loaded {len(diags)} PH diagrams")

    # Test a few PH-based descriptors
    for desc in ['betti_curves', 'persistence_entropy', 'persistence_statistics']:
        cfg = rules.get_descriptor_config(desc, 'discrete_cells', 'grayscale')
        features = extract_features(
            desc, diagrams=diags[:10],
            params=cfg['params'], expected_dim=cfg['total_dim'])
        print(f"  {desc:35s}  shape={features.shape}  "
              f"range=[{features.min():.4f}, {features.max():.4f}]  "
              f"NaN={np.isnan(features).sum()}")

    print("\nDescriptor runner test PASSED")
