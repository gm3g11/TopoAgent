"""Compute 25 cheap (non-PH) dataset characterization features.

11 groups:
  Structure (4), Intensity (6), Gradient (2), Topology proxies (3),
  Texture (2), Frequency (1), Stability (1), Components (2),
  Polarity (2), Scale (1), Fine texture (1)
"""

import numpy as np
from scipy import ndimage, stats
from collections import OrderedDict

from .config import FEATURE_NAMES, N_REPEATS, SAMPLES_PER_REPEAT


def _safe(val):
    """Replace NaN/Inf with 0."""
    return float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))


def _otsu_threshold(image):
    """Simple Otsu thresholding without skimage dependency."""
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.5

    sum_total = (hist * bin_centers).sum()
    sum_bg = 0.0
    weight_bg = 0
    max_var = 0.0
    threshold = 0.5

    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = bin_centers[i]

    return threshold


def _connected_components_stats(binary_mask):
    """Label connected components and return stats."""
    labeled, n_components = ndimage.label(binary_mask)
    if n_components == 0:
        return 0, np.array([0.0]), 0
    sizes = ndimage.sum(binary_mask, labeled, range(1, n_components + 1))
    sizes = np.array(sizes)
    return n_components, sizes, labeled


def _compute_glcm_features(image, distances=[1], angles=[0]):
    """Simplified GLCM computation without skimage.feature dependency."""
    # Quantize to 8 levels for speed
    n_levels = 8
    quantized = np.clip((image * n_levels).astype(int), 0, n_levels - 1)

    h, w = quantized.shape
    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)

    # Horizontal adjacency (angle=0, distance=1)
    for i in range(h):
        for j in range(w - 1):
            a, b = quantized[i, j], quantized[i, j + 1]
            glcm[a, b] += 1
            glcm[b, a] += 1  # Symmetric

    total = glcm.sum()
    if total == 0:
        return 0.0, 1.0  # contrast=0, homogeneity=1

    glcm_norm = glcm / total

    # Contrast: sum_{i,j} (i-j)^2 * P(i,j)
    i_idx, j_idx = np.meshgrid(range(n_levels), range(n_levels), indexing='ij')
    contrast = np.sum((i_idx - j_idx) ** 2 * glcm_norm)

    # Homogeneity: sum_{i,j} P(i,j) / (1 + |i-j|)
    homogeneity = np.sum(glcm_norm / (1 + np.abs(i_idx - j_idx)))

    return _safe(contrast), _safe(homogeneity)


def compute_cheap_features(images, labels, max_samples=100, seed=42):
    """Compute 25 cheap dataset characterization features.

    Args:
        images: np.ndarray of shape (N, H, W), float32 in [0, 1]
        labels: np.ndarray of shape (N,), integer class labels
        max_samples: max number of images to use for per-image features
        seed: random seed for subsampling

    Returns:
        OrderedDict with 25 feature name-value pairs
    """
    rng = np.random.RandomState(seed)

    n_total = len(labels)
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_classes)

    # Subsample images for per-image features
    if n_total > max_samples:
        idx = rng.choice(n_total, max_samples, replace=False)
        imgs = images[idx]
    else:
        imgs = images

    n_imgs = len(imgs)

    # ─── Structure (4) ───────────────────────────────────────────────────
    samples_per_class = n_total / max(n_classes, 1)
    class_imbalance = class_counts.max() / max(class_counts.min(), 1)

    # ─── Intensity (6) ───────────────────────────────────────────────────
    pixel_values = imgs.ravel()
    # Subsample pixels for speed if too many
    if len(pixel_values) > 500000:
        pixel_values = rng.choice(pixel_values, 500000, replace=False)

    intensity_mean = pixel_values.mean()
    intensity_std = pixel_values.std()
    intensity_skewness = stats.skew(pixel_values)
    intensity_kurtosis = stats.kurtosis(pixel_values)
    p5, p95 = np.percentile(pixel_values, [5, 95])
    intensity_p95_p5 = p95 - p5

    # Entropy of intensity histogram
    hist, _ = np.histogram(pixel_values, bins=256, range=(0, 1))
    hist_norm = hist / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    intensity_entropy = -np.sum(hist_norm * np.log2(hist_norm))

    # ─── Gradient (2) ────────────────────────────────────────────────────
    edge_densities = []
    gradient_means = []
    for img in imgs:
        gy, gx = np.gradient(img)
        grad_mag = np.sqrt(gx**2 + gy**2)
        gradient_means.append(grad_mag.mean())
        # Edge density: fraction of pixels with gradient > 2*mean
        threshold = 2 * grad_mag.mean()
        edge_densities.append((grad_mag > threshold).mean() if threshold > 0 else 0.0)

    edge_density = np.mean(edge_densities)
    gradient_mean = np.mean(gradient_means)

    # ─── Topology Proxies (3) ────────────────────────────────────────────
    otsu_components_list = []
    otsu_holes_list = []
    fill_ratios = []

    for img in imgs:
        thresh = _otsu_threshold(img)
        binary = (img > thresh).astype(np.uint8)

        # Connected components of foreground
        n_comp, sizes, _ = _connected_components_stats(binary)
        otsu_components_list.append(n_comp)

        # Holes: connected components of background (inverted)
        n_holes, _, _ = _connected_components_stats(1 - binary)
        # Subtract 1 for the infinite background component
        otsu_holes_list.append(max(0, n_holes - 1))

        # Fill ratio
        filled = ndimage.binary_fill_holes(binary)
        fill_ratio = filled.sum() / max(binary.sum(), 1)
        fill_ratios.append(fill_ratio)

    otsu_components = np.mean(otsu_components_list)
    otsu_holes = np.mean(otsu_holes_list)
    binary_fill_ratio = np.mean(fill_ratios)

    # ─── Texture (2) ────────────────────────────────────────────────────
    # GLCM on a subset for speed
    glcm_subset = imgs[:min(50, n_imgs)]
    contrasts = []
    homogeneities = []
    for img in glcm_subset:
        # Downsample for GLCM speed
        small = img[::4, ::4] if img.shape[0] > 64 else img
        c, h = _compute_glcm_features(small)
        contrasts.append(c)
        homogeneities.append(h)

    glcm_contrast = np.mean(contrasts)
    glcm_homogeneity = np.mean(homogeneities)

    # ─── Frequency (1) ──────────────────────────────────────────────────
    fft_ratios = []
    for img in imgs[:min(50, n_imgs)]:
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        h, w = img.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8  # Low frequency radius
        y, x = np.ogrid[:h, :w]
        low_mask = ((y - cy)**2 + (x - cx)**2) <= r**2
        total_energy = magnitude.sum()
        low_energy = magnitude[low_mask].sum()
        fft_ratios.append(low_energy / max(total_energy, 1e-10))

    fft_low_freq_ratio = np.mean(fft_ratios)

    # ─── Stability (1) ──────────────────────────────────────────────────
    # Otsu stability: higher = more stable (inverse CV of component counts)
    cv = np.std(otsu_components_list) / max(np.mean(otsu_components_list), 1e-10)
    otsu_stability = 1.0 / (1.0 + cv)

    # ─── Components (2) ─────────────────────────────────────────────────
    comp_sizes_all = []
    for img in imgs:
        thresh = _otsu_threshold(img)
        binary = (img > thresh).astype(np.uint8)
        n_comp, sizes, _ = _connected_components_stats(binary)
        if n_comp > 0:
            comp_sizes_all.extend(sizes.tolist())

    if len(comp_sizes_all) > 0:
        comp_sizes_arr = np.array(comp_sizes_all)
        component_size_mean = comp_sizes_arr.mean()
        component_size_cv = comp_sizes_arr.std() / max(comp_sizes_arr.mean(), 1e-10)
    else:
        component_size_mean = 0.0
        component_size_cv = 0.0

    # ─── Polarity (2) ───────────────────────────────────────────────────
    fg_bg_contrasts = []
    polarities = []
    for img in imgs:
        thresh = _otsu_threshold(img)
        fg = img[img > thresh]
        bg = img[img <= thresh]
        if len(fg) > 0 and len(bg) > 0:
            fg_mean = fg.mean()
            bg_mean = bg.mean()
            fg_bg_contrasts.append(abs(fg_mean - bg_mean))
            # Polarity: is foreground brighter (+1) or darker (-1)?
            polarities.append(1.0 if fg_mean > bg_mean else -1.0)
        else:
            fg_bg_contrasts.append(0.0)
            polarities.append(0.0)

    fg_bg_contrast = np.mean(fg_bg_contrasts)
    polarity = np.mean(polarities)

    # ─── Scale (1) ──────────────────────────────────────────────────────
    largest_ratios = []
    for img in imgs:
        thresh = _otsu_threshold(img)
        binary = (img > thresh).astype(np.uint8)
        n_comp, sizes, _ = _connected_components_stats(binary)
        if n_comp > 0 and binary.sum() > 0:
            largest_ratios.append(sizes.max() / binary.sum())
        else:
            largest_ratios.append(0.0)

    largest_component_ratio = np.mean(largest_ratios)

    # ─── Fine Texture (1) ───────────────────────────────────────────────
    laplacian_vars = []
    for img in imgs[:min(50, n_imgs)]:
        lap = ndimage.laplace(img)
        laplacian_vars.append(lap.var())

    laplacian_variance = np.mean(laplacian_vars)

    # ─── Assemble ───────────────────────────────────────────────────────
    features = OrderedDict([
        ("n_samples", _safe(n_total)),
        ("n_classes", _safe(n_classes)),
        ("samples_per_class", _safe(samples_per_class)),
        ("class_imbalance", _safe(class_imbalance)),
        ("intensity_mean", _safe(intensity_mean)),
        ("intensity_std", _safe(intensity_std)),
        ("intensity_skewness", _safe(intensity_skewness)),
        ("intensity_kurtosis", _safe(intensity_kurtosis)),
        ("intensity_p95_p5", _safe(intensity_p95_p5)),
        ("intensity_entropy", _safe(intensity_entropy)),
        ("edge_density", _safe(edge_density)),
        ("gradient_mean", _safe(gradient_mean)),
        ("otsu_components", _safe(otsu_components)),
        ("otsu_holes", _safe(otsu_holes)),
        ("binary_fill_ratio", _safe(binary_fill_ratio)),
        ("glcm_contrast", _safe(glcm_contrast)),
        ("glcm_homogeneity", _safe(glcm_homogeneity)),
        ("fft_low_freq_ratio", _safe(fft_low_freq_ratio)),
        ("otsu_stability", _safe(otsu_stability)),
        ("component_size_mean", _safe(component_size_mean)),
        ("component_size_cv", _safe(component_size_cv)),
        ("fg_bg_contrast", _safe(fg_bg_contrast)),
        ("polarity", _safe(polarity)),
        ("largest_component_ratio", _safe(largest_component_ratio)),
        ("laplacian_variance", _safe(laplacian_variance)),
    ])

    assert len(features) == 25, f"Expected 25 features, got {len(features)}"
    assert list(features.keys()) == FEATURE_NAMES, "Feature names mismatch"

    return features


def compute_cheap_features_stable(images, labels, n_repeats=N_REPEATS,
                                  samples_per_repeat=SAMPLES_PER_REPEAT, seed=42):
    """Compute features with stability averaging over multiple random subsamples.

    Args:
        images: np.ndarray of shape (N, H, W), float32 in [0, 1]
        labels: np.ndarray of shape (N,), integer class labels
        n_repeats: number of random subsamples
        samples_per_repeat: images per subsample
        seed: base random seed

    Returns:
        OrderedDict with 25 averaged feature values
    """
    all_features = []

    for rep in range(n_repeats):
        rep_seed = seed + rep * 1000
        feat = compute_cheap_features(images, labels, max_samples=samples_per_repeat,
                                      seed=rep_seed)
        all_features.append(feat)

    # Average across repeats
    averaged = OrderedDict()
    for name in FEATURE_NAMES:
        values = [f[name] for f in all_features]
        averaged[name] = _safe(np.mean(values))

    return averaged
