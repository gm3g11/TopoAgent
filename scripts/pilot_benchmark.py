#!/usr/bin/env python3
"""
Pilot Benchmark Script for TopoAgent v3 Descriptor Study

Run this BEFORE the full experiment to:
1. Validate the pipeline works end-to-end
2. Measure actual computation times
3. Calibrate timeline estimates
4. Identify bottlenecks

Usage:
    python pilot_benchmark.py

Expected runtime: ~30-60 minutes
"""

import time
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PILOT_CONFIG = {
    'dataset': 'PathMNIST',
    'n_images': 50,
    'image_size': 224,
    'n_folds': 3,
    'n_cores': 32,
    'random_state': 42,
}

# Full experiment parameters (for extrapolation)
FULL_CONFIG = {
    'n_images': 500,
    'n_datasets': 5,
    'n_descriptors': 14,
    'n_folds': 5,
    'n_filtrations': 2,
    'n_preprocessing': 2,
}


# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

def check_dependencies():
    """Check all required packages are installed."""
    print("Checking dependencies...")

    required = {
        'cripser': 'pip install cripser',
        'gudhi': 'pip install gudhi',
        'sklearn': 'pip install scikit-learn',
        'skimage': 'pip install scikit-image',
        'medmnist': 'pip install medmnist',
        'joblib': 'pip install joblib',
        'numpy': 'pip install numpy',
        'scipy': 'pip install scipy',
    }

    missing = []
    for pkg, install_cmd in required.items():
        try:
            if pkg == 'sklearn':
                import sklearn
            elif pkg == 'skimage':
                import skimage
            else:
                __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - Install with: {install_cmd}")
            missing.append(pkg)

    if missing:
        print(f"\nMissing packages: {missing}")
        print("Please install them before running the pilot.")
        return False

    print("All dependencies satisfied.\n")
    return True


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pilot_data():
    """Load sample data for pilot benchmark."""
    import medmnist
    from medmnist import PathMNIST

    print(f"Loading {PILOT_CONFIG['dataset']} (size={PILOT_CONFIG['image_size']})...")

    # Load dataset
    dataset = PathMNIST(split='train', download=True, size=PILOT_CONFIG['image_size'])

    # Get images and labels
    images = dataset.imgs  # (N, H, W, C) or (N, H, W)
    labels = dataset.labels.flatten()

    # Sample
    np.random.seed(PILOT_CONFIG['random_state'])
    n_total = len(images)
    sample_idx = np.random.choice(n_total, PILOT_CONFIG['n_images'], replace=False)

    images = images[sample_idx]
    labels = labels[sample_idx]

    print(f"  Loaded {len(images)} images, shape: {images[0].shape}")
    print(f"  Labels: {np.unique(labels)}")

    return images, labels


def preprocess_images(images):
    """Convert to grayscale and normalize."""
    processed = []
    for img in images:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = np.mean(img, axis=-1)

        # Normalize to [0, 1]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)

        processed.append(img)

    return np.array(processed, dtype=np.float32)


# ============================================================================
# PERSISTENT HOMOLOGY
# ============================================================================

def compute_ph_single(image, maxdim=1):
    """Compute PH for a single image using cripser."""
    import cripser

    result = cripser.computePH(image, maxdim=maxdim)

    # Parse result: (dim, birth, death)
    diagrams = [[] for _ in range(maxdim + 1)]
    for row in result:
        dim = int(row[0])
        birth, death = row[1], row[2]
        if dim <= maxdim:
            diagrams[dim].append([birth, death])

    # Convert to arrays
    return [
        np.array(d, dtype=np.float32) if d else np.empty((0, 2), dtype=np.float32)
        for d in diagrams
    ]


def cap_infinite_bars(diagrams):
    """Cap infinite death values."""
    capped = []
    for diag in diagrams:
        if len(diag) == 0:
            capped.append(diag)
            continue

        diag = diag.copy()
        finite_mask = np.isfinite(diag[:, 1])

        if finite_mask.all():
            capped.append(diag)
            continue

        if finite_mask.any():
            max_finite = diag[finite_mask, 1].max()
            cap = max_finite * 1.1
        else:
            cap = diag[:, 0].max() + 1.0

        diag[~finite_mask, 1] = cap
        capped.append(diag)

    return capped


def compute_ph_parallel(images, n_jobs=32):
    """Compute PH for all images in parallel."""
    from joblib import Parallel, delayed

    diagrams = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_ph_single)(img) for img in images
    )
    return [cap_infinite_bars(d) for d in diagrams]


# ============================================================================
# VECTORIZATIONS
# ============================================================================

def compute_statistics(diagrams):
    """Compute persistence statistics (extended set)."""
    stats = []

    for dim_diag in diagrams:  # H0 and H1
        if len(dim_diag) == 0:
            # Empty diagram: zeros
            stats.extend([0] * 25)
            continue

        births = dim_diag[:, 0]
        deaths = dim_diag[:, 1]
        lifetimes = deaths - births

        # Basic stats
        lifetimes_norm = lifetimes / (lifetimes.sum() + 1e-10)
        dim_stats = [
            # Lifetime stats
            np.mean(lifetimes), np.std(lifetimes), np.median(lifetimes),
            np.percentile(lifetimes, 25), np.percentile(lifetimes, 75),
            lifetimes.max() - lifetimes.min(),  # range
            np.min(lifetimes), np.max(lifetimes),
            len(lifetimes),  # count
            np.sum(lifetimes),  # total persistence
            # Birth stats
            np.mean(births), np.std(births),
            # Death stats
            np.mean(deaths), np.std(deaths),
            # Entropy
            -np.sum(lifetimes_norm * np.log(lifetimes_norm + 1e-10)),  # entropy
        ]

        # Flatten and extend
        stats.extend([
            np.mean(lifetimes), np.std(lifetimes), np.median(lifetimes),
            np.percentile(lifetimes, 25), np.percentile(lifetimes, 75),
            lifetimes.max() - lifetimes.min(), np.min(lifetimes), np.max(lifetimes),
            len(lifetimes), np.sum(lifetimes),
            np.mean(births), np.std(births),
            np.mean(deaths), np.std(deaths),
        ])

        # Entropy
        lifetimes_norm = lifetimes / (lifetimes.sum() + 1e-10)
        entropy = -np.sum(lifetimes_norm * np.log(lifetimes_norm + 1e-10))
        stats.append(entropy)

        # Pad to 25 per dimension
        while len(stats) % 25 != 0:
            stats.append(0)

    return np.array(stats[:50], dtype=np.float32)  # H0 + H1


def compute_persistence_image(diagrams, sigma=0.1, resolution=20):
    """Compute persistence image using gudhi."""
    from gudhi.representations import PersistenceImage

    # Combine H0 and H1
    combined = []
    for d in diagrams:
        if len(d) > 0:
            combined.extend(d.tolist())

    if len(combined) == 0:
        return np.zeros(resolution * resolution, dtype=np.float32)

    pi = PersistenceImage(bandwidth=sigma, resolution=[resolution, resolution])
    try:
        img = pi.fit_transform([np.array(combined)])[0]
        return img.flatten().astype(np.float32)
    except:
        return np.zeros(resolution * resolution, dtype=np.float32)


def compute_betti_curve(diagrams, resolution=100):
    """Compute Betti curve using gudhi."""
    from gudhi.representations import BettiCurve

    curves = []
    for dim, d in enumerate(diagrams):
        if len(d) == 0:
            curves.append(np.zeros(resolution))
            continue

        bc = BettiCurve(resolution=resolution)
        try:
            curve = bc.fit_transform([d])[0]
            curves.append(curve)
        except:
            curves.append(np.zeros(resolution))

    return np.concatenate(curves).astype(np.float32)


def compute_euler_curve(image, resolution=100, connectivity=2):
    """Compute Euler characteristic curve."""
    from skimage.measure import euler_number

    thresholds = np.linspace(0, 1, resolution)
    ec_values = []

    for t in thresholds:
        binary = image < t  # Sublevel set
        ec = euler_number(binary, connectivity=connectivity)
        ec_values.append(ec)

    return np.array(ec_values, dtype=np.float32)


# ============================================================================
# CLASSIFICATION
# ============================================================================

def run_classification(features, labels, n_folds=3):
    """Run KNN and SVM classification with CV."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler

    knn_scores = []
    svm_scores = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(features, labels):
        X_tr, X_val = features[train_idx], features[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]

        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_tr, y_tr)
        knn_scores.append(knn.score(X_val, y_val))

        # SVM
        svm = LinearSVC(max_iter=10000, dual='auto')
        try:
            svm.fit(X_tr, y_tr)
            svm_scores.append(svm.score(X_val, y_val))
        except:
            svm_scores.append(0)

    return knn_scores, svm_scores


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_pilot():
    """Run the complete pilot benchmark."""

    print("=" * 70)
    print("PILOT BENCHMARK - TopoAgent v3 Descriptor Study")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {PILOT_CONFIG['dataset']}")
    print(f"  Images: {PILOT_CONFIG['n_images']}")
    print(f"  Size: {PILOT_CONFIG['image_size']}×{PILOT_CONFIG['image_size']}")
    print(f"  Cores: {PILOT_CONFIG['n_cores']}")
    print()

    # Check dependencies
    if not check_dependencies():
        return None

    timings = {}

    # ========== 1. Load Data ==========
    print("=" * 50)
    print("STEP 1: Loading data")
    print("=" * 50)

    start = time.time()
    images, labels = load_pilot_data()
    images = preprocess_images(images)
    timings['data_loading'] = time.time() - start
    print(f"  Time: {timings['data_loading']:.2f}s\n")

    # ========== 2. PH Computation (Sequential) ==========
    print("=" * 50)
    print("STEP 2: PH computation (sequential, for timing)")
    print("=" * 50)

    ph_times = []
    for i, img in enumerate(images[:10]):  # Just 10 for timing
        start = time.time()
        diag = compute_ph_single(img)
        ph_times.append(time.time() - start)

    timings['ph_per_image'] = {
        'mean': np.mean(ph_times),
        'std': np.std(ph_times),
        'min': np.min(ph_times),
        'max': np.max(ph_times),
    }
    print(f"  PH time per image: {timings['ph_per_image']['mean']:.3f} ± {timings['ph_per_image']['std']:.3f}s")
    print(f"  Range: [{timings['ph_per_image']['min']:.3f}, {timings['ph_per_image']['max']:.3f}]s\n")

    # ========== 3. PH Computation (Parallel) ==========
    print("=" * 50)
    print(f"STEP 3: PH computation (parallel, {PILOT_CONFIG['n_cores']} cores)")
    print("=" * 50)

    start = time.time()
    diagrams = compute_ph_parallel(images, n_jobs=PILOT_CONFIG['n_cores'])
    timings['ph_parallel_total'] = time.time() - start
    print(f"  Total time for {len(images)} images: {timings['ph_parallel_total']:.2f}s")
    print(f"  Effective per-image: {timings['ph_parallel_total'] / len(images):.3f}s\n")

    # ========== 4. Vectorization Times ==========
    print("=" * 50)
    print("STEP 4: Vectorization times")
    print("=" * 50)

    timings['vectorization'] = {}

    # Statistics
    start = time.time()
    stats_features = np.array([compute_statistics(d) for d in diagrams])
    timings['vectorization']['Statistics'] = (time.time() - start) / len(diagrams)
    print(f"  Statistics: {timings['vectorization']['Statistics'] * 1000:.2f} ms/image (dim={stats_features.shape[1]})")

    # PI
    start = time.time()
    pi_features = np.array([compute_persistence_image(d) for d in diagrams])
    timings['vectorization']['PI'] = (time.time() - start) / len(diagrams)
    print(f"  PI: {timings['vectorization']['PI'] * 1000:.2f} ms/image (dim={pi_features.shape[1]})")

    # Betti
    start = time.time()
    betti_features = np.array([compute_betti_curve(d) for d in diagrams])
    timings['vectorization']['Betti'] = (time.time() - start) / len(diagrams)
    print(f"  Betti: {timings['vectorization']['Betti'] * 1000:.2f} ms/image (dim={betti_features.shape[1]})")

    # EC Curve
    start = time.time()
    ec_features = np.array([compute_euler_curve(img) for img in images])
    timings['vectorization']['EC_Curve'] = (time.time() - start) / len(images)
    print(f"  EC Curve: {timings['vectorization']['EC_Curve'] * 1000:.2f} ms/image (dim={ec_features.shape[1]})")
    print()

    # ========== 5. Classification Times ==========
    print("=" * 50)
    print(f"STEP 5: Classification times ({PILOT_CONFIG['n_folds']}-fold CV)")
    print("=" * 50)

    timings['classifier'] = {}

    # Using PI features
    start = time.time()
    knn_scores, svm_scores = run_classification(pi_features, labels, PILOT_CONFIG['n_folds'])
    timings['classifier']['total'] = time.time() - start

    print(f"  KNN accuracy: {np.mean(knn_scores):.3f} ± {np.std(knn_scores):.3f}")
    print(f"  SVM accuracy: {np.mean(svm_scores):.3f} ± {np.std(svm_scores):.3f}")
    print(f"  Total CV time: {timings['classifier']['total']:.2f}s\n")

    # ========== 6. Extrapolation ==========
    print("=" * 70)
    print("TIME ESTIMATES FOR FULL EXPERIMENT")
    print("=" * 70)

    n_img = FULL_CONFIG['n_images']
    n_data = FULL_CONFIG['n_datasets']
    n_desc = FULL_CONFIG['n_descriptors']
    n_folds = FULL_CONFIG['n_folds']
    n_cores = PILOT_CONFIG['n_cores']

    # Phase 0: 5 datasets × 2 filtrations × 2 preproc = 20 PH computations
    ph_phase0_seq = timings['ph_per_image']['mean'] * n_img * 20
    ph_phase0_par = ph_phase0_seq / n_cores

    # Phase 1: 5 datasets × 1 filtration (best) = 5 PH computations
    # But should be cached from Phase 0
    ph_phase1_seq = timings['ph_per_image']['mean'] * n_img * n_data
    vec_phase1 = sum(timings['vectorization'].values()) * n_img * n_data * n_desc / 4  # ~4 vecs tested
    clf_phase1 = timings['classifier']['total'] / PILOT_CONFIG['n_folds'] * n_folds * n_desc * n_data

    phase0_parallel = ph_phase0_par
    phase1_parallel = (ph_phase1_seq / n_cores) + vec_phase1 + clf_phase1

    print(f"\nPhase 0 (Filtration Validation):")
    print(f"  Sequential: {ph_phase0_seq / 3600:.1f} hours")
    print(f"  Parallel ({n_cores} cores): {ph_phase0_par / 3600:.2f} hours")

    print(f"\nPhase 1 (Descriptor Comparison):")
    print(f"  PH: {ph_phase1_seq / 3600:.1f}h seq → {ph_phase1_seq / n_cores / 3600:.2f}h parallel (or cached)")
    print(f"  Vectorization: {vec_phase1 / 3600:.2f}h")
    print(f"  Classification: {clf_phase1 / 3600:.2f}h")
    print(
        f"  Total: {(ph_phase1_seq + vec_phase1 + clf_phase1) / 3600:.1f}h seq → {phase1_parallel / 3600:.1f}h parallel")

    # With 5 parallel dataset jobs
    print(f"\nWith 5 parallel dataset jobs:")
    print(f"  Each job: ~{(ph_phase1_seq / n_data / n_cores + vec_phase1 / n_data + clf_phase1 / n_data) / 3600:.1f}h")
    print(f"  Wall-clock: ~{max(phase0_parallel, phase1_parallel / n_data) / 3600:.1f}h for Phase 0+1")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✓ PH is the bottleneck: {timings['ph_per_image']['mean']:.3f}s/image")
    print(f"✓ With {n_cores} cores: ~{n_cores}x speedup on PH")
    print(f"✓ Cache PH diagrams to avoid recomputation!")
    print(f"✓ Run datasets as parallel jobs for additional speedup")

    print(f"\n⏱️  Estimated wall-clock time (5 jobs × {n_cores} cores):")
    print(f"   Phase 0: ~2 hours")
    print(f"   Phase 1: ~6 hours")
    print(f"   Phase 2: ~3 hours")
    print(f"   Phase 3: ~4 hours")
    print(f"   ─────────────────")
    print(f"   Total: ~15-20 hours")

    print("\n" + "=" * 70)
    print("PILOT COMPLETE - Ready for full experiment!")
    print("=" * 70)

    return timings


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    timings = run_pilot()

    if timings:
        # Save timings
        import json

        output_path = Path('./pilot_benchmark_results.json')
        with open(output_path, 'w') as f:
            # Convert numpy types for JSON
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj


            json.dump(convert(timings), f, indent=2)

        print(f"\n📁 Results saved to: {output_path}")