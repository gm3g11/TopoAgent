# TopoAgent Tools Reference

Complete reference for all 15 TDA tools.

## Tool Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Preprocessing | 3 | Prepare images for TDA |
| Filtration | 3 | Create topological filtrations |
| Homology | 3 | Compute persistent homology |
| Features | 3 | Extract/compare features |
| Classification | 3 | Classify using features |

---

## Preprocessing Tools

### 1. ImageLoaderTool

**Name:** `image_loader`

**Description:** Load and preprocess medical images for topological analysis.

**When to Use:**
- First tool in any analysis pipeline
- Loading DICOM, PNG, JPEG, TIFF images
- When you need normalized grayscale data

**Input:**
```python
{
    "image_path": str,           # Path to image file
    "normalize": bool = True,    # Normalize to [0, 1]
    "grayscale": bool = True,    # Convert to grayscale
    "target_size": Optional[Tuple[int, int]] = None  # Resize
}
```

**Output:**
```python
{
    "success": bool,
    "image_array": List[List[float]],  # 2D array
    "shape": Tuple[int, int],
    "metadata": Dict  # Format-specific metadata
}
```

---

### 2. BinarizationTool

**Name:** `binarization`

**Description:** Convert grayscale images to binary using adaptive thresholding.

**When to Use:**
- Before computing binary topology
- When separating foreground/background
- For lesion boundary detection

**Methods:**
- `otsu`: Automatic optimal threshold
- `adaptive_mean`: Local mean thresholding
- `adaptive_gaussian`: Local Gaussian thresholding
- `manual`: User-specified threshold

**Input:**
```python
{
    "image_array": List[List[float]],
    "method": str = "otsu",
    "threshold": Optional[float] = None,  # For manual
    "block_size": int = 11,  # For adaptive
    "c_value": float = 2     # For adaptive
}
```

**Output:**
```python
{
    "success": bool,
    "binary_image": List[List[float]],
    "threshold_value": float,
    "foreground_ratio": float
}
```

---

### 3. NoiseFilterTool

**Name:** `noise_filter`

**Description:** Apply noise filtering to reduce topological artifacts.

**When to Use:**
- Noisy images causing spurious features
- Before filtration on low-quality images
- Salt-and-pepper noise removal

**Methods:**
- `gaussian`: Smooth edges, general denoising
- `median`: Preserve edges, remove impulse noise
- `bilateral`: Edge-preserving smoothing

**Input:**
```python
{
    "image_array": List[List[float]],
    "method": str = "gaussian",
    "sigma": float = 1.0,      # For Gaussian
    "kernel_size": int = 3     # For median
}
```

---

## Filtration Tools

### 4. SublevelFiltrationTool

**Name:** `sublevel_filtration`

**Description:** Create sublevel set filtration for persistent homology.

**When to Use:**
- **Bright features on dark background**
- Skin lesions, lung nodules, tumors
- Objects that appear lighter than surroundings

**How it Works:**
- Tracks regions where pixel value ≤ threshold
- As threshold increases, regions grow and merge
- Captures "birth" when feature appears, "death" when it merges

**Input:**
```python
{
    "image_array": List[List[float]],  # Normalized [0,1]
    "num_thresholds": int = 100
}
```

---

### 5. SuperlevelFiltrationTool

**Name:** `superlevel_filtration`

**Description:** Create superlevel set filtration for persistent homology.

**When to Use:**
- **Dark features on bright background**
- Blood vessels, cavities, airways
- Objects that appear darker than surroundings

**How it Works:**
- Tracks regions where pixel value ≥ threshold
- As threshold decreases, regions grow and merge
- Internally converts to sublevel by negating image

**Input:**
```python
{
    "image_array": List[List[float]],
    "num_thresholds": int = 100
}
```

---

### 6. CubicalComplexTool

**Name:** `cubical_complex`

**Description:** Build cubical complex for structured grid data.

**When to Use:**
- 2D or 3D medical images
- When pixel/voxel structure matters
- CT scans, MRI volumes

**Input:**
```python
{
    "image_array": Union[List[List], List[List[List]]],  # 2D or 3D
    "filtration_type": str = "sublevel"  # or "superlevel"
}
```

**Output:**
```python
{
    "success": bool,
    "complex_type": "cubical",
    "dimensions": int,  # 2 or 3
    "persistence_pairs": List[Dict],  # If GUDHI available
    "betti_numbers": List[int]
}
```

---

## Homology Tools

### 7. ComputePHTool

**Name:** `compute_ph`

**Description:** Compute persistent homology (H0, H1, H2).

**What it Captures:**
- **H0**: Connected components (distinct regions/objects)
- **H1**: Loops/holes (ring structures, boundaries)
- **H2**: Voids (3D cavities)

**When to Use:**
- After filtration to get topological features
- Core step in any TDA pipeline

**Input:**
```python
{
    "image_array": List[List[float]],
    "filtration_type": str = "sublevel",
    "max_dimension": int = 1  # 0, 1, or 2
}
```

**Output:**
```python
{
    "success": bool,
    "persistence": {
        "H0": [{"birth": float, "death": float, "persistence": float}, ...],
        "H1": [...],
    },
    "statistics": {
        "H0": {"count": int, "total_persistence": float, ...},
        "H1": {...}
    }
}
```

---

### 8. PersistenceDiagramTool

**Name:** `persistence_diagram`

**Description:** Generate and analyze persistence diagrams.

**What it Shows:**
- Points (birth, death) for each topological feature
- Points far from diagonal = significant features
- Points near diagonal = noise

**When to Use:**
- Visualizing topological structure
- Identifying significant features
- Computing diagram statistics

**Input:**
```python
{
    "persistence_data": Dict[str, List[Dict]],  # From compute_ph
    "output_path": Optional[str] = None,  # Save visualization
    "title": str = "Persistence Diagram"
}
```

**Output:**
```python
{
    "success": bool,
    "diagram_analysis": {
        "H0": {
            "num_features": int,
            "significant_features": {"count": int, "top_5": [...]},
            "persistence_entropy": float
        },
        ...
    },
    "interpretation": str  # Human-readable
}
```

---

### 9. PersistenceImageTool

**Name:** `persistence_image`

**Description:** Convert persistence diagrams to fixed-size vectors.

**Why Use It:**
- Persistence diagrams have variable size
- ML classifiers need fixed-size input
- Stable representation for comparison

**Input:**
```python
{
    "persistence_data": Dict[str, List[Dict]],
    "resolution": int = 20,           # NxN grid
    "sigma": float = 0.1,             # Gaussian bandwidth
    "weight_function": str = "linear"  # linear, squared, const
}
```

**Output:**
```python
{
    "success": bool,
    "images": {"H0": [[...]], "H1": [[...]]},  # 2D arrays
    "feature_vectors": {"H0": [...], "H1": [...]},  # Flattened
    "combined_vector": [...],  # All dimensions concatenated
    "vector_length": int
}
```

---

## Feature Tools

### 10. TopologicalFeaturesTool

**Name:** `topological_features`

**Description:** Extract statistical features from persistence diagrams.

**Features Extracted (per dimension):**
- Count, sum, mean, std, min, max, median of persistence
- 25th/75th percentile, IQR
- Persistence entropy
- Amplitude
- Birth/death mean, midpoint mean

**Cross-dimension features:**
- Total persistence, total count
- H0/H1 ratio
- Entropy difference

**Input:**
```python
{
    "persistence_data": Dict[str, List[Dict]]
}
```

**Output:**
```python
{
    "success": bool,
    "features_by_dimension": {"H0": {...}, "H1": {...}},
    "feature_vector": [float, ...],  # Ready for classifier
    "feature_names": [str, ...],     # Feature labels
    "num_features": int
}
```

---

### 11. WassersteinDistanceTool

**Name:** `wasserstein_distance`

**Description:** Compare persistence diagrams using Wasserstein distance.

**What it Measures:**
- Optimal transport cost between two diagrams
- Lower distance = more similar topological structure

**When to Use:**
- Comparing image to reference patterns
- Measuring similarity between samples
- Template matching

**Input:**
```python
{
    "diagram1": Dict[str, List[Dict]],
    "diagram2": Dict[str, List[Dict]],
    "p": int = 2  # Order (1 or 2)
}
```

**Output:**
```python
{
    "success": bool,
    "distances_by_dimension": {"H0": float, "H1": float},
    "total_distance": float,
    "interpretation": str
}
```

---

### 12. BottleneckDistanceTool

**Name:** `bottleneck_distance`

**Description:** Compare persistence diagrams using bottleneck distance.

**How it Differs from Wasserstein:**
- Uses max (infinity norm) instead of sum
- More sensitive to outliers
- Better for detecting presence/absence of features

**Input:**
```python
{
    "diagram1": Dict[str, List[Dict]],
    "diagram2": Dict[str, List[Dict]]
}
```

---

## Classification Tools

### 13. KNNClassifierTool

**Name:** `knn_classifier`

**Description:** k-Nearest Neighbors on topological features.

**When to Use:**
- Simple, interpretable classification
- When reference database is available
- Comparing to known patterns

**Input:**
```python
{
    "feature_vector": List[float],
    "reference_features": Optional[List[List[float]]] = None,
    "reference_labels": Optional[List[str]] = None,
    "k": int = 5
}
```

**Output:**
```python
{
    "success": bool,
    "predicted_class": str,
    "confidence": float,
    "weighted_confidence": float,
    "nearest_neighbors": [{"label": str, "distance": float, "rank": int}, ...],
    "class_votes": Dict[str, int]
}
```

---

### 14. MLPClassifierTool

**Name:** `mlp_classifier`

**Description:** Neural network classifier on topological features.

**When to Use:**
- Complex non-linear decision boundaries
- Pre-trained model available
- Large reference datasets

**Input:**
```python
{
    "feature_vector": List[float],
    "model_path": Optional[str] = None  # Pre-trained model
}
```

**Output:**
```python
{
    "success": bool,
    "predicted_class": str,
    "confidence": float,
    "class_probabilities": Dict[str, float]
}
```

---

### 15. EnsembleClassifierTool

**Name:** `ensemble_classifier`

**Description:** Combine predictions from multiple classifiers.

**Voting Methods:**
- `hard`: Majority vote (weighted by classifier weight)
- `soft`: Weighted average of probabilities

**When to Use:**
- Improving robustness
- Combining KNN and MLP predictions
- Reducing variance

**Input:**
```python
{
    "predictions": List[Dict],  # Outputs from knn/mlp classifiers
    "weights": Optional[List[float]] = None,
    "voting": str = "soft"  # hard or soft
}
```

**Output:**
```python
{
    "success": bool,
    "predicted_class": str,
    "confidence": float,
    "individual_predictions": [...],
    "class_scores": Dict[str, float]  # For soft voting
}
```

---

## Typical Pipeline

```python
# 1. Load image
img = image_loader(image_path="scan.png")

# 2. Preprocess (optional)
filtered = noise_filter(image_array=img["image_array"], method="gaussian")

# 3. Compute persistent homology
ph = compute_ph(image_array=filtered["filtered_image"], filtration_type="sublevel")

# 4. Extract features
features = topological_features(persistence_data=ph["persistence"])

# 5. Classify
result = knn_classifier(feature_vector=features["feature_vector"], k=5)
```
