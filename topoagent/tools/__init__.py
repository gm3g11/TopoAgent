"""TopoAgent TDA Tools.

37+ specialized tools for Topological Data Analysis of medical images:

DESCRIPTORS MODULE (13 descriptors for experiments):
- See topoagent.tools.descriptors for all 13 experiment descriptors
- Includes: persistence_statistics, persistence_entropy, carlsson_coordinates,
  tropical_coordinates, atol, persistence_codebook, euler_characteristic_curve,
  lbp_texture, persistence_image, betti_curves, persistence_landscapes,
  persistence_silhouette, minkowski_functionals

STANDARD TOOLS (29):

Preprocessing (4):
- ImageLoaderTool: Load and normalize images
- BinarizationTool: Adaptive thresholding
- NoiseFilterTool: Noise reduction
- ImageAnalyzerTool: Analyze image for adaptive TDA config (v3)

Filtration (3):
- SublevelFiltrationTool: For bright features
- SuperlevelFiltrationTool: For dark features
- CubicalComplexTool: For structured grid data

Homology (3):
- ComputePHTool: Compute persistent homology
- PersistenceDiagramTool: Generate and analyze PD
- PersistenceImageTool: Convert PD to vectors

Features (3):
- TopologicalFeaturesTool: Extract statistics
- WassersteinDistanceTool: Compare diagrams
- BottleneckDistanceTool: Alternative comparison

Classification (4):
- KNNClassifierTool: k-NN classifier
- MLPClassifierTool: Neural network classifier (sklearn)
- EnsembleClassifierTool: Combined prediction
- PyTorchClassifierTool: Pre-trained PyTorch MLP for DermaMNIST

Invariants (2):
- EulerCharacteristicTool: Compute Euler characteristic
- TotalPersistenceStatsTool: Lifespan statistics

Vectorization (3):
- BettiCurvesTool: Betti numbers over filtration
- PersistenceLandscapesTool: Landscape vectorization
- PersistenceSilhouetteTool: Weighted silhouettes

Morphology (3):
- BettiRatiosTool: Betti number ratios
- MinkowskiFunctionalsTool: Morphological descriptors
- AnisotropicMFTool: Directional analysis

Texture (2):
- FractalDimensionTool: Box-counting dimension
- LacunarityTool: Texture heterogeneity

Advanced (2):
- WeightedECTTool: Euler Characteristic Transform
- PersistentLaplacianTool: Spectral persistent homology
"""

# Preprocessing tools
from .preprocessing import (
    ImageLoaderTool,
    BinarizationTool,
    NoiseFilterTool,
    ImageAnalyzerTool  # v3: Adaptive analysis
)

# Filtration tools
from .filtration import (
    SublevelFiltrationTool,
    SuperlevelFiltrationTool,
    CubicalComplexTool
)

# Homology tools
from .homology import (
    ComputePHTool,
    PersistenceDiagramTool,
    PersistenceImageTool
)

# Feature tools
from .features import (
    TopologicalFeaturesTool,
    WassersteinDistanceTool,
    BottleneckDistanceTool
)

# Classification tools
from .classification import (
    KNNClassifierTool,
    MLPClassifierTool,
    EnsembleClassifierTool,
    PyTorchClassifierTool
)

# Invariants tools (NEW)
from .invariants import (
    EulerCharacteristicTool,
    TotalPersistenceStatsTool
)

# Vectorization tools (NEW)
from .vectorization import (
    BettiCurvesTool,
    PersistenceLandscapesTool,
    PersistenceSilhouetteTool
)

# Morphology tools (NEW)
from .morphology import (
    BettiRatiosTool,
    MinkowskiFunctionalsTool,
    AnisotropicMFTool
)

# Texture tools (NEW)
from .texture import (
    FractalDimensionTool,
    LacunarityTool
)

# Advanced tools (NEW)
from .advanced import (
    WeightedECTTool,
    PersistentLaplacianTool
)

# Descriptors module (for experiments)
from . import descriptors

__all__ = [
    # Preprocessing
    "ImageLoaderTool",
    "BinarizationTool",
    "NoiseFilterTool",
    "ImageAnalyzerTool",  # v3: Adaptive analysis
    # Filtration
    "SublevelFiltrationTool",
    "SuperlevelFiltrationTool",
    "CubicalComplexTool",
    # Homology
    "ComputePHTool",
    "PersistenceDiagramTool",
    "PersistenceImageTool",
    # Features
    "TopologicalFeaturesTool",
    "WassersteinDistanceTool",
    "BottleneckDistanceTool",
    # Classification
    "KNNClassifierTool",
    "MLPClassifierTool",
    "EnsembleClassifierTool",
    "PyTorchClassifierTool",
    # Invariants (NEW)
    "EulerCharacteristicTool",
    "TotalPersistenceStatsTool",
    # Vectorization (NEW)
    "BettiCurvesTool",
    "PersistenceLandscapesTool",
    "PersistenceSilhouetteTool",
    # Morphology (NEW)
    "BettiRatiosTool",
    "MinkowskiFunctionalsTool",
    "AnisotropicMFTool",
    # Texture (NEW)
    "FractalDimensionTool",
    "LacunarityTool",
    # Advanced (NEW)
    "WeightedECTTool",
    "PersistentLaplacianTool",
    # Descriptors module
    "descriptors"
]


def get_all_tools():
    """Get instances of all TDA tools.

    Returns:
        Dictionary of tool_name -> tool instance
    """
    return {
        # Preprocessing
        "image_loader": ImageLoaderTool(),
        "binarization": BinarizationTool(),
        "noise_filter": NoiseFilterTool(),
        "image_analyzer": ImageAnalyzerTool(),  # v3: Adaptive analysis
        # Filtration
        "sublevel_filtration": SublevelFiltrationTool(),
        "superlevel_filtration": SuperlevelFiltrationTool(),
        "cubical_complex": CubicalComplexTool(),
        # Homology
        "compute_ph": ComputePHTool(),
        "persistence_diagram": PersistenceDiagramTool(),
        "persistence_image": PersistenceImageTool(),
        # Features
        "topological_features": TopologicalFeaturesTool(),
        "wasserstein_distance": WassersteinDistanceTool(),
        "bottleneck_distance": BottleneckDistanceTool(),
        # Classification
        "knn_classifier": KNNClassifierTool(),
        "mlp_classifier": MLPClassifierTool(),
        "ensemble_classifier": EnsembleClassifierTool(),
        "pytorch_classifier": PyTorchClassifierTool(),
        # Invariants (NEW)
        "euler_characteristic": EulerCharacteristicTool(),
        "total_persistence_stats": TotalPersistenceStatsTool(),
        # Vectorization (NEW)
        "betti_curves": BettiCurvesTool(),
        "persistence_landscapes": PersistenceLandscapesTool(),
        "persistence_silhouette": PersistenceSilhouetteTool(),
        # Morphology (NEW)
        "betti_ratios": BettiRatiosTool(),
        "minkowski_functionals": MinkowskiFunctionalsTool(),
        "anisotropic_mf": AnisotropicMFTool(),
        # Texture (NEW)
        "fractal_dimension": FractalDimensionTool(),
        "lacunarity": LacunarityTool(),
        # Advanced (NEW)
        "weighted_ect": WeightedECTTool(),
        "persistent_laplacian": PersistentLaplacianTool()
    }
