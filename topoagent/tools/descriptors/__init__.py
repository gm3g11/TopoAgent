"""TopoAgent Descriptor Tools.

15 topology descriptors for medical image analysis (Benchmark3/4).

PH Vectorizations (11):
1. PersistenceStatisticsTool: Statistical features (28D basic)
2. PersistenceImageTool: Grid-based vectorization (800D) - from homology/
3. PersistenceLandscapesTool: Landscape representation (800D) - from vectorization/
4. BettiCurvesTool: Betti numbers over filtration (200D) - from vectorization/
5. PersistenceSilhouetteTool: Weighted silhouettes (200D) - from vectorization/
6. PersistenceEntropyTool: Shannon entropy (2D)
7. TropicalCoordinatesTool: Tropical rational functions (14D)
8. ATOLTool: Automatic Topologically-Oriented Learning (32D)
9. PersistenceCodebookTool: Bag-of-words representation (64D)
10. TemplateFunctionsTool: Tent function evaluations (50D)
11. EulerCharacteristicCurveTool: EC over filtration (100D)

Beyond PH (2):
12. MinkowskiFunctionalsTool: Area, Perimeter, EC curves (30D) - from morphology/
13. EulerCharacteristicTransformTool: Directional EC (640D)

Baselines (2):
14. LBPTextureTool: Local Binary Pattern histogram (54D)
15. EdgeHistogramTool: Edge orientation histogram (80D)

Also includes PH computation utilities (ph_computation.py).
"""

# Descriptors implemented in this module
from .persistence_statistics import PersistenceStatisticsTool
from .persistence_entropy import PersistenceEntropyTool
from .carlsson_coordinates import CarlssonCoordinatesTool
from .tropical_coordinates import TropicalCoordinatesTool
from .atol_descriptor import ATOLTool
from .persistence_codebook import PersistenceCodebookTool
from .euler_characteristic_curve import EulerCharacteristicCurveTool
from .lbp_texture import LBPTextureTool
from .ph_computation import compute_ph_fast, PHCache

from .template_functions import TemplateFunctionsTool
from .euler_characteristic_transform import EulerCharacteristicTransformTool
from .edge_histogram import EdgeHistogramTool

# Re-export existing tools that are also descriptors
from ..homology import PersistenceImageTool
from ..vectorization import (
    BettiCurvesTool,
    PersistenceLandscapesTool,
    PersistenceSilhouetteTool
)
from ..morphology import MinkowskiFunctionalsTool

__all__ = [
    # PH computation utilities
    "compute_ph_fast",
    "PHCache",
    # PH Vectorizations (11)
    "PersistenceStatisticsTool",
    "PersistenceImageTool",
    "PersistenceLandscapesTool",
    "BettiCurvesTool",
    "PersistenceSilhouetteTool",
    "PersistenceEntropyTool",
    "CarlssonCoordinatesTool",
    "TropicalCoordinatesTool",
    "ATOLTool",
    "PersistenceCodebookTool",
    "TemplateFunctionsTool",
    "EulerCharacteristicCurveTool",
    # Beyond PH (2)
    "MinkowskiFunctionalsTool",
    "EulerCharacteristicTransformTool",
    # Baselines (2)
    "LBPTextureTool",
    "EdgeHistogramTool",
]


def get_all_descriptors():
    """Get instances of all 15 descriptor tools (Benchmark3/4).

    Returns:
        Dictionary of descriptor_name -> tool instance
    """
    return {
        # PH Vectorizations (11)
        "persistence_statistics": PersistenceStatisticsTool(),
        "persistence_image": PersistenceImageTool(),
        "persistence_landscapes": PersistenceLandscapesTool(),
        "betti_curves": BettiCurvesTool(),
        "persistence_silhouette": PersistenceSilhouetteTool(),
        "persistence_entropy": PersistenceEntropyTool(),
        "tropical_coordinates": TropicalCoordinatesTool(),
        "ATOL": ATOLTool(),
        "persistence_codebook": PersistenceCodebookTool(),
        "template_functions": TemplateFunctionsTool(),
        "euler_characteristic_curve": EulerCharacteristicCurveTool(),
        # Beyond PH (2)
        "minkowski_functionals": MinkowskiFunctionalsTool(),
        "euler_characteristic_transform": EulerCharacteristicTransformTool(),
        # Baselines (2)
        "lbp_texture": LBPTextureTool(),
        "edge_histogram": EdgeHistogramTool(),
    }


# Expected output dimensions (Benchmark3 configs)
EXPECTED_DIMS = {
    'persistence_statistics': 28,       # 14 stats × 2 hom dims (basic)
    'persistence_image': 800,           # 2 × 20 × 20
    'persistence_landscapes': 400,      # 4 layers × 100 bins (combine_dims)
    'betti_curves': 200,                # 2 × 100
    'persistence_silhouette': 200,      # 2 × 100
    'persistence_entropy': 200,         # 2 × 100 (vector mode)
    'tropical_coordinates': 40,         # 5 terms × 4 coords × 2 dims
    'atol': 32,                         # 16 clusters × 2 dims
    'persistence_codebook': 64,         # 32 codewords × 2 dims
    'template_functions': 50,           # 25 templates × 2 dims
    'euler_characteristic_curve': 100,  # 100 thresholds
    'minkowski_functionals': 30,        # 3 functionals × 10 thresholds
    'euler_characteristic_transform': 640,  # 32 dirs × 20 heights
    'lbp_texture': 54,                  # multi-scale: 10+18+26
    'edge_histogram': 80,              # 8 orientations × 10 cells
}
