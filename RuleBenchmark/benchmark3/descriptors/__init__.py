"""Benchmark3 descriptor tools (fixed copies + re-exports).

Fixed local copies (5 tools with code changes):
- persistence_image: Removed per-image max-normalization
- betti_curves: Changed normalize default to False
- tropical_coordinates: Changed PAD_VALUE from -1.0 to 0.0
- atol_descriptor: Fixed numpy fallback sigma for 2D arrays
- persistence_statistics: Fixed empty-case n_stats counts
"""

# Fixed local copies (5 tools with code changes)
from .persistence_image import PersistenceImageTool
from .betti_curves import BettiCurvesTool
from .tropical_coordinates import TropicalCoordinatesTool
from .atol_descriptor import ATOLTool
from .persistence_statistics import PersistenceStatisticsTool

# Unchanged tools re-exported from originals
from topoagent.tools.descriptors import (
    PersistenceEntropyTool,
    PersistenceCodebookTool,
    EulerCharacteristicCurveTool,
    TemplateFunctionsTool,
    LBPTextureTool,
    EdgeHistogramTool,
    EulerCharacteristicTransformTool,
)
# Optional extras (may not exist if removed from topoagent)
try:
    from topoagent.tools.descriptors import (
        CarlssonCoordinatesTool,
        ComplexPolynomialTool,
        HeatKernelTool,
        TopologicalVectorTool,
        PersistenceLengthsTool,
    )
except ImportError:
    pass
from topoagent.tools.vectorization import (
    PersistenceLandscapesTool,
    PersistenceSilhouetteTool,
)
from topoagent.tools.morphology import MinkowskiFunctionalsTool
