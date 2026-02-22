"""TopoBenchmark: Agent Benchmark for Adaptive TDA Descriptor Selection.

Evaluates whether TopoAgent can select the right TDA descriptor for each
medical image dataset, measuring both process quality (descriptor selection
accuracy) and outcome quality (mean balanced accuracy).

Ground truth from benchmark4: 26 datasets x 15 descriptors x 6 classifiers.
"""

from .ground_truth import GroundTruth, load_ground_truth
from .config import DATASET_DESCRIPTIONS, PROTOCOL1_CONFIG, PROTOCOL2_CONFIG
from .metrics import compute_mba, compute_dsa, compute_regret
