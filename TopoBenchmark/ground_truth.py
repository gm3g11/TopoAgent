"""Ground Truth from Benchmark4 Results.

Loads all results/benchmark4/raw/*.json files and provides:
- (dataset, descriptor) -> accuracy lookup
- Oracle: best descriptor per dataset
- Fixed-descriptor baselines
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Path to benchmark4 raw results
PROJECT_ROOT = Path(__file__).parent.parent
RAW_RESULTS_PATH = PROJECT_ROOT / "results" / "benchmark4" / "raw"

# All 15 descriptors
ALL_DESCRIPTORS = [
    "persistence_image", "persistence_landscapes", "betti_curves",
    "persistence_silhouette", "persistence_entropy", "persistence_statistics",
    "tropical_coordinates", "persistence_codebook", "ATOL",
    "template_functions", "minkowski_functionals", "euler_characteristic_curve",
    "euler_characteristic_transform", "edge_histogram", "lbp_texture",
]

# All 26 datasets
ALL_DATASETS = [
    "BloodMNIST", "TissueMNIST", "PathMNIST", "OCTMNIST", "OrganAMNIST",
    "RetinaMNIST", "PneumoniaMNIST", "BreastMNIST", "DermaMNIST",
    "OrganCMNIST", "OrganSMNIST", "ISIC2019", "Kvasir", "BrainTumorMRI",
    "MURA", "BreakHis", "NCT_CRC_HE", "MalariaCell", "IDRiD", "PCam",
    "LC25000", "SIPaKMeD", "AML_Cytomorphology", "APTOS2019", "GasHisSDB",
    "Chaoyang",
]


@dataclass
class ResultEntry:
    """Single (dataset, descriptor) result."""
    dataset: str
    descriptor: str
    best_classifier: str
    balanced_accuracy: float
    all_classifiers: Dict[str, float] = field(default_factory=dict)
    object_type: str = ""
    color_mode: str = ""
    n_classes: int = 0


@dataclass
class GroundTruth:
    """Complete ground truth from benchmark4."""

    # Core data
    results: Dict[Tuple[str, str], ResultEntry] = field(default_factory=dict)

    # Oracle per dataset
    oracle: Dict[str, ResultEntry] = field(default_factory=dict)

    # Summary
    n_datasets: int = 0
    n_descriptors: int = 0
    n_results: int = 0
    mba: float = 0.0  # Oracle MBA
    datasets_covered: List[str] = field(default_factory=list)
    descriptors_covered: List[str] = field(default_factory=list)

    def get_accuracy(self, dataset: str, descriptor: str) -> Optional[float]:
        """Look up balanced accuracy for a (dataset, descriptor) pair."""
        entry = self.results.get((dataset, descriptor))
        return entry.balanced_accuracy if entry else None

    def get_oracle_descriptor(self, dataset: str) -> Optional[str]:
        """Get the oracle-best descriptor for a dataset."""
        entry = self.oracle.get(dataset)
        return entry.descriptor if entry else None

    def get_oracle_accuracy(self, dataset: str) -> Optional[float]:
        """Get the oracle accuracy for a dataset."""
        entry = self.oracle.get(dataset)
        return entry.balanced_accuracy if entry else None

    def get_dataset_rankings(self, dataset: str) -> List[Tuple[str, float]]:
        """Get all descriptors ranked by accuracy for a dataset."""
        rankings = []
        for (ds, desc), entry in self.results.items():
            if ds == dataset:
                rankings.append((desc, entry.balanced_accuracy))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_top_n_descriptors(self, dataset: str, n: int = 3) -> List[str]:
        """Get top-n descriptors for a dataset."""
        rankings = self.get_dataset_rankings(dataset)
        return [desc for desc, _ in rankings[:n]]

    def compute_mba_for_selections(
        self, selections: Dict[str, str]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute MBA for a set of descriptor selections.

        Args:
            selections: {dataset: descriptor} mapping

        Returns:
            (mba, per_dataset_accuracies)
        """
        accuracies = {}
        for dataset, descriptor in selections.items():
            acc = self.get_accuracy(dataset, descriptor)
            if acc is not None:
                accuracies[dataset] = acc

        if not accuracies:
            return 0.0, {}

        mba = np.mean(list(accuracies.values()))
        return float(mba), accuracies


def load_ground_truth(
    results_dir: Optional[Path] = None,
    best_classifier_strategy: str = "best_per_entry",
) -> GroundTruth:
    """Load ground truth from benchmark4 raw JSON files.

    Args:
        results_dir: Path to results/benchmark4/raw/. Uses default if None.
        best_classifier_strategy: How to pick the best classifier.
            "best_per_entry": Best classifier per (dataset, descriptor) pair.
            "tabpfn": Always use TabPFN balanced_accuracy.

    Returns:
        GroundTruth object with all results loaded.
    """
    if results_dir is None:
        results_dir = RAW_RESULTS_PATH

    gt = GroundTruth()

    # Load all JSON files
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {results_dir}")

    for fpath in json_files:
        with open(fpath) as f:
            data = json.load(f)

        dataset = data["dataset"]
        descriptor = data["descriptor"]

        # Find best classifier
        classifiers = data.get("classifiers", {})
        if not classifiers:
            continue

        all_clf_accs = {}
        for clf_name, clf_data in classifiers.items():
            ba = clf_data.get("balanced_accuracy_mean")
            if ba is not None:
                all_clf_accs[clf_name] = ba

        if not all_clf_accs:
            continue

        if best_classifier_strategy == "tabpfn" and "TabPFN" in all_clf_accs:
            best_clf = "TabPFN"
            best_acc = all_clf_accs["TabPFN"]
        else:
            best_clf = max(all_clf_accs, key=all_clf_accs.get)
            best_acc = all_clf_accs[best_clf]

        entry = ResultEntry(
            dataset=dataset,
            descriptor=descriptor,
            best_classifier=best_clf,
            balanced_accuracy=best_acc,
            all_classifiers=all_clf_accs,
            object_type=data.get("object_type", ""),
            color_mode=data.get("color_mode", ""),
            n_classes=data.get("n_classes", 0),
        )

        gt.results[(dataset, descriptor)] = entry

    # Compute oracle: best descriptor per dataset
    dataset_best = {}
    for (dataset, descriptor), entry in gt.results.items():
        if dataset not in dataset_best or entry.balanced_accuracy > dataset_best[dataset].balanced_accuracy:
            dataset_best[dataset] = entry

    gt.oracle = dataset_best
    gt.n_results = len(gt.results)
    gt.datasets_covered = sorted(set(d for d, _ in gt.results.keys()))
    gt.descriptors_covered = sorted(set(desc for _, desc in gt.results.keys()))
    gt.n_datasets = len(gt.datasets_covered)
    gt.n_descriptors = len(gt.descriptors_covered)

    # Oracle MBA
    if gt.oracle:
        gt.mba = float(np.mean([e.balanced_accuracy for e in gt.oracle.values()]))

    return gt


if __name__ == "__main__":
    gt = load_ground_truth()
    print(f"Loaded {gt.n_results} results for {gt.n_datasets} datasets, "
          f"{gt.n_descriptors} descriptors")
    print(f"Oracle MBA: {gt.mba:.4f}")
    print()

    print("Oracle per dataset:")
    for dataset in sorted(gt.oracle.keys()):
        entry = gt.oracle[dataset]
        print(f"  {dataset:25s} -> {entry.descriptor:30s} "
              f"({entry.balanced_accuracy:.4f}, {entry.best_classifier})")

    # Count how many times each descriptor is best
    print("\nBest descriptor counts:")
    desc_counts = {}
    for entry in gt.oracle.values():
        desc_counts[entry.descriptor] = desc_counts.get(entry.descriptor, 0) + 1
    for desc, count in sorted(desc_counts.items(), key=lambda x: -x[1]):
        print(f"  {desc:30s} -> {count} datasets")
