#!/usr/bin/env python3
"""Train PyTorch MLP classifier for DermaMNIST using PI features.

This script:
1. Loads DermaMNIST train/val splits from medmnist
2. Computes persistence homology for each image using compute_ph
3. Generates persistence image (PI) feature vectors
4. Trains a PyTorch MLP classifier
5. Saves the model to models/dermamnist_pi_mlp.pt

Usage:
    python scripts/train_classifier.py --epochs 100 --output models/
    python scripts/train_classifier.py --batch-size 64 --lr 0.001
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Dataset configurations
DATASET_CONFIGS = {
    "dermamnist": {
        "classes": [
            "actinic keratosis",
            "basal cell carcinoma",
            "benign keratosis",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular lesions"
        ],
        "num_classes": 7,
        "medmnist_class": "DermaMNIST",
        "description": "Dermatoscopic images of pigmented skin lesions"
    },
    "pathmnist": {
        "classes": [
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth muscle",
            "normal colon mucosa",
            "cancer-associated stroma",
            "colorectal adenocarcinoma"
        ],
        "num_classes": 9,
        "medmnist_class": "PathMNIST",
        "description": "Colorectal cancer histology"
    },
    "bloodmnist": {
        "classes": [
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature granulocytes",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet"
        ],
        "num_classes": 8,
        "medmnist_class": "BloodMNIST",
        "description": "Blood cell microscopy images"
    },
    "retinamnist": {
        "classes": [
            "no DR",
            "mild DR",
            "moderate DR",
            "severe DR",
            "proliferative DR"
        ],
        "num_classes": 5,
        "medmnist_class": "RetinaMNIST",
        "description": "Retinal fundus images for diabetic retinopathy"
    },
    "organamnist": {
        "classes": [
            "bladder",
            "femur-left",
            "femur-right",
            "heart",
            "kidney-left",
            "kidney-right",
            "liver",
            "lung-left",
            "lung-right",
            "pancreas",
            "spleen"
        ],
        "num_classes": 11,
        "medmnist_class": "OrganAMNIST",
        "description": "Abdominal CT organ segmentation"
    }
}

# Default to DermaMNIST for backwards compatibility
DERMAMNIST_CLASSES = DATASET_CONFIGS["dermamnist"]["classes"]


class MedMNIST_MLP(nn.Module):
    """PyTorch MLP for MedMNIST classification.

    Architecture:
    - Input: 800D (H0+H1 persistence images, 20x20 each)
    - Hidden: 256 -> 128 -> 64 (with ReLU + Dropout)
    - Output: configurable number of classes

    Supports all MedMNIST datasets:
    - DermaMNIST (7 classes)
    - PathMNIST (9 classes)
    - BloodMNIST (8 classes)
    - RetinaMNIST (5 classes)
    - OrganAMNIST (11 classes)
    """

    def __init__(self, input_dim: int = 800, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# Backwards compatibility alias
DermaMNIST_MLP = MedMNIST_MLP


def compute_persistence_homology(
    image: np.ndarray,
    max_dimension: int = 1,
    filtration_type: str = "sublevel"
) -> dict:
    """Compute persistent homology using GUDHI.

    Args:
        image: 2D grayscale image array
        max_dimension: Maximum homology dimension
        filtration_type: 'sublevel' or 'superlevel' (v3 adaptive support)

    Returns:
        Persistence data dictionary
    """
    try:
        import gudhi

        # Normalize to [0, 1]
        img = image.astype(np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())

        # Apply superlevel transformation if needed (v3)
        if filtration_type == "superlevel":
            img = -img  # Negate for superlevel filtration

        # Create cubical complex
        cubical = gudhi.CubicalComplex(
            dimensions=list(img.shape),
            top_dimensional_cells=img.flatten()
        )
        cubical.compute_persistence()

        # Extract persistence pairs
        persistence_by_dim = {}
        for dim in range(max_dimension + 1):
            intervals = cubical.persistence_intervals_in_dimension(dim)
            pairs = []
            for birth, death in intervals:
                if death == float('inf'):
                    death = img.max() if filtration_type == "sublevel" else -img.min()
                # Convert back from negated values for superlevel
                if filtration_type == "superlevel":
                    birth, death = -death, -birth
                pairs.append({
                    "birth": float(abs(birth)),
                    "death": float(abs(death)),
                    "persistence": float(abs(death - birth))
                })
            persistence_by_dim[f"H{dim}"] = pairs

        return {"success": True, "persistence": persistence_by_dim, "filtration_type": filtration_type}

    except ImportError:
        # Fallback to giotto-tda
        try:
            from gtda.homology import CubicalPersistence

            img = image.astype(np.float32)
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())

            # Apply superlevel transformation if needed (v3)
            if filtration_type == "superlevel":
                img = -img

            img_batch = img.reshape(1, *img.shape)
            cp = CubicalPersistence(
                homology_dimensions=list(range(max_dimension + 1)),
                coeff=2
            )
            diagrams = cp.fit_transform(img_batch)

            persistence_by_dim = {}
            for dim in range(max_dimension + 1):
                pairs = []
                dim_diag = diagrams[0][diagrams[0][:, 2] == dim]
                for birth, death, _ in dim_diag:
                    if not np.isinf(death):
                        # Convert back from negated values for superlevel
                        if filtration_type == "superlevel":
                            birth, death = -death, -birth
                        pairs.append({
                            "birth": float(abs(birth)),
                            "death": float(abs(death)),
                            "persistence": float(abs(death - birth))
                        })
                persistence_by_dim[f"H{dim}"] = pairs

            return {"success": True, "persistence": persistence_by_dim, "filtration_type": filtration_type}

        except ImportError:
            return {"success": False, "error": "No TDA library available"}


def compute_persistence_image(
    persistence_data: dict,
    resolution: int = 20,
    sigma: float = 0.1
) -> List[float]:
    """Convert persistence diagram to persistence image feature vector.

    Args:
        persistence_data: Persistence pairs by dimension
        resolution: PI grid resolution
        sigma: Gaussian kernel bandwidth

    Returns:
        Combined feature vector (H0 + H1)
    """
    feature_vectors = {}

    for dim_key, pairs in persistence_data.items():
        if not pairs or not isinstance(pairs, list):
            feature_vectors[dim_key] = [0.0] * (resolution * resolution)
            continue

        # Convert to birth/persistence coordinates
        points = []
        for p in pairs:
            if isinstance(p, dict) and "birth" in p and "death" in p:
                birth = p["birth"]
                persistence = p["death"] - p["birth"]
                if persistence > 0:
                    points.append((birth, persistence))

        if not points:
            feature_vectors[dim_key] = [0.0] * (resolution * resolution)
            continue

        # Get data range
        births = [p[0] for p in points]
        persistences = [p[1] for p in points]

        birth_min, birth_max = min(births), max(births)
        pers_max = max(persistences)

        birth_range = birth_max - birth_min if birth_max > birth_min else 1
        pers_range = pers_max if pers_max > 0 else 1

        # Create grid
        birth_bins = np.linspace(birth_min - 0.1*birth_range, birth_max + 0.1*birth_range, resolution)
        pers_bins = np.linspace(0, pers_max + 0.1*pers_range, resolution)

        # Initialize image
        image = np.zeros((resolution, resolution))

        # Add Gaussian for each point
        for birth, persistence in points:
            weight = persistence  # Linear weight
            for i, b in enumerate(birth_bins):
                for j, p in enumerate(pers_bins):
                    dist_sq = (b - birth)**2 + (p - persistence)**2
                    gaussian = np.exp(-dist_sq / (2 * sigma**2))
                    image[j, i] += weight * gaussian

        # Normalize
        if image.max() > 0:
            image = image / image.max()

        feature_vectors[dim_key] = image.flatten().tolist()

    # Combine H0 and H1
    combined = []
    for dim in sorted(feature_vectors.keys()):
        combined.extend(feature_vectors[dim])

    return combined


def load_medmnist(
    dataset_name: str = "dermamnist",
    split: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a MedMNIST dataset.

    Args:
        dataset_name: Name of the dataset (dermamnist, pathmnist, bloodmnist, retinamnist, organamnist)
        split: 'train', 'val', or 'test'

    Returns:
        Tuple of (images, labels)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    try:
        import medmnist

        # Dynamically get the dataset class
        dataset_class = getattr(medmnist, config["medmnist_class"])

        # Download and load dataset
        dataset = dataset_class(split=split, download=True, size=28)

        images = []
        labels = []

        for img, label in dataset:
            # Convert PIL to numpy and grayscale
            img_np = np.array(img)
            if len(img_np.shape) == 3:
                img_np = np.mean(img_np, axis=2)  # RGB to grayscale
            images.append(img_np)
            labels.append(int(label[0]))

        return np.array(images), np.array(labels)

    except ImportError:
        print("medmnist not installed. Install with: pip install medmnist")
        sys.exit(1)


def load_dermamnist(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Load DermaMNIST dataset (backwards compatibility wrapper).

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        Tuple of (images, labels)
    """
    return load_medmnist("dermamnist", split)


def extract_features(
    images: np.ndarray,
    resolution: int = 20,
    sigma: float = 0.1,
    desc: str = "Extracting features",
    filtration_type: str = "sublevel"  # v3: Adaptive filtration
) -> np.ndarray:
    """Extract PI features from all images.

    Args:
        images: Array of images
        resolution: PI resolution
        sigma: Gaussian bandwidth
        desc: Progress bar description
        filtration_type: 'sublevel' or 'superlevel' (v3 adaptive)

    Returns:
        Feature matrix (n_samples x feature_dim)
    """
    features = []

    for img in tqdm(images, desc=desc):
        # Compute PH with specified filtration type (v3)
        ph_result = compute_persistence_homology(img, filtration_type=filtration_type)

        if ph_result.get("success", False):
            persistence_data = ph_result["persistence"]
            feature_vec = compute_persistence_image(persistence_data, resolution, sigma)
        else:
            # Empty features on failure
            feature_vec = [0.0] * (resolution * resolution * 2)

        features.append(feature_vec)

    return np.array(features, dtype=np.float32)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard ones.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = "cpu",
    use_full_weights: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    num_classes: Optional[int] = None
) -> Tuple[MedMNIST_MLP, dict]:
    """Train PyTorch MLP with configurable class balancing.

    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
        use_full_weights: Use full inverse frequency weights (vs sqrt)
        use_focal_loss: Use focal loss instead of cross-entropy
        focal_gamma: Gamma parameter for focal loss
        num_classes: Number of output classes (auto-detected if None)

    Returns:
        Tuple of (trained model, training history)
    """
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_features),
        torch.LongTensor(val_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = train_features.shape[1]
    # Auto-detect num_classes from labels if not provided
    if num_classes is None:
        num_classes = len(np.unique(train_labels))
    model = MedMNIST_MLP(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)

    # Compute class weights to handle imbalanced dataset
    class_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"  Class distribution: {class_counts}")

    if use_full_weights:
        # Full inverse frequency - more aggressive for minority classes
        # This gives melanoma (class 4, 115 samples) much higher weight vs nevi (class 5, 6705 samples)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes
        weight_type = "full inverse"
    else:
        # Sqrt inverse frequency - moderate balancing (original behavior)
        class_weights = np.sqrt(1.0 / (class_counts + 1e-6))
        class_weights = class_weights / class_weights.sum() * num_classes
        weight_type = "sqrt inverse"

    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"  Class weights ({weight_type}): {class_weights.cpu().numpy().round(2)}")

    # Loss function selection
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        print(f"  Using Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Using CrossEntropyLoss")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training history
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch MLP for DermaMNIST")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=20, help="PI resolution")
    parser.add_argument("--sigma", type=float, default=0.1, help="PI sigma")
    parser.add_argument("--output", type=str, default="models/", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--weighted", action="store_true",
                        help="Use full inverse frequency class weights (more aggressive for minority classes)")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of cross-entropy (better for extreme imbalance)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--dataset", type=str, default="dermamnist",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="MedMNIST dataset to train on")
    parser.add_argument("--filtration", type=str, default="sublevel",
                        choices=["sublevel", "superlevel"],
                        help="Filtration type for persistent homology (v3 adaptive)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset]
    num_classes = dataset_config["num_classes"]
    class_names = dataset_config["classes"]

    print("="*60)
    print("TopoAgent Classifier Training (v3)")
    print("="*60)
    print(f"Dataset: {args.dataset} ({num_classes} classes)")
    print(f"Description: {dataset_config['description']}")
    print(f"Filtration: {args.filtration}")  # v3
    print(f"PI Resolution: {args.resolution}x{args.resolution}")
    print(f"Expected feature dim: {args.resolution * args.resolution * 2}")
    print("="*60)

    # Load data
    print(f"\n[1/4] Loading {args.dataset}...")
    train_images, train_labels = load_medmnist(args.dataset, "train")
    val_images, val_labels = load_medmnist(args.dataset, "val")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")

    # Extract features with specified filtration type (v3)
    print(f"\n[2/4] Extracting PI features ({args.filtration} filtration)...")
    train_features = extract_features(
        train_images, args.resolution, args.sigma, "Train features",
        filtration_type=args.filtration
    )
    val_features = extract_features(
        val_images, args.resolution, args.sigma, "Val features",
        filtration_type=args.filtration
    )
    print(f"  Feature dimension: {train_features.shape[1]}")

    # Train model
    print("\n[3/4] Training PyTorch MLP...")
    print(f"  Full class weights: {args.weighted}")
    print(f"  Focal loss: {args.focal_loss}")
    model, history = train_model(
        train_features, train_labels,
        val_features, val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_full_weights=args.weighted,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.gamma,
        num_classes=num_classes
    )

    # Save model
    print("\n[4/4] Saving model...")
    # Name model based on dataset, filtration, and training config (v3)
    model_base = f"{args.dataset}_pi_mlp"

    # Add filtration type to model name (v3)
    if args.filtration != "sublevel":
        model_base = f"{model_base}_{args.filtration}"

    if args.focal_loss:
        model_name = f"{model_base}_focal.pt"
    elif args.weighted:
        model_name = f"{model_base}_weighted.pt"
    else:
        model_name = f"{model_base}.pt"
    model_path = output_dir / model_name

    # Save model with metadata (v3: includes filtration_type)
    torch.save({
        "model_state_dict": model.state_dict(),
        "dataset": args.dataset,
        "num_classes": num_classes,
        "class_names": class_names,
        "input_dim": args.resolution * args.resolution * 2,
        "weighted": args.weighted,
        "focal_loss": args.focal_loss,
        "filtration_type": args.filtration  # v3
    }, model_path)
    print(f"  Model saved to: {model_path}")

    # Save history
    history_path = output_dir / f"{args.dataset}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to: {history_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
