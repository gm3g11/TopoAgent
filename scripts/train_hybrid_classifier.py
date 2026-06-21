#!/usr/bin/env python3
"""Train hybrid CNN+TDA classifier for DermaMNIST.

This script trains a classifier using combined features:
- 800D persistence image (PI) features from TDA
- 512D features from pretrained ResNet18

Architecture:
    Image → [ResNet18 pretrained] → 512D CNN features
          → [TDA Pipeline]        → 800D PI features
                                  ↓
                        Concatenate: 1312D
                                  ↓
                        MLP: 1312 → 256 → 128 → 7

Expected improvement: +10-15% accuracy over TDA-only baseline.

Usage:
    python scripts/train_hybrid_classifier.py --epochs 100 --output models/
    python scripts/train_hybrid_classifier.py --weighted --focal-loss
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

# Import from existing train_classifier
from scripts.train_classifier import (
    DERMAMNIST_CLASSES,
    compute_persistence_homology,
    compute_persistence_image,
    load_dermamnist,
    FocalLoss
)


class HybridMLP(nn.Module):
    """PyTorch MLP for hybrid TDA+CNN classification.

    Architecture:
    - Input: 1312D (800D TDA + 512D CNN)
    - Hidden: 256 -> 128 -> 64 (with BatchNorm + ReLU + Dropout)
    - Output: 7 classes

    Uses BatchNorm for better training stability with high-dim input.
    """

    def __init__(
        self,
        tda_dim: int = 800,
        cnn_dim: int = 512,
        num_classes: int = 7,
        dropout: float = 0.4
    ):
        super().__init__()

        input_dim = tda_dim + cnn_dim

        # Separate feature processing before fusion
        self.tda_branch = nn.Sequential(
            nn.Linear(tda_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cnn_branch = nn.Sequential(
            nn.Linear(cnn_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),  # 128 + 128 from branches
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

        self.tda_dim = tda_dim
        self.cnn_dim = cnn_dim

    def forward(self, x):
        # Split into TDA and CNN features
        tda_features = x[:, :self.tda_dim]
        cnn_features = x[:, self.tda_dim:]

        # Process each branch
        tda_out = self.tda_branch(tda_features)
        cnn_out = self.cnn_branch(cnn_features)

        # Concatenate and classify
        combined = torch.cat([tda_out, cnn_out], dim=1)
        return self.fusion(combined)


def extract_cnn_features(
    images: np.ndarray,
    model_name: str = "resnet18",
    batch_size: int = 32,
    device: str = "cpu",
    desc: str = "Extracting CNN features"
) -> np.ndarray:
    """Extract CNN features from images using pretrained ResNet.

    Args:
        images: Array of images (N, H, W) or (N, H, W, C)
        model_name: Pretrained model name
        batch_size: Batch size for inference
        device: 'cpu' or 'cuda'
        desc: Progress bar description

    Returns:
        Feature matrix (N, feature_dim)
    """
    import torchvision.models as models
    import torchvision.transforms as transforms

    # Load model
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        feature_dim = 512
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
        feature_dim = 512
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Remove classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc=desc):
            batch_images = images[i:i+batch_size]
            batch_tensors = []

            for img in batch_images:
                # Ensure RGB
                if len(img.shape) == 2:
                    img_rgb = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 1:
                    img_rgb = np.concatenate([img] * 3, axis=-1)
                else:
                    img_rgb = img

                # Ensure uint8
                if img_rgb.dtype != np.uint8:
                    if img_rgb.max() <= 1.0:
                        img_rgb = (img_rgb * 255).astype(np.uint8)
                    else:
                        img_rgb = img_rgb.astype(np.uint8)

                tensor = transform(img_rgb)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors).to(device)
            batch_features = model(batch).squeeze(-1).squeeze(-1).cpu().numpy()
            features.append(batch_features)

    return np.vstack(features)


def extract_pi_features(
    images: np.ndarray,
    resolution: int = 20,
    sigma: float = 0.1,
    desc: str = "Extracting PI features"
) -> np.ndarray:
    """Extract persistence image features from all images.

    Args:
        images: Array of images
        resolution: PI resolution
        sigma: Gaussian bandwidth
        desc: Progress bar description

    Returns:
        Feature matrix (n_samples x feature_dim)
    """
    features = []

    for img in tqdm(images, desc=desc):
        # Compute PH
        ph_result = compute_persistence_homology(img)

        if ph_result.get("success", False):
            persistence_data = ph_result["persistence"]
            feature_vec = compute_persistence_image(persistence_data, resolution, sigma)
        else:
            # Empty features on failure
            feature_vec = [0.0] * (resolution * resolution * 2)

        features.append(feature_vec)

    return np.array(features, dtype=np.float32)


def train_hybrid_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    tda_dim: int = 800,
    cnn_dim: int = 512,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = "cpu",
    use_full_weights: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0
) -> Tuple[HybridMLP, dict]:
    """Train hybrid MLP classifier.

    Args:
        train_features: Training features (TDA + CNN concatenated)
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        tda_dim: Dimension of TDA features
        cnn_dim: Dimension of CNN features
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
        use_full_weights: Use full inverse frequency weights
        use_focal_loss: Use focal loss
        focal_gamma: Focal loss gamma

    Returns:
        Tuple of (trained model, training history)
    """
    num_classes = len(DERMAMNIST_CLASSES)

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
    model = HybridMLP(tda_dim=tda_dim, cnn_dim=cnn_dim, num_classes=num_classes)
    model = model.to(device)

    # Compute class weights
    class_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"  Class distribution: {class_counts}")

    if use_full_weights:
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes
        weight_type = "full inverse"
    else:
        class_weights = np.sqrt(1.0 / (class_counts + 1e-6))
        class_weights = class_weights / class_weights.sum() * num_classes
        weight_type = "sqrt inverse"

    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"  Class weights ({weight_type}): {class_weights.cpu().numpy().round(2)}")

    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        print(f"  Using Focal Loss (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("  Using CrossEntropyLoss")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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
    parser = argparse.ArgumentParser(description="Train hybrid CNN+TDA classifier")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=20, help="PI resolution")
    parser.add_argument("--sigma", type=float, default=0.1, help="PI sigma")
    parser.add_argument("--cnn-model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="Pretrained CNN model")
    parser.add_argument("--output", type=str, default="models/", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--weighted", action="store_true",
                        help="Use full inverse frequency class weights")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine CNN feature dimension
    cnn_dim = 512 if args.cnn_model in ["resnet18", "resnet34"] else 2048
    tda_dim = args.resolution * args.resolution * 2  # H0 + H1

    print("="*60)
    print("TopoAgent Hybrid Classifier Training (TDA + CNN)")
    print("="*60)
    print(f"Dataset: DermaMNIST (7 classes)")
    print(f"TDA Features: {tda_dim}D (PI {args.resolution}x{args.resolution} x 2)")
    print(f"CNN Features: {cnn_dim}D ({args.cnn_model})")
    print(f"Total Features: {tda_dim + cnn_dim}D")
    print("="*60)

    # Load data
    print("\n[1/5] Loading DermaMNIST...")
    train_images, train_labels = load_dermamnist("train")
    val_images, val_labels = load_dermamnist("val")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")

    # Extract TDA features
    print("\n[2/5] Extracting TDA (PI) features...")
    train_pi = extract_pi_features(
        train_images, args.resolution, args.sigma, "Train PI"
    )
    val_pi = extract_pi_features(
        val_images, args.resolution, args.sigma, "Val PI"
    )
    print(f"  TDA feature dimension: {train_pi.shape[1]}")

    # Extract CNN features
    print("\n[3/5] Extracting CNN features...")
    train_cnn = extract_cnn_features(
        train_images, args.cnn_model, args.batch_size, args.device, "Train CNN"
    )
    val_cnn = extract_cnn_features(
        val_images, args.cnn_model, args.batch_size, args.device, "Val CNN"
    )
    print(f"  CNN feature dimension: {train_cnn.shape[1]}")

    # Concatenate features
    print("\n[4/5] Combining features...")
    train_features = np.concatenate([train_pi, train_cnn], axis=1)
    val_features = np.concatenate([val_pi, val_cnn], axis=1)
    print(f"  Hybrid feature dimension: {train_features.shape[1]}")

    # Train model
    print("\n[5/5] Training hybrid MLP...")
    print(f"  Full class weights: {args.weighted}")
    print(f"  Focal loss: {args.focal_loss}")
    model, history = train_hybrid_model(
        train_features, train_labels,
        val_features, val_labels,
        tda_dim=tda_dim,
        cnn_dim=cnn_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_full_weights=args.weighted,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.gamma
    )

    # Save model
    print("\n[6/6] Saving model...")
    if args.focal_loss:
        model_name = "dermamnist_hybrid_focal.pt"
    elif args.weighted:
        model_name = "dermamnist_hybrid_weighted.pt"
    else:
        model_name = "dermamnist_hybrid.pt"
    model_path = output_dir / model_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "tda_dim": tda_dim,
        "cnn_dim": cnn_dim,
        "cnn_model": args.cnn_model,
        "num_classes": len(DERMAMNIST_CLASSES)
    }, model_path)
    print(f"  Model saved to: {model_path}")

    # Save history
    history_path = output_dir / "hybrid_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to: {history_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
