"""k-NN Classifier Tool for TopoAgent.

Classify using k-Nearest Neighbors on topological features.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class KNNClassifierInput(BaseModel):
    """Input schema for KNNClassifierTool."""
    feature_vector: List[float] = Field(
        ..., description="Feature vector from topological_features or persistence_image tool"
    )
    reference_features: Optional[List[List[float]]] = Field(
        None, description="Reference feature vectors for training (if not using pre-trained)"
    )
    reference_labels: Optional[List[str]] = Field(
        None, description="Labels for reference features"
    )
    k: int = Field(5, description="Number of neighbors to consider")


class KNNClassifierTool(BaseTool):
    """Classify using k-Nearest Neighbors on topological features.

    Simple but effective classifier for topological feature vectors.
    Works well when reference examples are available.
    """

    name: str = "knn_classifier"
    description: str = (
        "Classify images using k-Nearest Neighbors on topological features. "
        "Requires topological feature vector from previous analysis. "
        "Compares query features to reference database of labeled examples. "
        "Input: feature vector, reference features/labels (optional), k. "
        "Output: predicted class, confidence, nearest neighbors."
    )
    args_schema: Type[BaseModel] = KNNClassifierInput

    # Pre-loaded reference database (can be set externally)
    reference_db: Optional[Dict] = None

    def _run(
        self,
        feature_vector: List[float],
        reference_features: Optional[List[List[float]]] = None,
        reference_labels: Optional[List[str]] = None,
        k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Classify using k-NN.

        Args:
            feature_vector: Query feature vector
            reference_features: Optional reference features
            reference_labels: Optional reference labels
            k: Number of neighbors

        Returns:
            Classification result
        """
        try:
            query = np.array(feature_vector)

            # Get reference data
            if reference_features is not None and reference_labels is not None:
                X_ref = np.array(reference_features)
                y_ref = reference_labels
            elif self.reference_db is not None:
                X_ref = np.array(self.reference_db["features"])
                y_ref = self.reference_db["labels"]
            else:
                # No reference data - return feature analysis only
                return {
                    "success": True,
                    "predicted_class": "unknown",
                    "confidence": 0.0,
                    "note": "No reference data provided. Set reference_features/labels or load reference_db.",
                    "feature_summary": {
                        "vector_length": len(feature_vector),
                        "mean": float(np.mean(feature_vector)),
                        "std": float(np.std(feature_vector)),
                        "max": float(np.max(feature_vector)),
                        "min": float(np.min(feature_vector))
                    }
                }

            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_ref_scaled = scaler.fit_transform(X_ref)
            query_scaled = scaler.transform(query.reshape(1, -1))

            # Compute distances
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(query_scaled, X_ref_scaled)[0]

            # Get k nearest neighbors
            k = min(k, len(distances))
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]
            nearest_labels = [y_ref[i] for i in nearest_indices]

            # Vote for class
            from collections import Counter
            label_counts = Counter(nearest_labels)
            predicted_class = label_counts.most_common(1)[0][0]

            # Confidence based on vote proportion
            confidence = label_counts[predicted_class] / k * 100

            # Distance-weighted confidence
            weights = 1 / (nearest_distances + 1e-6)
            weighted_votes = {}
            for label, weight in zip(nearest_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            total_weight = sum(weighted_votes.values())
            weighted_confidence = weighted_votes[predicted_class] / total_weight * 100

            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "weighted_confidence": float(weighted_confidence),
                "k": k,
                "nearest_neighbors": [
                    {
                        "label": label,
                        "distance": float(dist),
                        "rank": i + 1
                    }
                    for i, (label, dist) in enumerate(zip(nearest_labels, nearest_distances))
                ],
                "class_votes": dict(label_counts),
                "method": "k-NN with Euclidean distance"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def load_reference_database(self, features: List[List[float]], labels: List[str]):
        """Load reference database for classification.

        Args:
            features: List of feature vectors
            labels: Corresponding labels
        """
        self.reference_db = {
            "features": features,
            "labels": labels
        }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
