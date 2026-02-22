"""MLP Classifier Tool for TopoAgent.

Classify using Multi-Layer Perceptron on topological features.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class MLPClassifierInput(BaseModel):
    """Input schema for MLPClassifierTool."""
    feature_vector: List[float] = Field(
        ..., description="Feature vector from topological_features or persistence_image tool"
    )
    model_path: Optional[str] = Field(
        None, description="Path to pre-trained MLP model (optional)"
    )


class MLPClassifierTool(BaseTool):
    """Classify using Multi-Layer Perceptron on topological features.

    Neural network classifier that can learn complex decision boundaries.
    Requires pre-trained model or reference data.
    """

    name: str = "mlp_classifier"
    description: str = (
        "Classify images using a neural network (MLP) on topological features. "
        "Can capture complex non-linear patterns in topological signatures. "
        "Requires pre-trained model for best results. "
        "Input: feature vector, optional model path. "
        "Output: predicted class, confidence, class probabilities."
    )
    args_schema: Type[BaseModel] = MLPClassifierInput

    # Pre-trained model
    model: Optional[Any] = None
    scaler: Optional[Any] = None
    classes: Optional[List[str]] = None

    def _run(
        self,
        feature_vector: List[float],
        model_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Classify using MLP.

        Args:
            feature_vector: Query feature vector
            model_path: Optional path to pre-trained model

        Returns:
            Classification result
        """
        try:
            query = np.array(feature_vector).reshape(1, -1)

            # Load model if path provided
            if model_path:
                self._load_model(model_path)

            # Check if model is available
            if self.model is None:
                # No model - create and train a simple one on-the-fly
                # This is just for demonstration; real usage should have pre-trained model
                return {
                    "success": True,
                    "predicted_class": "unknown",
                    "confidence": 0.0,
                    "note": "No pre-trained MLP model available. Use load_model() or provide model_path.",
                    "feature_summary": {
                        "vector_length": len(feature_vector),
                        "mean": float(np.mean(feature_vector)),
                        "std": float(np.std(feature_vector)),
                        "non_zero": int(np.sum(np.array(feature_vector) != 0))
                    }
                }

            # Normalize features
            if self.scaler:
                query_scaled = self.scaler.transform(query)
            else:
                query_scaled = query

            # Predict
            prediction = self.model.predict(query_scaled)[0]
            probabilities = self.model.predict_proba(query_scaled)[0]

            # Get class probabilities
            class_probs = {}
            if self.classes:
                for i, cls in enumerate(self.classes):
                    class_probs[cls] = float(probabilities[i])
            else:
                for i, prob in enumerate(probabilities):
                    class_probs[f"class_{i}"] = float(prob)

            confidence = float(max(probabilities) * 100)

            return {
                "success": True,
                "predicted_class": str(prediction),
                "confidence": confidence,
                "class_probabilities": class_probs,
                "method": "Multi-Layer Perceptron"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _load_model(self, path: str):
        """Load pre-trained model from file.

        Args:
            path: Path to model file
        """
        import joblib
        data = joblib.load(path)
        self.model = data.get("model")
        self.scaler = data.get("scaler")
        self.classes = data.get("classes")

    def train(
        self,
        features: List[List[float]],
        labels: List[str],
        hidden_layers: tuple = (100, 50),
        save_path: Optional[str] = None
    ):
        """Train MLP classifier on reference data.

        Args:
            features: Training feature vectors
            labels: Training labels
            hidden_layers: Hidden layer sizes
            save_path: Optional path to save model
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib

        X = np.array(features)
        y = labels

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train MLP
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=500,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.classes = list(self.model.classes_)

        # Save if path provided
        if save_path:
            joblib.dump({
                "model": self.model,
                "scaler": self.scaler,
                "classes": self.classes
            }, save_path)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
