"""PyTorch MLP Classifier Tool for TopoAgent.

Pre-trained PyTorch MLP classifier for DermaMNIST skin lesion classification
using Persistence Image (PI) features.
"""

from typing import Any, Dict, Optional, Type, List
import os
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


# DermaMNIST class labels (7 classes)
DERMAMNIST_CLASSES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions"
]


class PyTorchClassifierInput(BaseModel):
    """Input schema for PyTorchClassifierTool."""
    feature_vector: Optional[List[float]] = Field(
        None,
        description="Feature vector from persistence_image tool (will be auto-injected if missing)"
    )
    model_path: Optional[str] = Field(
        None,
        description="Path to pre-trained PyTorch model (.pt file)"
    )


class PyTorchClassifierTool(BaseTool):
    """Classify skin lesions using pre-trained PyTorch MLP on PI features.

    This tool uses a trained neural network classifier for accurate predictions,
    replacing LLM heuristics-based classification which had 0% accuracy.

    The MLP architecture:
    - Input: 800D (H0+H1 persistence images, 20x20 each)
    - Hidden: 256 -> 128 -> 64 (with ReLU + Dropout)
    - Output: 7 classes (DermaMNIST)
    """

    name: str = "pytorch_classifier"
    description: str = (
        "Classify skin lesions using a pre-trained PyTorch neural network. "
        "Takes PI feature vectors (800D from persistence_image tool) and outputs "
        "predicted class, confidence score, and class probabilities. "
        "MUCH more accurate than LLM heuristics. "
        "Feature vector is AUTO-INJECTED from persistence_image output."
    )
    args_schema: Type[BaseModel] = PyTorchClassifierInput

    # Cached model
    _model: Optional[Any] = None
    _model_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_default_model_path(self) -> str:
        """Get default model path."""
        # Try several locations
        possible_paths = [
            "models/dermamnist_pi_mlp.pt",
            os.path.join(os.path.dirname(__file__), "../../../models/dermamnist_pi_mlp.pt"),
            os.path.expanduser("~/.topoagent/models/dermamnist_pi_mlp.pt"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return possible_paths[0]  # Return first as default even if not exists

    def _load_model(self, path: str):
        """Load PyTorch model from file.

        Args:
            path: Path to .pt model file
        """
        try:
            import torch
            import torch.nn as nn

            # Define MLP architecture
            class DermaMNIST_MLP(nn.Module):
                def __init__(self, input_dim=800, num_classes=7):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, num_classes)
                    )

                def forward(self, x):
                    return self.layers(x)

            # Load model
            model = DermaMNIST_MLP()
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()

            self._model = model
            self._model_path = path

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")

    def _run(
        self,
        feature_vector: Optional[List[float]] = None,
        model_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Classify using PyTorch MLP.

        Args:
            feature_vector: PI feature vector (auto-injected if None)
            model_path: Optional path to model file

        Returns:
            Classification result with prediction, confidence, probabilities
        """
        try:
            import torch
            import torch.nn.functional as F

            # Check feature vector
            if feature_vector is None or len(feature_vector) == 0:
                return {
                    "success": False,
                    "error": "No feature vector provided. Run persistence_image first.",
                    "suggestion": "Call persistence_image tool before pytorch_classifier"
                }

            # Validate feature vector length
            expected_len = 800  # 20x20 x 2 dimensions (H0 + H1)
            if len(feature_vector) != expected_len:
                # Pad or truncate if necessary
                if len(feature_vector) < expected_len:
                    feature_vector = feature_vector + [0.0] * (expected_len - len(feature_vector))
                else:
                    feature_vector = feature_vector[:expected_len]

            # Load model if needed
            path = model_path or self._get_default_model_path()
            if self._model is None or self._model_path != path:
                if os.path.exists(path):
                    self._load_model(path)
                else:
                    # No trained model - provide informative error
                    return {
                        "success": False,
                        "error": f"No trained model found at {path}",
                        "suggestion": "Run 'python scripts/train_classifier.py' to train the model",
                        "feature_summary": {
                            "vector_length": len(feature_vector),
                            "mean": float(np.mean(feature_vector)),
                            "std": float(np.std(feature_vector)),
                            "non_zero": int(np.sum(np.array(feature_vector) != 0))
                        }
                    }

            # Convert to tensor
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

            # Inference
            with torch.no_grad():
                logits = self._model(x)
                probs = F.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()

            # Format class probabilities
            class_probabilities = {
                cls: round(probs[0, i].item() * 100, 2)
                for i, cls in enumerate(DERMAMNIST_CLASSES)
            }

            # Sort by probability for top predictions
            sorted_probs = sorted(
                class_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return {
                "success": True,
                "predicted_class": DERMAMNIST_CLASSES[pred_idx],
                "class_id": pred_idx,
                "confidence": round(confidence * 100, 2),
                "class_probabilities": class_probabilities,
                "probabilities": [probs[0, i].item() * 100 for i in range(len(DERMAMNIST_CLASSES))],
                "top_3_predictions": [
                    {"class": cls, "probability": prob}
                    for cls, prob in sorted_probs[:3]
                ],
                "method": "PyTorch MLP (trained on DermaMNIST)",
                "feature_dim": len(feature_vector),
                "model_path": path
            }

        except ImportError:
            return {
                "success": False,
                "error": "PyTorch not installed. Install with: pip install torch",
                "fallback": "Using sklearn MLP instead"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
