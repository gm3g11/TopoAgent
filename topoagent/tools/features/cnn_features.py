"""CNN Feature Extraction Tool for TopoAgent.

Extracts features from pretrained CNN (ResNet18) for hybrid TDA+CNN classification.
"""

from typing import Any, Dict, Optional, Type, List, Union
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class CNNFeaturesInput(BaseModel):
    """Input schema for CNNFeaturesTool."""
    image_path: Optional[str] = Field(
        None, description="Path to input image file"
    )
    image_array: Optional[List] = Field(
        None, description="Image as nested list (2D grayscale or 3D RGB)"
    )
    model_name: str = Field(
        default="resnet18",
        description="Pretrained CNN model name (resnet18, resnet34, resnet50)"
    )


class CNNFeaturesTool(BaseTool):
    """Extract features from pretrained CNN for hybrid classification.

    Uses a pretrained ResNet model (ImageNet weights) to extract 512D features
    that capture visual patterns complementary to TDA features.

    Combined with 800D persistence image features, this gives a 1312D
    hybrid feature vector that captures both topological and visual information.

    Reference: "Deep Residual Learning for Image Recognition" (He et al., 2016)
    """

    name: str = "cnn_features"
    description: str = (
        "Extract visual features from a pretrained CNN (ResNet18). "
        "These features capture color, texture, and shape patterns that complement "
        "topological features from TDA. Useful for hybrid TDA+CNN classification. "
        "Input: image path or array. "
        "Output: 512D feature vector from ResNet18's penultimate layer."
    )
    args_schema: Type[BaseModel] = CNNFeaturesInput

    # Cached model to avoid reloading
    _model: Any = None
    _device: str = "cpu"

    def _get_model(self, model_name: str = "resnet18"):
        """Get or create the CNN model.

        Args:
            model_name: Name of the pretrained model

        Returns:
            Tuple of (model, transform)
        """
        if self._model is not None:
            return self._model

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            # Set device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load pretrained model
            if model_name == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                model = models.resnet18(weights=weights)
            elif model_name == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                model = models.resnet34(weights=weights)
            elif model_name == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                model = models.resnet50(weights=weights)
            else:
                # Default to resnet18
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                model = models.resnet18(weights=weights)

            # Remove final classification layer to get features
            # ResNet: keep everything up to avgpool
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
            model.to(self._device)

            # ImageNet preprocessing
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            self._model = model
            return model

        except ImportError as e:
            raise ImportError(f"PyTorch/torchvision required for CNN features: {e}")

    def _run(
        self,
        image_path: Optional[str] = None,
        image_array: Optional[List] = None,
        model_name: str = "resnet18",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Extract CNN features.

        Args:
            image_path: Path to input image
            image_array: Image as nested list
            model_name: Pretrained model name

        Returns:
            Dictionary with feature vector
        """
        try:
            import torch
            from PIL import Image

            # Load image
            if image_path is not None:
                img = Image.open(image_path)
                img_array = np.array(img)
            elif image_array is not None:
                img_array = np.array(image_array)
            else:
                return {
                    "success": False,
                    "error": "Either image_path or image_array must be provided"
                }

            # Ensure 3 channels (RGB)
            if len(img_array.shape) == 2:
                # Grayscale -> RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                # Single channel -> RGB
                img_array = np.concatenate([img_array] * 3, axis=-1)

            # Ensure uint8
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)

            # Get model
            model = self._get_model(model_name)

            # Preprocess
            input_tensor = self._transform(img_array)
            input_batch = input_tensor.unsqueeze(0).to(self._device)

            # Extract features
            with torch.no_grad():
                features = model(input_batch)
                features = features.squeeze().cpu().numpy()

            # Flatten to 1D
            features = features.flatten()

            return {
                "success": True,
                "feature_vector": features.tolist(),
                "feature_dim": len(features),
                "model_name": model_name,
                "device": self._device
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)


class HybridFeaturesTool(BaseTool):
    """Combine TDA and CNN features for hybrid classification.

    Concatenates persistence image features (800D) with CNN features (512D)
    to create a 1312D hybrid feature vector.
    """

    name: str = "hybrid_features"
    description: str = (
        "Combine TDA persistence image features with CNN visual features. "
        "Creates a 1312D hybrid feature vector (800D TDA + 512D CNN). "
        "This combination captures both topological structure and visual patterns. "
        "Input: PI features and image. "
        "Output: concatenated hybrid feature vector."
    )

    def _run(
        self,
        pi_features: Optional[List[float]] = None,
        cnn_features: Optional[List[float]] = None,
        image_path: Optional[str] = None,
        image_array: Optional[List] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Combine TDA and CNN features.

        Args:
            pi_features: 800D persistence image features
            cnn_features: 512D CNN features (or compute from image)
            image_path: Path to image (if CNN features not provided)
            image_array: Image array (if CNN features not provided)

        Returns:
            Dictionary with hybrid feature vector
        """
        try:
            # Get CNN features if not provided
            if cnn_features is None:
                cnn_tool = CNNFeaturesTool()
                cnn_result = cnn_tool._run(
                    image_path=image_path,
                    image_array=image_array
                )
                if not cnn_result.get("success", False):
                    return cnn_result
                cnn_features = cnn_result["feature_vector"]

            # Validate PI features
            if pi_features is None:
                return {
                    "success": False,
                    "error": "pi_features (persistence image) required"
                }

            # Concatenate features
            hybrid = list(pi_features) + list(cnn_features)

            return {
                "success": True,
                "feature_vector": hybrid,
                "feature_dim": len(hybrid),
                "pi_dim": len(pi_features),
                "cnn_dim": len(cnn_features)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
