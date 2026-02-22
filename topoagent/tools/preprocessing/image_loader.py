"""Image Loader Tool for TopoAgent.

Loads and normalizes medical images from various formats (DICOM, PNG, JPEG).
"""

from typing import Any, Dict, Optional, Type, Tuple, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class ImageLoaderInput(BaseModel):
    """Input schema for ImageLoaderTool."""
    image_path: str = Field(..., description="Path to the medical image file")
    normalize: bool = Field(True, description="Whether to normalize pixel values to [0, 1]")
    grayscale: bool = Field(True, description="Whether to convert to grayscale")
    target_size: Optional[List[int]] = Field(None, description="Optional target size as [height, width]")


class ImageLoaderTool(BaseTool):
    """Load and normalize medical images for topological analysis.

    Supports formats: DICOM, PNG, JPEG, TIFF, BMP
    Automatically handles different bit depths and color spaces.
    """

    name: str = "image_loader"
    description: str = (
        "Load and preprocess medical images for topological analysis. "
        "Supports DICOM, PNG, JPEG, and other common formats. "
        "Use this tool first when analyzing any medical image. "
        "Input: image path, optional normalization and size parameters. "
        "Output: numpy array of pixel values, image metadata."
    )
    args_schema: Type[BaseModel] = ImageLoaderInput

    def _run(
        self,
        image_path: str,
        normalize: bool = True,
        grayscale: bool = True,
        target_size: Optional[List[int]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Load and preprocess the image.

        Args:
            image_path: Path to image file
            normalize: Normalize to [0, 1]
            grayscale: Convert to grayscale
            target_size: Optional resize target

        Returns:
            Dictionary with image array and metadata
        """
        try:
            from PIL import Image
            import os

            # Check file exists
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            # Load image based on format
            if image_path.lower().endswith('.dcm'):
                # DICOM format
                image_array, metadata = self._load_dicom(image_path)
            else:
                # Standard image format
                image_array, metadata = self._load_standard(image_path)

            # Convert to grayscale if needed
            if grayscale and len(image_array.shape) == 3:
                image_array = np.mean(image_array, axis=2)

            # Resize if target size specified
            if target_size is not None:
                from skimage.transform import resize
                image_array = resize(image_array, target_size, preserve_range=True)

            # Normalize to [0, 1]
            if normalize:
                img_min, img_max = image_array.min(), image_array.max()
                if img_max > img_min:
                    image_array = (image_array - img_min) / (img_max - img_min)
                else:
                    image_array = np.zeros_like(image_array)

            return {
                "success": True,
                "image_array": image_array.tolist(),
                "shape": image_array.shape,
                "dtype": str(image_array.dtype),
                "metadata": metadata,
                "normalized": normalize,
                "grayscale": grayscale
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _load_dicom(self, path: str) -> Tuple[np.ndarray, Dict]:
        """Load DICOM file.

        Args:
            path: Path to DICOM file

        Returns:
            Tuple of (image array, metadata dict)
        """
        try:
            import pydicom
            ds = pydicom.dcmread(path)
            image_array = ds.pixel_array.astype(np.float32)

            metadata = {
                "format": "DICOM",
                "modality": str(ds.get("Modality", "Unknown")),
                "patient_id": str(ds.get("PatientID", "Unknown")),
                "study_date": str(ds.get("StudyDate", "Unknown")),
                "bits_stored": int(ds.get("BitsStored", 0)),
                "rows": int(ds.get("Rows", 0)),
                "columns": int(ds.get("Columns", 0))
            }
            return image_array, metadata
        except ImportError:
            # Fallback if pydicom not available
            from PIL import Image
            img = Image.open(path)
            return np.array(img, dtype=np.float32), {"format": "DICOM (fallback)"}

    def _load_standard(self, path: str) -> Tuple[np.ndarray, Dict]:
        """Load standard image format (PNG, JPEG, etc).

        Args:
            path: Path to image file

        Returns:
            Tuple of (image array, metadata dict)
        """
        from PIL import Image
        import os

        img = Image.open(path)
        image_array = np.array(img, dtype=np.float32)

        metadata = {
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "filename": os.path.basename(path)
        }

        return image_array, metadata

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
