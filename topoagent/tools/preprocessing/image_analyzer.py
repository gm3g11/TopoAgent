"""Image Analyzer Tool for TopoAgent v3.

Analyzes image characteristics to guide adaptive PH configuration:
- Filtration type selection (sublevel vs superlevel)
- Noise estimation for preprocessing decisions
- PI parameter recommendations (sigma, weight function)
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class ImageAnalyzerInput(BaseModel):
    """Input schema for ImageAnalyzerTool."""
    image_array: List[List[float]] = Field(
        ..., description="2D grayscale image array (values in [0, 1])"
    )


class ImageAnalyzerTool(BaseTool):
    """Analyze image characteristics for adaptive TDA configuration.

    This tool computes image statistics that guide:
    1. Filtration type selection (sublevel vs superlevel)
    2. Noise filtering decisions
    3. Persistence image parameters (sigma, weight function)

    Key insight: Different image characteristics require different PH settings.
    - Dark features on bright background → superlevel filtration
    - Bright features on dark background → sublevel filtration
    - High noise → apply filtering first
    """

    name: str = "image_analyzer"
    description: str = (
        "Analyze image characteristics to determine optimal TDA configuration. "
        "Computes bright/dark ratios, SNR estimate, and contrast metrics. "
        "Recommends: filtration type (sublevel/superlevel), noise filtering, "
        "and persistence image parameters. "
        "Use BEFORE compute_ph to configure the pipeline adaptively. "
        "Input: image array. Output: analysis results and recommendations."
    )
    args_schema: Type[BaseModel] = ImageAnalyzerInput

    def _run(
        self,
        image_array: List[List[float]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Analyze image and provide adaptive recommendations.

        Args:
            image_array: 2D grayscale image array

        Returns:
            Dictionary with image statistics and recommendations
        """
        try:
            # Convert to numpy array
            img = np.array(image_array, dtype=np.float32)

            # Ensure normalized to [0, 1]
            if img.max() > 1.0:
                img = img / 255.0

            # === 1. INTENSITY ANALYSIS ===
            # Compute bright/dark ratios for filtration selection
            bright_threshold = 0.7
            dark_threshold = 0.3

            bright_ratio = float((img > bright_threshold).mean())
            dark_ratio = float((img < dark_threshold).mean())
            mid_ratio = float(((img >= dark_threshold) & (img <= bright_threshold)).mean())

            # === 2. SNR ESTIMATION ===
            # Estimate signal-to-noise ratio using local variance method
            snr_estimate = self._estimate_snr(img)

            # === 3. CONTRAST ANALYSIS ===
            # Michelson contrast: (I_max - I_min) / (I_max + I_min)
            if img.max() + img.min() > 0:
                contrast = float((img.max() - img.min()) / (img.max() + img.min()))
            else:
                contrast = 0.0

            # Global variance as another contrast measure
            variance = float(np.var(img))

            # === 4. EDGE/TEXTURE ANALYSIS ===
            edge_density = self._compute_edge_density(img)

            # === 5. ADAPTIVE RECOMMENDATIONS ===
            recommendations = self._generate_recommendations(
                bright_ratio=bright_ratio,
                dark_ratio=dark_ratio,
                snr_estimate=snr_estimate,
                contrast=contrast,
                variance=variance,
                edge_density=edge_density
            )

            return {
                "success": True,
                "image_statistics": {
                    "bright_ratio": round(bright_ratio, 4),
                    "dark_ratio": round(dark_ratio, 4),
                    "mid_ratio": round(mid_ratio, 4),
                    "snr_estimate": round(snr_estimate, 2),
                    "contrast": round(contrast, 4),
                    "variance": round(variance, 6),
                    "edge_density": round(edge_density, 4),
                    "mean_intensity": round(float(img.mean()), 4),
                    "std_intensity": round(float(img.std()), 4),
                    "shape": list(img.shape)
                },
                "recommendations": recommendations,
                "reasoning": self._generate_reasoning(
                    bright_ratio, dark_ratio, snr_estimate, contrast, recommendations
                )
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _estimate_snr(self, img: np.ndarray) -> float:
        """Estimate signal-to-noise ratio.

        Uses local variance method: compare smoothed image variance to noise variance.

        Args:
            img: Normalized image array

        Returns:
            SNR estimate (higher = cleaner signal)
        """
        try:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(img, sigma=1.0)
        except ImportError:
            # Fallback: simple box filter
            kernel_size = 3
            from scipy.ndimage import uniform_filter
            try:
                smoothed = uniform_filter(img, size=kernel_size)
            except ImportError:
                # Manual simple smoothing
                smoothed = img.copy()
                for i in range(1, img.shape[0] - 1):
                    for j in range(1, img.shape[1] - 1):
                        smoothed[i, j] = img[i-1:i+2, j-1:j+2].mean()

        # Noise estimate from residual
        noise = img - smoothed
        noise_var = np.var(noise)

        # Signal variance from smoothed
        signal_var = np.var(smoothed)

        # SNR ratio (protect against division by zero)
        if noise_var > 1e-10:
            snr = signal_var / noise_var
        else:
            snr = 100.0  # Very clean signal

        return float(snr)

    def _compute_edge_density(self, img: np.ndarray) -> float:
        """Compute edge density using gradient magnitude.

        Args:
            img: Normalized image array

        Returns:
            Edge density (0-1 scale)
        """
        # Compute gradients using numpy
        gy, gx = np.gradient(img)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)

        # Normalize and compute density of significant edges
        if gradient_magnitude.max() > 0:
            normalized = gradient_magnitude / gradient_magnitude.max()
            edge_threshold = 0.2
            edge_density = (normalized > edge_threshold).mean()
        else:
            edge_density = 0.0

        return float(edge_density)

    def _generate_recommendations(
        self,
        bright_ratio: float,
        dark_ratio: float,
        snr_estimate: float,
        contrast: float,
        variance: float,
        edge_density: float
    ) -> Dict[str, Any]:
        """Generate adaptive recommendations based on image analysis.

        Args:
            bright_ratio: Ratio of bright pixels (>0.7)
            dark_ratio: Ratio of dark pixels (<0.3)
            snr_estimate: Signal-to-noise ratio estimate
            contrast: Michelson contrast
            variance: Global variance
            edge_density: Edge density metric

        Returns:
            Dictionary of recommendations
        """
        recommendations = {}

        # === FILTRATION TYPE ===
        # Key decision: sublevel captures bright features, superlevel captures dark features
        # For dermoscopy: lesions (features of interest) are often dark on lighter skin background
        #
        # Logic: We want to capture the FEATURES, not the background
        # - If background is bright (high bright_ratio) → features are dark → use SUPERLEVEL
        # - If background is dark (high dark_ratio) → features are bright → use SUBLEVEL
        #
        if bright_ratio > dark_ratio + 0.1:
            # Bright background → dark features (lesions) → superlevel captures dark structures
            recommendations["filtration_type"] = "superlevel"
            recommendations["filtration_reason"] = "Bright background detected; using superlevel to capture dark features (lesions)"
        elif dark_ratio > bright_ratio + 0.1:
            # Dark background → bright features → sublevel captures bright structures
            recommendations["filtration_type"] = "sublevel"
            recommendations["filtration_reason"] = "Dark background detected; using sublevel to capture bright features"
        else:
            # Balanced → default to sublevel (more common in medical imaging)
            recommendations["filtration_type"] = "sublevel"
            recommendations["filtration_reason"] = "Balanced intensity distribution, using sublevel default"

        # === NOISE FILTERING ===
        if snr_estimate < 5:
            recommendations["noise_filter"] = "median"
            recommendations["noise_reason"] = f"Low SNR ({snr_estimate:.1f}) indicates noisy image"
        elif snr_estimate < 10:
            recommendations["noise_filter"] = "gaussian"
            recommendations["noise_reason"] = f"Moderate SNR ({snr_estimate:.1f}), light filtering recommended"
        else:
            recommendations["noise_filter"] = None
            recommendations["noise_reason"] = f"High SNR ({snr_estimate:.1f}), no filtering needed"

        # === PI SIGMA ===
        # Higher sigma for noisier/denser persistence diagrams
        if snr_estimate > 15:
            recommendations["pi_sigma"] = 0.05
            recommendations["sigma_reason"] = "Clean signal → use small sigma for sharp features"
        elif snr_estimate > 5:
            recommendations["pi_sigma"] = 0.1
            recommendations["sigma_reason"] = "Moderate signal → standard sigma"
        else:
            recommendations["pi_sigma"] = 0.2
            recommendations["sigma_reason"] = "Noisy signal → larger sigma for smoothing"

        # === PI WEIGHT FUNCTION ===
        if snr_estimate > 15 and contrast > 0.5:
            recommendations["pi_weight"] = "squared"
            recommendations["weight_reason"] = "High SNR + contrast → squared weight emphasizes persistent features"
        else:
            recommendations["pi_weight"] = "linear"
            recommendations["weight_reason"] = "Standard linear weight for balanced weighting"

        # === COMPLEXITY ESTIMATE ===
        # Based on edge density and variance
        if edge_density > 0.3 or variance > 0.05:
            recommendations["complexity"] = "high"
        elif edge_density > 0.15 or variance > 0.02:
            recommendations["complexity"] = "medium"
        else:
            recommendations["complexity"] = "low"

        return recommendations

    def _generate_reasoning(
        self,
        bright_ratio: float,
        dark_ratio: float,
        snr_estimate: float,
        contrast: float,
        recommendations: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for the recommendations.

        Args:
            bright_ratio: Ratio of bright pixels
            dark_ratio: Ratio of dark pixels
            snr_estimate: SNR estimate
            contrast: Contrast measure
            recommendations: Generated recommendations

        Returns:
            Reasoning string
        """
        lines = []

        lines.append(f"Image Analysis Summary:")
        lines.append(f"- Intensity distribution: {bright_ratio*100:.1f}% bright, {dark_ratio*100:.1f}% dark")
        lines.append(f"- Signal quality: SNR = {snr_estimate:.1f} ({'good' if snr_estimate > 10 else 'moderate' if snr_estimate > 5 else 'poor'})")
        lines.append(f"- Contrast: {contrast:.2f} ({'high' if contrast > 0.5 else 'moderate' if contrast > 0.3 else 'low'})")
        lines.append("")
        lines.append(f"Adaptive Configuration:")
        lines.append(f"- Filtration: {recommendations['filtration_type']} ({recommendations['filtration_reason']})")
        lines.append(f"- Noise filter: {recommendations['noise_filter'] or 'none'}")
        lines.append(f"- PI sigma: {recommendations['pi_sigma']}")
        lines.append(f"- PI weight: {recommendations['pi_weight']}")
        lines.append(f"- Complexity: {recommendations['complexity']}")

        return "\n".join(lines)

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
