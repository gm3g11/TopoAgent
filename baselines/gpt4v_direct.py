#!/usr/bin/env python
"""GPT-4V Direct Image Classification Baseline.

This baseline sends images directly to GPT-4V (or GPT-4o with vision) for classification
without using any TDA tools. This demonstrates the value of topological analysis by
comparing against pure vision model performance.

Usage:
    # Single image
    python baselines/gpt4v_direct.py --image path/to/image.png --dataset dermamnist

    # Dataset evaluation
    python baselines/gpt4v_direct.py --dataset dermamnist --n-samples 100

    # Using Ollama with vision model
    python baselines/gpt4v_direct.py --dataset dermamnist --n-samples 100 --ollama --ollama-model llava
"""

import os
import sys
import json
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import Counter

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# Dataset class labels
DATASET_LABELS = {
    "dermamnist": {
        "0": "actinic keratoses",
        "1": "basal cell carcinoma",
        "2": "benign keratosis",
        "3": "dermatofibroma",
        "4": "melanoma",
        "5": "melanocytic nevi",
        "6": "vascular lesions"
    },
    "pathmnist": {
        "0": "adipose", "1": "background", "2": "debris",
        "3": "lymphocytes", "4": "mucus", "5": "smooth muscle",
        "6": "normal colon mucosa", "7": "cancer-associated stroma",
        "8": "colorectal adenocarcinoma epithelium"
    },
    "bloodmnist": {
        "0": "basophil", "1": "eosinophil", "2": "erythroblast",
        "3": "immature granulocyte", "4": "lymphocyte", "5": "monocyte",
        "6": "neutrophil", "7": "platelet"
    },
    "retinamnist": {
        "0": "no disease", "1": "mild DR", "2": "moderate DR",
        "3": "severe DR", "4": "proliferative DR"
    },
    "pneumoniamnist": {
        "0": "normal", "1": "pneumonia"
    }
}


class GPT4VDirectClassifier:
    """Direct image classification using GPT-4V without TDA tools."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        use_ollama: bool = False,
        ollama_model: str = "llava",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """Initialize GPT-4V classifier.

        Args:
            model_name: OpenAI model name (gpt-4o, gpt-4-turbo, etc.)
            api_key: OpenAI API key (uses env if not provided)
            temperature: Model temperature
            max_tokens: Max response tokens
            use_ollama: Use Ollama instead of OpenAI
            ollama_model: Ollama vision model name
            ollama_base_url: Ollama server URL
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url

        if use_ollama:
            self._init_ollama()
        else:
            self._init_openai(api_key)

    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import requests
            # Test connection
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server not responding at {self.ollama_base_url}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama: {e}")

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type for image.

        Args:
            image_path: Path to image file

        Returns:
            MIME type string
        """
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return mime_types.get(ext, "image/png")

    def _build_prompt(self, dataset: str) -> str:
        """Build classification prompt for dataset.

        Args:
            dataset: Dataset name

        Returns:
            Prompt string
        """
        labels = DATASET_LABELS.get(dataset, {})
        label_list = "\n".join([f"  {idx}: {name}" for idx, name in labels.items()])

        prompts = {
            "dermamnist": f"""Look at this image and identify which visual pattern category it belongs to.

Categories:
{label_list}

Based on the colors, textures, and shapes visible in the image, respond with ONLY the category number (0-6) and name that best matches.

Format: [number]: [name]""",

            "pathmnist": f"""You are a medical image classifier specializing in histopathology images.

Analyze this colorectal histopathology image and classify it into exactly ONE of the following tissue categories:

{label_list}

Respond with ONLY the class number (0-8) followed by the class name. Do not include any other text.

Format: [number]: [class name]""",

            "bloodmnist": f"""You are a medical image classifier specializing in blood cell microscopy images.

Analyze this blood cell microscopy image and classify it into exactly ONE of the following cell types:

{label_list}

Respond with ONLY the class number (0-7) followed by the class name. Do not include any other text.

Format: [number]: [class name]""",

            "retinamnist": f"""You are a medical image classifier specializing in retinal fundus images.

Analyze this retinal fundus image and classify it based on diabetic retinopathy severity into exactly ONE of:

{label_list}

Respond with ONLY the class number (0-4) followed by the class name. Do not include any other text.

Format: [number]: [class name]""",

            "pneumoniamnist": f"""You are a medical image classifier specializing in chest X-ray images.

Analyze this chest X-ray image and classify it into exactly ONE of:

{label_list}

Respond with ONLY the class number (0-1) followed by the class name. Do not include any other text.

Format: [number]: [class name]"""
        }

        return prompts.get(dataset, f"Classify this medical image into one of these categories:\n{label_list}\n\nRespond with only the class number and name.")

    def classify(
        self,
        image_path: str,
        dataset: str = "dermamnist"
    ) -> Dict[str, Any]:
        """Classify a single image.

        Args:
            image_path: Path to image file
            dataset: Dataset name for context

        Returns:
            Classification result dictionary
        """
        start_time = time.time()

        try:
            # Encode image
            image_base64 = self._encode_image(image_path)
            mime_type = self._get_image_mime_type(image_path)

            # Build prompt
            prompt = self._build_prompt(dataset)

            if self.use_ollama:
                response_text = self._call_ollama(image_base64, prompt)
            else:
                response_text = self._call_openai(image_base64, mime_type, prompt)

            # Parse response
            prediction, confidence = self._parse_response(response_text, dataset)

            return {
                "success": True,
                "prediction": prediction,
                "raw_prediction": response_text.strip(),
                "confidence": confidence,
                "time": time.time() - start_time,
                "model": self.ollama_model if self.use_ollama else self.model_name
            }

        except Exception as e:
            return {
                "success": False,
                "prediction": "error",
                "error": str(e),
                "time": time.time() - start_time,
                "model": self.ollama_model if self.use_ollama else self.model_name
            }

    def _call_openai(self, image_base64: str, mime_type: str, prompt: str) -> str:
        """Call OpenAI API with image.

        Args:
            image_base64: Base64 encoded image
            mime_type: Image MIME type
            prompt: Text prompt

        Returns:
            Response text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content

    def _call_ollama(self, image_base64: str, prompt: str) -> str:
        """Call Ollama API with image.

        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt

        Returns:
            Response text
        """
        import requests

        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")

        return response.json().get("response", "")

    def _parse_response(self, response: str, dataset: str) -> Tuple[int, float]:
        """Parse model response to extract class prediction.

        Args:
            response: Raw model response
            dataset: Dataset name

        Returns:
            Tuple of (predicted class index, confidence)
        """
        import re

        labels = DATASET_LABELS.get(dataset, {})
        n_classes = len(labels)

        # Try to extract class number from response
        # Pattern: number followed by colon or just number at start
        patterns = [
            r'^(\d+)\s*:',  # "4: melanoma"
            r'^class\s*(\d+)',  # "class 4"
            r'(\d+)',  # Any number
        ]

        response_clean = response.strip().lower()

        for pattern in patterns:
            match = re.search(pattern, response_clean)
            if match:
                class_idx = int(match.group(1))
                if 0 <= class_idx < n_classes:
                    # Confidence is fixed since we don't get probabilities
                    return class_idx, 0.5

        # Try to match class names
        for idx, name in labels.items():
            if name.lower() in response_clean:
                return int(idx), 0.4

        # Default to unknown
        return -1, 0.0


def evaluate_gpt4v_direct(
    dataset: str,
    n_samples: int = 100,
    model_name: str = "gpt-4o",
    seed: int = 42,
    temperature: float = 0.1,
    use_ollama: bool = False,
    ollama_model: str = "llava",
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate GPT-4V direct classification on a dataset.

    Args:
        dataset: MedMNIST dataset name
        n_samples: Number of samples
        model_name: OpenAI model name
        seed: Random seed
        temperature: Model temperature
        use_ollama: Use Ollama instead of OpenAI
        ollama_model: Ollama vision model name
        verbose: Print progress

    Returns:
        Evaluation results
    """
    start_time = time.time()

    if verbose:
        backend = f"Ollama/{ollama_model}" if use_ollama else model_name
        print(f"\n=== GPT-4V Direct Classification on {dataset} ===")
        print(f"Samples: {n_samples}, Model: {backend}")

    # Load dataset
    try:
        import medmnist
        from medmnist import INFO

        info = INFO[dataset]
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        test_dataset = DataClass(split='test', download=True)
        if verbose:
            print(f"Dataset loaded: {len(test_dataset)} samples, {n_classes} classes")

    except ImportError:
        print("Error: MedMNIST not installed")
        return {"error": "MedMNIST not installed"}

    # Create classifier
    classifier = GPT4VDirectClassifier(
        model_name=model_name,
        temperature=temperature,
        use_ollama=use_ollama,
        ollama_model=ollama_model
    )

    # Sample indices
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)

    # Create temp directory
    temp_dir = Path("temp_gpt4v_eval")
    temp_dir.mkdir(exist_ok=True)

    # Evaluate
    predictions = []
    ground_truths = []
    timings = []
    errors = []

    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        label = int(label.squeeze())

        # Save image temporarily
        img_path = temp_dir / f"sample_{idx}.png"
        if hasattr(img, 'save'):
            img.save(img_path)
        else:
            from PIL import Image
            Image.fromarray(np.array(img).squeeze()).save(img_path)

        # Classify
        result = classifier.classify(str(img_path), dataset)

        predictions.append(result.get("prediction", -1))
        ground_truths.append(label)
        timings.append(result.get("time", 0))

        if not result.get("success"):
            errors.append({"sample_idx": int(idx), "error": result.get("error", "unknown")})

        # Progress
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n_samples - i - 1)
            print(f"  Progress: {i + 1}/{n_samples} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

        # Rate limiting - add delay between requests
        time.sleep(0.5)

    # Compute metrics
    from scripts.evaluate import compute_comprehensive_metrics
    metrics = compute_comprehensive_metrics(
        predictions, ground_truths, info['label'], n_classes
    )

    total_time = time.time() - start_time

    results = {
        "method": "gpt4v_direct",
        "dataset": dataset,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "model": ollama_model if use_ollama else model_name,
        "backend": "ollama" if use_ollama else "openai",
        "temperature": temperature,
        "seed": seed,
        "metrics": metrics,
        "timing": {
            "total_seconds": float(total_time),
            "avg_per_sample": float(np.mean(timings)) if timings else 0.0,
        },
        "errors": errors,
        "n_errors": len(errors),
        "timestamp": datetime.now().isoformat()
    }

    # Cleanup
    for f in temp_dir.glob("*.png"):
        f.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    if verbose:
        print(f"\nResults:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1: {metrics['macro_f1']*100:.2f}%")
        print(f"  Total time: {total_time:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="GPT-4V Direct Classification Baseline")

    parser.add_argument("--image", "-i", type=str,
                       help="Single image path")
    parser.add_argument("--dataset", "-d", type=str, default="dermamnist",
                       choices=["dermamnist", "pathmnist", "bloodmnist",
                                "retinamnist", "pneumoniamnist"],
                       help="MedMNIST dataset")
    parser.add_argument("--n-samples", "-n", type=int, default=100,
                       help="Number of samples")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o",
                       help="OpenAI model name")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--temperature", "-t", type=float, default=0.1,
                       help="Model temperature")

    # Ollama support
    parser.add_argument("--ollama", action="store_true",
                       help="Use Ollama instead of OpenAI")
    parser.add_argument("--ollama-model", type=str, default="llava",
                       help="Ollama vision model name")

    parser.add_argument("--output", "-o", type=str,
                       help="Output JSON file")

    args = parser.parse_args()

    if args.image:
        # Single image classification
        classifier = GPT4VDirectClassifier(
            model_name=args.model,
            temperature=args.temperature,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model
        )
        result = classifier.classify(args.image, args.dataset)
        print(f"Prediction: {result}")

    else:
        # Dataset evaluation
        results = evaluate_gpt4v_direct(
            dataset=args.dataset,
            n_samples=args.n_samples,
            model_name=args.model,
            seed=args.seed,
            temperature=args.temperature,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
