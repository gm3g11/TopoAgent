"""Ensemble Classifier Tool for TopoAgent.

Combine predictions from multiple classifiers for robust classification.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class EnsembleClassifierInput(BaseModel):
    """Input schema for EnsembleClassifierTool."""
    predictions: List[Dict[str, Any]] = Field(
        ..., description="List of predictions from individual classifiers (knn_classifier, mlp_classifier outputs)"
    )
    weights: Optional[List[float]] = Field(
        None, description="Optional weights for each classifier (must sum to 1)"
    )
    voting: str = Field(
        "soft", description="Voting method: 'hard' (majority vote) or 'soft' (weighted probabilities)"
    )


class EnsembleClassifierTool(BaseTool):
    """Combine predictions from multiple classifiers.

    Ensemble methods often improve accuracy and robustness
    by combining predictions from diverse classifiers.
    """

    name: str = "ensemble_classifier"
    description: str = (
        "Combine predictions from multiple classifiers for more robust classification. "
        "Takes outputs from knn_classifier and/or mlp_classifier tools. "
        "Methods: 'hard' voting (majority wins), 'soft' voting (weighted confidences). "
        "Input: list of classifier predictions. "
        "Output: combined prediction with aggregated confidence."
    )
    args_schema: Type[BaseModel] = EnsembleClassifierInput

    def _run(
        self,
        predictions: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        voting: str = "soft",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Combine classifier predictions.

        Args:
            predictions: List of classifier outputs
            weights: Optional weights for each classifier
            voting: Voting method ('hard' or 'soft')

        Returns:
            Combined classification result
        """
        try:
            # Filter successful predictions
            valid_predictions = [
                p for p in predictions
                if isinstance(p, dict) and p.get("success", False) and p.get("predicted_class")
            ]

            if not valid_predictions:
                return {
                    "success": False,
                    "error": "No valid predictions to combine"
                }

            n_classifiers = len(valid_predictions)

            # Set default weights
            if weights is None:
                weights = [1.0 / n_classifiers] * n_classifiers
            else:
                # Normalize weights
                total = sum(weights)
                weights = [w / total for w in weights]

            if voting == "hard":
                result = self._hard_voting(valid_predictions, weights)
            else:  # soft
                result = self._soft_voting(valid_predictions, weights)

            result["num_classifiers"] = n_classifiers
            result["voting_method"] = voting
            result["individual_predictions"] = [
                {
                    "classifier": i + 1,
                    "prediction": p.get("predicted_class"),
                    "confidence": p.get("confidence", p.get("weighted_confidence", 0)),
                    "weight": weights[i]
                }
                for i, p in enumerate(valid_predictions)
            ]

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _hard_voting(
        self,
        predictions: List[Dict],
        weights: List[float]
    ) -> Dict[str, Any]:
        """Hard voting (majority vote).

        Args:
            predictions: Classifier predictions
            weights: Classifier weights

        Returns:
            Voting result
        """
        from collections import Counter

        # Count weighted votes
        weighted_votes = {}
        for pred, weight in zip(predictions, weights):
            cls = pred.get("predicted_class")
            if cls:
                weighted_votes[cls] = weighted_votes.get(cls, 0) + weight

        # Find winner
        if not weighted_votes:
            return {
                "success": False,
                "error": "No valid class predictions"
            }

        predicted_class = max(weighted_votes.keys(), key=lambda x: weighted_votes[x])
        total_weight = sum(weighted_votes.values())
        confidence = weighted_votes[predicted_class] / total_weight * 100

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "vote_distribution": weighted_votes,
            "method": "hard_voting"
        }

    def _soft_voting(
        self,
        predictions: List[Dict],
        weights: List[float]
    ) -> Dict[str, Any]:
        """Soft voting (weighted probabilities).

        Args:
            predictions: Classifier predictions
            weights: Classifier weights

        Returns:
            Voting result
        """
        # Aggregate class probabilities
        class_scores = {}

        for pred, weight in zip(predictions, weights):
            # Try to get class probabilities
            probs = pred.get("class_probabilities", {})

            if probs:
                # Use actual probabilities
                for cls, prob in probs.items():
                    class_scores[cls] = class_scores.get(cls, 0) + weight * prob
            else:
                # Use confidence as probability for predicted class
                cls = pred.get("predicted_class")
                conf = pred.get("confidence", pred.get("weighted_confidence", 50)) / 100
                if cls:
                    class_scores[cls] = class_scores.get(cls, 0) + weight * conf

        # Normalize
        total = sum(class_scores.values())
        if total > 0:
            class_scores = {k: v / total for k, v in class_scores.items()}

        # Find winner
        if not class_scores:
            return {
                "success": False,
                "error": "No valid class scores"
            }

        predicted_class = max(class_scores.keys(), key=lambda x: class_scores[x])
        confidence = class_scores[predicted_class] * 100

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "class_scores": {k: float(v) for k, v in class_scores.items()},
            "method": "soft_voting"
        }

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
