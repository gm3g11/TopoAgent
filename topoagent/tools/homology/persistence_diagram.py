"""Persistence Diagram Tool for TopoAgent.

Generate and visualize persistence diagrams from topological data.
"""

from typing import Any, Dict, Optional, Type, List
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


class PersistenceDiagramInput(BaseModel):
    """Input schema for PersistenceDiagramTool."""
    persistence_data: Dict[str, List[Dict[str, float]]] = Field(
        ..., description="Persistence data from compute_ph tool (format: {'H0': [...], 'H1': [...]})"
    )
    output_path: Optional[str] = Field(None, description="Optional path to save diagram visualization")
    title: str = Field("Persistence Diagram", description="Title for the diagram")


class PersistenceDiagramTool(BaseTool):
    """Generate and analyze persistence diagrams.

    Persistence diagrams visualize birth-death pairs of topological features.
    Points far from the diagonal represent significant/persistent features.
    """

    name: str = "persistence_diagram"
    description: str = (
        "Generate and analyze persistence diagrams from computed homology. "
        "Persistence diagrams plot (birth, death) for each topological feature. "
        "Points FAR from diagonal = significant features (high persistence). "
        "Points NEAR diagonal = noise or short-lived features. "
        "Input: persistence data from compute_ph tool. "
        "Output: diagram statistics, significant features, optional visualization."
    )
    args_schema: Type[BaseModel] = PersistenceDiagramInput

    def _run(
        self,
        persistence_data: Dict[str, List[Dict[str, float]]],
        output_path: Optional[str] = None,
        title: str = "Persistence Diagram",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Generate persistence diagram analysis.

        Args:
            persistence_data: Persistence pairs by dimension
            output_path: Optional save path for visualization
            title: Diagram title

        Returns:
            Dictionary with diagram analysis
        """
        try:
            # Analyze persistence diagram
            analysis = {}

            for dim_key, pairs in persistence_data.items():
                if not pairs or not isinstance(pairs, list):
                    continue

                # Skip if not proper persistence pairs
                if not all(isinstance(p, dict) and "birth" in p and "death" in p for p in pairs):
                    continue

                # Extract birth/death values
                births = [p["birth"] for p in pairs]
                deaths = [p["death"] for p in pairs]
                persistences = [p["persistence"] if "persistence" in p else abs(p["death"] - p["birth"]) for p in pairs]

                # Calculate statistics
                dim_analysis = {
                    "num_features": len(pairs),
                    "birth_range": [float(min(births)), float(max(births))],
                    "death_range": [float(min(deaths)), float(max(deaths))],
                    "persistence_stats": {
                        "min": float(min(persistences)),
                        "max": float(max(persistences)),
                        "mean": float(np.mean(persistences)),
                        "median": float(np.median(persistences)),
                        "std": float(np.std(persistences)),
                        "total": float(sum(persistences))
                    }
                }

                # Identify significant features (persistence > median + 1 std)
                threshold = np.median(persistences) + np.std(persistences)
                significant = [p for p in pairs if p.get("persistence", abs(p["death"] - p["birth"])) > threshold]
                dim_analysis["significant_features"] = {
                    "count": len(significant),
                    "threshold": float(threshold),
                    "top_5": sorted(significant, key=lambda x: x.get("persistence", 0), reverse=True)[:5]
                }

                # Persistence entropy
                if persistences:
                    total_pers = sum(persistences)
                    if total_pers > 0:
                        probs = [p / total_pers for p in persistences]
                        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                        dim_analysis["persistence_entropy"] = float(entropy)

                analysis[dim_key] = dim_analysis

            # Generate visualization if path provided
            visualization_info = None
            if output_path:
                visualization_info = self._generate_visualization(persistence_data, output_path, title)

            return {
                "success": True,
                "diagram_analysis": analysis,
                "visualization": visualization_info,
                "interpretation": self._generate_interpretation(analysis)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_visualization(
        self,
        persistence_data: Dict,
        output_path: str,
        title: str
    ) -> Dict[str, Any]:
        """Generate persistence diagram visualization.

        Args:
            persistence_data: Persistence pairs
            output_path: Save path
            title: Diagram title

        Returns:
            Visualization info dictionary
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))

            colors = {'H0': 'blue', 'H1': 'red', 'H2': 'green'}
            markers = {'H0': 'o', 'H1': 's', 'H2': '^'}

            all_births = []
            all_deaths = []

            for dim_key, pairs in persistence_data.items():
                if not pairs or not isinstance(pairs, list):
                    continue
                if not all(isinstance(p, dict) and "birth" in p and "death" in p for p in pairs):
                    continue

                births = [p["birth"] for p in pairs]
                deaths = [p["death"] for p in pairs]

                all_births.extend(births)
                all_deaths.extend(deaths)

                ax.scatter(
                    births, deaths,
                    c=colors.get(dim_key, 'gray'),
                    marker=markers.get(dim_key, 'o'),
                    label=dim_key,
                    alpha=0.7,
                    s=50
                )

            # Add diagonal line
            if all_births and all_deaths:
                min_val = min(min(all_births), min(all_deaths))
                max_val = max(max(all_births), max(all_deaths))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Diagonal')

            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(title)
            ax.legend()
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            return {
                "saved_to": output_path,
                "dimensions_plotted": list(persistence_data.keys())
            }

        except Exception as e:
            return {
                "error": f"Visualization failed: {str(e)}"
            }

    def _generate_interpretation(self, analysis: Dict) -> str:
        """Generate human-readable interpretation of the diagram.

        Args:
            analysis: Diagram analysis results

        Returns:
            Interpretation string
        """
        interpretations = []

        for dim_key, dim_analysis in analysis.items():
            if dim_key == "H0":
                n_features = dim_analysis["num_features"]
                n_significant = dim_analysis["significant_features"]["count"]
                interpretations.append(
                    f"H0 (Connected Components): {n_features} total, {n_significant} significant. "
                    f"This indicates approximately {n_significant} distinct regions/objects."
                )
            elif dim_key == "H1":
                n_features = dim_analysis["num_features"]
                n_significant = dim_analysis["significant_features"]["count"]
                interpretations.append(
                    f"H1 (Loops/Holes): {n_features} total, {n_significant} significant. "
                    f"This suggests {n_significant} distinct ring/loop structures or boundaries."
                )
            elif dim_key == "H2":
                n_features = dim_analysis["num_features"]
                n_significant = dim_analysis["significant_features"]["count"]
                interpretations.append(
                    f"H2 (Voids): {n_features} total, {n_significant} significant. "
                    f"This indicates {n_significant} enclosed cavity structures (3D data)."
                )

        return " ".join(interpretations) if interpretations else "No interpretable features found."

    async def _arun(self, *args, **kwargs) -> Dict[str, Any]:
        """Async version."""
        return self._run(*args, **kwargs)
