"""Portfolio Memory — Tracks fusion outcomes for sequential learning.

Records genuinely new information that the agent couldn't predict from
pre-computed single-descriptor accuracies: when does multi-descriptor
fusion help vs hurt?

Usage:
    from topoagent.memory.portfolio_memory import PortfolioMemory

    memory = PortfolioMemory()
    memory.record(
        dataset="TissueMNIST",
        object_type="discrete_cells",
        portfolio=["ATOL", "edge_histogram"],
        fusion_accuracy=0.42,
        primary_accuracy=0.40,
        fusion_strategy="concat_pair",
    )
    context = memory.get_fusion_context("discrete_cells")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PortfolioMemory:
    """Persistent memory for fusion portfolio outcomes.

    Tracks which descriptor combinations work for which object types,
    providing genuinely learned context for future portfolio decisions.
    """

    def __init__(self, memory_dir: Optional[str] = None):
        if memory_dir is None:
            memory_dir = str(Path(__file__).parent)
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_path = self.memory_dir / "portfolio_outcomes.json"

    def load(self) -> Dict:
        """Load portfolio outcomes from disk."""
        if self.portfolio_path.exists():
            with open(self.portfolio_path, "r") as f:
                return json.load(f)
        return {"version": 1, "outcomes": [], "summary": {}}

    def save(self, data: Dict) -> None:
        """Save portfolio outcomes to disk."""
        with open(self.portfolio_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def record(
        self,
        dataset: str,
        object_type: str,
        portfolio: List[str],
        fusion_accuracy: float,
        primary_accuracy: float,
        fusion_strategy: str = "concat_pair",
    ) -> None:
        """Record a fusion experiment outcome.

        Args:
            dataset: Dataset name
            object_type: Object type
            portfolio: List of descriptor names (len >= 2)
            fusion_accuracy: Accuracy of concatenated features
            primary_accuracy: Accuracy of the primary (first) descriptor alone
            fusion_strategy: 'concat_pair', 'concat_triple', or 'primary_only'
        """
        data = self.load()
        fusion_gain = fusion_accuracy - primary_accuracy

        outcome = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "object_type": object_type,
            "portfolio": portfolio,
            "primary": portfolio[0],
            "secondary": portfolio[1] if len(portfolio) > 1 else None,
            "tertiary": portfolio[2] if len(portfolio) > 2 else None,
            "fusion_accuracy": round(fusion_accuracy, 6),
            "primary_accuracy": round(primary_accuracy, 6),
            "fusion_gain": round(fusion_gain, 6),
            "fusion_helped": fusion_gain > 0.001,
            "fusion_strategy": fusion_strategy,
        }
        data["outcomes"].append(outcome)

        # Update per-object-type summary
        summary = data.setdefault("summary", {})
        ot_summary = summary.setdefault(object_type, {
            "total": 0,
            "fusion_helped": 0,
            "mean_gain": 0.0,
            "best_pair": None,
            "best_gain": -1.0,
            "worst_pair": None,
            "worst_gain": 1.0,
        })

        n = ot_summary["total"]
        ot_summary["mean_gain"] = (ot_summary["mean_gain"] * n + fusion_gain) / (n + 1)
        ot_summary["total"] = n + 1
        if fusion_gain > 0.001:
            ot_summary["fusion_helped"] += 1
        if fusion_gain > ot_summary["best_gain"]:
            ot_summary["best_gain"] = round(fusion_gain, 6)
            ot_summary["best_pair"] = portfolio[:2]
        if fusion_gain < ot_summary["worst_gain"]:
            ot_summary["worst_gain"] = round(fusion_gain, 6)
            ot_summary["worst_pair"] = portfolio[:2]
        ot_summary["mean_gain"] = round(ot_summary["mean_gain"], 6)

        self.save(data)

    def get_fusion_context(self, object_type: Optional[str] = None) -> str:
        """Format fusion outcomes as context for LLM prompt injection.

        Returns a summary of when fusion helped/hurt for the given object type.
        """
        data = self.load()
        outcomes = data.get("outcomes", [])
        summary = data.get("summary", {})

        if not outcomes:
            return "(no fusion history available — this is the first portfolio evaluation)"

        sections = []
        sections.append("## Fusion History (from past portfolio evaluations)")

        if object_type and object_type in summary:
            ot_sum = summary[object_type]
            n = ot_sum["total"]
            helped = ot_sum["fusion_helped"]
            pct = (helped / n * 100) if n > 0 else 0
            sections.append(f"\n### {object_type} ({n} experiments)")
            sections.append(f"  - Fusion helped: {helped}/{n} ({pct:.0f}%)")
            sections.append(f"  - Mean gain: {ot_sum['mean_gain']:+.3f}")
            if ot_sum.get("best_pair"):
                sections.append(f"  - Best pair: {'+'.join(ot_sum['best_pair'])} "
                                f"(gain={ot_sum['best_gain']:+.3f})")
            if ot_sum.get("worst_pair"):
                sections.append(f"  - Worst pair: {'+'.join(ot_sum['worst_pair'])} "
                                f"(gain={ot_sum['worst_gain']:+.3f})")

            # Show recent outcomes for this type
            type_outcomes = [o for o in outcomes if o["object_type"] == object_type]
            if type_outcomes:
                sections.append(f"\n  Recent outcomes for {object_type}:")
                for o in type_outcomes[-5:]:
                    portfolio_str = "+".join(o["portfolio"])
                    marker = "HELPED" if o["fusion_helped"] else "HURT"
                    sections.append(
                        f"    - {o['dataset']}: {portfolio_str} → "
                        f"gain={o['fusion_gain']:+.3f} ({marker})"
                    )
        else:
            # Show global summary
            sections.append(f"\nGlobal summary ({len(outcomes)} experiments):")
            for ot, ot_sum in summary.items():
                n = ot_sum["total"]
                helped = ot_sum["fusion_helped"]
                sections.append(
                    f"  - {ot}: {helped}/{n} helped, mean gain={ot_sum['mean_gain']:+.3f}"
                )

        return "\n".join(sections)

    def get_fusion_recommendations(self, object_type: str) -> List[Dict]:
        """Get portfolios with positive fusion gain for this object type.

        Returns list of dicts sorted by fusion_gain descending.
        """
        data = self.load()
        outcomes = data.get("outcomes", [])

        positive = [
            o for o in outcomes
            if o["object_type"] == object_type and o["fusion_helped"]
        ]
        positive.sort(key=lambda x: x["fusion_gain"], reverse=True)
        return positive

    def clear(self) -> None:
        """Clear all portfolio memory (for fresh experiments)."""
        self.save({"version": 1, "outcomes": [], "summary": {}})
