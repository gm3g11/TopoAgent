"""Skill Memory — Voyager-style skill library + Reflexion verbal memory.

Persistent skill learning that grows over time:
- learned_rules.json: Structured rules from benchmark feedback
- reflections.json: Verbal reflections from LLM
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SkillMemory:
    """Persistent skill learning — Voyager-style skill library + Reflexion verbal memory."""

    def __init__(self, memory_dir: Optional[str] = None):
        if memory_dir is None:
            memory_dir = str(Path(__file__).parent)
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.rules_path = self.memory_dir / "learned_rules.json"
        self.reflections_path = self.memory_dir / "reflections.json"

    def load_learned_rules(self) -> Dict:
        """Load learned rules from disk."""
        if self.rules_path.exists():
            with open(self.rules_path, "r") as f:
                return json.load(f)
        return {"version": 1, "rules": [], "updated_rankings": {}}

    def save_learned_rules(self, rules: Dict) -> None:
        """Save learned rules to disk."""
        with open(self.rules_path, "w") as f:
            json.dump(rules, f, indent=2, default=str)

    def load_reflections(self) -> List[Dict]:
        """Load reflections from disk."""
        if self.reflections_path.exists():
            with open(self.reflections_path, "r") as f:
                data = json.load(f)
                return data.get("reflections", [])
        return []

    def save_reflection(self, reflection: Dict) -> None:
        """Append a verbal reflection to persistent storage (deduplicates)."""
        reflections = self.load_reflections()

        # Deduplicate: keep only the latest reflection per (context, choice)
        key = (reflection.get("context", ""), reflection.get("choice", ""))
        reflections = [
            r for r in reflections
            if (r.get("context", ""), r.get("choice", "")) != key
        ]
        reflections.append(reflection)

        with open(self.reflections_path, "w") as f:
            json.dump({"reflections": reflections}, f, indent=2, default=str)

    def update_from_benchmark(
        self,
        agent_choice: str,
        agent_accuracy: float,
        oracle_choice: Optional[str],
        oracle_accuracy: Optional[float],
        object_type: str,
        dataset: str,
        params_used: Optional[Dict] = None,
        source: str = "protocol2_benchmark",
    ) -> None:
        """Auto-update after benchmark evaluation."""
        rules = self.load_learned_rules()

        was_optimal = (oracle_choice is None) or (agent_choice == oracle_choice)

        rule_entry = {
            "timestamp": datetime.now().isoformat(),
            "object_type": object_type,
            "dataset": dataset,
            "agent_choice": agent_choice,
            "agent_accuracy": agent_accuracy,
            "oracle_choice": oracle_choice,
            "oracle_accuracy": oracle_accuracy,
            "was_optimal": was_optimal,
            "params_used": params_used or {},
            "source": source,
        }
        rules["rules"].append(rule_entry)

        # Update rankings
        rankings = rules.setdefault("updated_rankings", {})
        ot_rankings = rankings.setdefault(object_type, [])

        # Update or add entry for agent_choice
        found = False
        for entry in ot_rankings:
            if entry["descriptor"] == agent_choice:
                # Running average
                old_trials = entry.get("trials", 1)
                entry["accuracy"] = (
                    entry["accuracy"] * old_trials + agent_accuracy
                ) / (old_trials + 1)
                entry["trials"] = old_trials + 1
                found = True
                break
        if not found:
            ot_rankings.append({
                "descriptor": agent_choice,
                "accuracy": agent_accuracy,
                "trials": 1,
            })

        # Sort by accuracy descending
        ot_rankings.sort(key=lambda x: x["accuracy"], reverse=True)

        self.save_learned_rules(rules)

    def record_reflection(
        self,
        context: str,
        choice: str,
        outcome: str,
        reflection_text: str,
        lesson: str,
    ) -> None:
        """Store verbal reflection from LLM."""
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "choice": choice,
            "outcome": outcome,
            "reflection": reflection_text,
            "lesson_learned": lesson,
        }
        self.save_reflection(reflection)

    def get_learned_context(self, object_type: Optional[str] = None) -> str:
        """Format learned rules + reflections for prompt injection.

        Args:
            object_type: If provided, filter to relevant rules/reflections

        Returns:
            Formatted string for LLM prompt
        """
        sections = []

        # Learned rules
        rules = self.load_learned_rules()
        updated_rankings = rules.get("updated_rankings", {})

        if updated_rankings:
            sections.append("## Learned Rules (from past runs)")
            for ot, rankings in updated_rankings.items():
                if object_type and ot != object_type:
                    continue
                for entry in rankings[:3]:
                    desc = entry["descriptor"]
                    acc = entry["accuracy"]
                    trials = entry.get("trials", 1)
                    sections.append(
                        f"- {ot}: {desc} confirmed ({trials} trial(s), {acc:.1%})"
                    )
            sections.append("")

        # Reflections
        reflections = self.load_reflections()
        relevant = reflections
        if object_type:
            relevant = [
                r for r in reflections
                if object_type in r.get("context", "")
            ]

        if relevant:
            sections.append("## Past Reflections")
            # Show last 5 relevant
            for r in relevant[-5:]:
                lesson = r.get("lesson_learned", "")
                if lesson:
                    sections.append(f'- "{lesson}"')
            sections.append("")

        if not sections:
            return "No learned rules or reflections yet."

        return "\n".join(sections)

    def get_updated_rankings(self, object_type: str) -> Optional[List]:
        """Get rankings updated by learned rules (merged with benchmark base)."""
        rules = self.load_learned_rules()
        return rules.get("updated_rankings", {}).get(object_type)
