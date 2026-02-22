"""Long-term Memory Module for TopoAgent.

Implements Ml: the long-term memory storing reflection experiences.
Ml = [ReflectionEntry_1, ReflectionEntry_2, ...]

From EndoAgent: Dual-memory adds +1.5% visual accuracy, +3.06% language accuracy.
"""

from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class ReflectionEntry:
    """A single reflection entry in long-term memory."""
    round: int
    error_analysis: str
    suggestion: str
    experience: str
    context: Optional[str] = None
    tool_sequence: Optional[List[str]] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    session_id: Optional[str] = None


@dataclass
class V9ExperienceEntry:
    """Structured experience entry for v9 pipeline.

    Key improvements over ReflectionEntry:
    - would_choose_again: binary signal for ACT decisions
    - ph_metrics: enables similarity-based retrieval
    - descriptor_params: enables parameter learning
    - feature_quality: specific outcomes, not vague text
    """
    object_type: str
    descriptor: str
    image_metrics: Dict[str, float]  # snr, contrast, edge_density
    ph_metrics: Dict[str, float]     # h0_count, h1_count, h0_avg_pers, h1_avg_pers
    feature_quality: Dict[str, float]  # sparsity, variance, dimension
    quality_verdict: str             # good/acceptable/poor
    lesson: str                      # specific: "PI with sigma=0.5 gave 2% sparsity on organ_shape"
    would_choose_again: bool         # actionable signal for ACT
    stance: str                      # inferred post-hoc: confirmed_hypothesis/switched_to_benchmark/chose_alternative
    descriptor_params: Dict[str, Any]  # actual params used
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class LongTermMemory:
    """Long-term memory manager (Ml from EndoAgent).

    Stores reflection experiences that can guide future sessions.
    Unlike short-term memory, this persists across sessions.

    Key insight from EndoAgent:
    - Reflection adds +26.5% visual accuracy
    - Dual-memory adds additional +1.5% visual, +3.06% language accuracy
    """

    def __init__(
        self,
        max_entries: int = 100,
        persistence_path: Optional[str] = None
    ):
        """Initialize long-term memory.

        Args:
            max_entries: Maximum entries to keep
            persistence_path: Optional path to persist memory
        """
        self.max_entries = max_entries
        self.persistence_path = persistence_path
        self._memory: List[Any] = []  # ReflectionEntry or V9ExperienceEntry

        # Load from disk if path exists
        if persistence_path and os.path.exists(persistence_path):
            self.load()

    def add(self, reflection: ReflectionEntry) -> None:
        """Add a reflection entry to memory.

        Ml = Ml ∪ {reflection_t}

        Args:
            reflection: Reflection entry to add
        """
        self._memory.append(reflection)

        # Trim if exceeds max
        if len(self._memory) > self.max_entries:
            self._memory = self._memory[-self.max_entries:]

        # Auto-save if persistence enabled
        if self.persistence_path:
            self.save()

    def add_from_dict(
        self,
        round: int,
        error_analysis: str,
        suggestion: str,
        experience: str,
        **kwargs
    ) -> None:
        """Add reflection from individual components.

        Args:
            round: Reflection round
            error_analysis: What went wrong
            suggestion: What to do next
            experience: Lesson learned
            **kwargs: Additional optional fields
        """
        entry = ReflectionEntry(
            round=round,
            error_analysis=error_analysis,
            suggestion=suggestion,
            experience=experience,
            **kwargs
        )
        self.add(entry)

    def get_recent(self, n: int = 5) -> List[ReflectionEntry]:
        """Get n most recent reflections.

        Args:
            n: Number to return

        Returns:
            List of reflections
        """
        return self._memory[-n:]

    def get_all(self) -> List[ReflectionEntry]:
        """Get all reflections.

        Returns:
            List of all reflections
        """
        return self._memory.copy()

    def get_by_session(self, session_id: str) -> List[ReflectionEntry]:
        """Get reflections from a specific session.

        Args:
            session_id: Session identifier

        Returns:
            List of reflections from session
        """
        return [e for e in self._memory
                if isinstance(e, ReflectionEntry) and e.session_id == session_id]

    def search_experiences(self, query: str, n: int = 3) -> List[ReflectionEntry]:
        """Search for relevant past experiences.

        Simple keyword-based search (can be enhanced with embeddings).

        Args:
            query: Search query
            n: Number of results

        Returns:
            Relevant reflections
        """
        query_lower = query.lower()
        scored = []

        for entry in self._memory:
            if not isinstance(entry, ReflectionEntry):
                continue
            # Simple scoring based on keyword matches
            text = f"{entry.error_analysis} {entry.suggestion} {entry.experience}".lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, entry))

        # Sort by score and return top n
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:n]]

    def get_common_suggestions(self) -> Dict[str, int]:
        """Get most common suggestions from memory.

        Returns:
            Dictionary of suggestion -> count
        """
        suggestions = {}
        for entry in self._memory:
            if not isinstance(entry, ReflectionEntry):
                continue
            # Normalize suggestion
            key = entry.suggestion[:100].strip().lower()
            suggestions[key] = suggestions.get(key, 0) + 1
        return dict(sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[:10])

    def get_tool_patterns(self) -> Dict[str, int]:
        """Get common tool sequence patterns.

        Returns:
            Dictionary of pattern -> count
        """
        patterns = {}
        for entry in self._memory:
            if not isinstance(entry, ReflectionEntry):
                continue
            if entry.tool_sequence:
                pattern = "->".join(entry.tool_sequence[:5])
                patterns[pattern] = patterns.get(pattern, 0) + 1
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10])

    def format_for_prompt(self, n: int = 3) -> str:
        """Format memory for LLM prompt.

        Args:
            n: Number of entries to include

        Returns:
            Formatted string
        """
        if not self._memory:
            return "No past experiences recorded."

        lines = ["Past experiences and lessons learned:"]
        for entry in self._memory[-n:]:
            if isinstance(entry, V9ExperienceEntry):
                choose_str = "YES" if entry.would_choose_again else "NO"
                lines.append(f"[v9 {entry.object_type}, {entry.descriptor}]:")
                lines.append(f"  - Quality: {entry.quality_verdict} | Choose again: {choose_str}")
                lines.append(f"  - Lesson: {entry.lesson[:100]}...")
            else:
                lines.append(f"Round {entry.round}:")
                lines.append(f"  - Analysis: {entry.error_analysis[:100]}...")
                lines.append(f"  - Suggestion: {entry.suggestion[:100]}...")
                lines.append(f"  - Experience: {entry.experience[:100]}...")

        return "\n".join(lines)

    def save(self) -> None:
        """Save memory to disk.

        Handles both ReflectionEntry and V9ExperienceEntry.
        V9 entries are marked with "_v9": true for deserialization.
        """
        if not self.persistence_path:
            return

        data = []
        for e in self._memory:
            if isinstance(e, V9ExperienceEntry):
                data.append({
                    "_v9": True,
                    "object_type": e.object_type,
                    "descriptor": e.descriptor,
                    "image_metrics": e.image_metrics,
                    "ph_metrics": e.ph_metrics,
                    "feature_quality": e.feature_quality,
                    "quality_verdict": e.quality_verdict,
                    "lesson": e.lesson,
                    "would_choose_again": e.would_choose_again,
                    "stance": e.stance,
                    "descriptor_params": e.descriptor_params,
                    "timestamp": e.timestamp,
                })
            else:
                data.append({
                    "round": e.round,
                    "error_analysis": e.error_analysis,
                    "suggestion": e.suggestion,
                    "experience": e.experience,
                    "context": e.context,
                    "tool_sequence": e.tool_sequence,
                    "timestamp": e.timestamp,
                    "session_id": e.session_id,
                })

        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load memory from disk.

        Handles both ReflectionEntry and V9ExperienceEntry.
        V9 entries are identified by the "_v9": true marker field.
        """
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return

        with open(self.persistence_path, 'r') as f:
            data = json.load(f)

        self._memory = []
        for d in data:
            if d.get("_v9"):
                self._memory.append(V9ExperienceEntry(
                    object_type=d["object_type"],
                    descriptor=d["descriptor"],
                    image_metrics=d.get("image_metrics", {}),
                    ph_metrics=d.get("ph_metrics", {}),
                    feature_quality=d.get("feature_quality", {}),
                    quality_verdict=d.get("quality_verdict", "unknown"),
                    lesson=d.get("lesson", ""),
                    would_choose_again=d.get("would_choose_again", True),
                    stance=d.get("stance", "AGREE"),
                    descriptor_params=d.get("descriptor_params", {}),
                    timestamp=d.get("timestamp", 0),
                ))
            else:
                self._memory.append(ReflectionEntry(
                    round=d["round"],
                    error_analysis=d["error_analysis"],
                    suggestion=d["suggestion"],
                    experience=d["experience"],
                    context=d.get("context"),
                    tool_sequence=d.get("tool_sequence"),
                    timestamp=d.get("timestamp", 0),
                    session_id=d.get("session_id"),
                ))

    def clear(self) -> None:
        """Clear all memory."""
        self._memory = []
        if self.persistence_path and os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Statistics dictionary
        """
        if not self._memory:
            return {"total_entries": 0}

        reflection_entries = [e for e in self._memory if isinstance(e, ReflectionEntry)]
        v9_entries = [e for e in self._memory if isinstance(e, V9ExperienceEntry)]

        stats: Dict[str, Any] = {
            "total_entries": len(self._memory),
            "reflection_entries": len(reflection_entries),
            "v9_entries": len(v9_entries),
        }

        if reflection_entries:
            stats["sessions"] = len(set(e.session_id for e in reflection_entries if e.session_id))
            stats["avg_round"] = sum(e.round for e in reflection_entries) / len(reflection_entries)
            stats["common_suggestions"] = self.get_common_suggestions()

        if v9_entries:
            stats["v9_object_types"] = list(set(e.object_type for e in v9_entries))
            stats["v9_descriptors"] = list(set(e.descriptor for e in v9_entries))
            stats["v9_would_choose_again_rate"] = (
                sum(1 for e in v9_entries if e.would_choose_again) / len(v9_entries)
            )

        return stats

    # ---- v9 methods ----

    def _ph_similarity(self, stored: Dict, current: Dict) -> float:
        """Cosine similarity on log-scaled PH metric vectors.

        Metrics: h0_count, h1_count, h0_avg_pers, h1_avg_pers, h1_h0_ratio.
        Log-scale counts (which span 10-10000), multiply persistence by 100
        to normalize into a comparable range.

        Args:
            stored: PH metrics from a stored entry
            current: PH metrics from the query

        Returns:
            Cosine similarity in [0, 1]
        """
        import math

        def to_vec(m):
            return [
                math.log1p(m.get("h0_count", 0)),
                math.log1p(m.get("h1_count", 0)),
                m.get("h0_avg_pers", 0) * 100,
                m.get("h1_avg_pers", 0) * 100,
                m.get("h1_h0_ratio", 1.0),
            ]

        a, b = to_vec(stored), to_vec(current)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_by_profile(
        self, object_type: str, ph_metrics: Dict, n: int = 5
    ) -> List[Tuple[float, V9ExperienceEntry]]:
        """Profile-based search over v9 experience entries.

        Exact match on object_type (with fallback to all v9 entries),
        ranked by cosine similarity on PH metrics.

        Args:
            object_type: The object type to match (e.g. "organ_shape")
            ph_metrics: Current PH metrics for similarity comparison
            n: Maximum number of results to return

        Returns:
            List of (similarity_score, entry) tuples sorted descending by score
        """
        v9_entries = [e for e in self._memory if isinstance(e, V9ExperienceEntry)]
        if not v9_entries:
            return []

        # Try exact object_type match first
        matched = [e for e in v9_entries if e.object_type == object_type]
        if not matched:
            # Fallback: use all v9 entries
            matched = v9_entries

        scored = []
        for entry in matched:
            sim = self._ph_similarity(entry.ph_metrics, ph_metrics)
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:n]

    def format_for_v9_prompt(
        self, object_type: str, ph_metrics: Dict, n: int = 5
    ) -> str:
        """Format retrieved v9 entries for the ACT prompt.

        Args:
            object_type: The object type to search for
            ph_metrics: Current PH metrics for similarity ranking
            n: Maximum number of entries to include

        Returns:
            Formatted string for LLM prompt consumption
        """
        v9_total = sum(1 for e in self._memory if isinstance(e, V9ExperienceEntry))
        results = self.search_by_profile(object_type, ph_metrics, n=n)

        if not results:
            return f"No v9 experiences recorded (total v9 entries: {v9_total})."

        lines = [
            f"Retrieved {len(results)} of {v9_total} v9 experiences "
            f"(object_type={object_type}):"
        ]
        for sim, entry in results:
            choose_str = "YES" if entry.would_choose_again else "NO"
            params_str = json.dumps(entry.descriptor_params) if entry.descriptor_params else "{}"
            lines.append(
                f"  [{entry.object_type}, {entry.descriptor}] "
                f"Quality: {entry.quality_verdict} | "
                f"Would choose again: {choose_str}"
            )
            lines.append(f"    PH similarity: {sim:.3f} | Params: {params_str}")
            lines.append(f"    Lesson: {entry.lesson}")

        return "\n".join(lines)

    def add_v9(self, entry: V9ExperienceEntry) -> None:
        """Store a v9 experience entry.

        Appends to the shared _memory list and auto-saves.
        V9 entries are serialized with a "_v9" marker field so they
        are distinguishable from ReflectionEntry during load().

        Args:
            entry: V9ExperienceEntry to store
        """
        self._memory.append(entry)

        # Trim if exceeds max
        if len(self._memory) > self.max_entries:
            self._memory = self._memory[-self.max_entries:]

        # Auto-save if persistence enabled
        if self.persistence_path:
            self.save()

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._memory)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self._memory)
