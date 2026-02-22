"""TopoAgent Reflection Module.

Implements EndoAgent's reflection mechanism which adds +26.5% visual accuracy.
The reflection system analyzes tool execution results and generates:
1. Error Analysis: What could be improved
2. Suggestion: What to do next
3. Experience: Reusable lessons learned

This module also handles the dual-memory mechanism:
- Short-term memory (Ms): Recent tool outputs
- Long-term memory (Ml): Past reflection experiences
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.messages import HumanMessage

from .state import ReflectionEntry, TopoAgentState


@dataclass
class ReflectionResult:
    """Result of a reflection operation."""
    entry: ReflectionEntry
    is_task_complete: bool
    confidence: float
    next_action_suggestion: str


class ReflectionEngine:
    """Engine for generating reflections on tool executions.

    Based on EndoAgent's reflection mechanism which showed:
    - Reflection alone: +26.5% visual accuracy
    - Dual-memory: +1.5% additional visual, +3.06% language accuracy
    """

    REFLECTION_PROMPT = """You are reflecting on a TDA step for SKIN LESION classification (DermaMNIST).

## Current Task
{task_description}

## Current Round: {current_round}/{max_rounds}

## Recent Tool Execution
Tool: {tool_name}
Input: {tool_input}
Output: {tool_output}
Success: {success}

## Short-term Memory (Recent Actions)
{short_term_memory}

## Long-term Memory (Past Experiences)
{long_term_memory}

## DermaMNIST Classes (for reference)
1. Melanocytic nevi (nv) - Benign moles
2. Melanoma (mel) - Malignant, HIGH H1_entropy, irregular boundaries
3. Benign keratosis (bkl) - Uniform texture
4. Basal cell carcinoma (bcc) - Defined borders
5. Actinic keratosis (akiec) - Flat pattern
6. Vascular lesions (vasc) - Distinctive H1 loops
7. Dermatofibroma (df) - Central clearing

## Reflection Questions

### 1. Feature Quality (for compute_ph or topological_features)
Analyze the TDA output:
- **H0 quality**: Component count and persistence? [few/moderate/many]
- **H1 quality**: Loop/boundary strength? [weak/moderate/strong]
- **Noise level**: Short-lived features? [low/medium/high]

### 2. Discriminative Power
Evaluate classification signal:
- H1_entropy value: [<0.8 = benign pattern, >1.5 = melanoma risk]
- H0_H1_ratio: [<2 = loop-rich, >5 = component-dominated]
- Signal strength: [weak/moderate/strong]
- Likely class: [class name based on patterns]

### 3. Error Analysis
If issues occurred:
- Failure reason: [format error/empty data/computation error]
- Missing data: What's needed?
- Unexpected results: Explanation?

### 4. Next Step
Recommend ONE of:
- CONTINUE: "Ready to classify with current features"
- AUGMENT: "Need betti_curves for better discrimination"
- RETRY: "Try superlevel filtration for dark lesions"
- DEBUG: "Image issue - check preprocessing"

### 5. Reusable Experience
Pattern: "For [lesion type] with [H0/H1 pattern], [approach] works [well/poorly]"

Respond in the following format:
ERROR_ANALYSIS: <what could be improved>
SUGGESTION: <what to do next>
EXPERIENCE: <generalizable lesson>
IS_COMPLETE: <true/false>
CONFIDENCE: <0-100>
"""

    def __init__(self, model):
        """Initialize the reflection engine.

        Args:
            model: LangChain chat model for generating reflections
        """
        self.model = model

    def reflect(
        self,
        state: TopoAgentState,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool = True
    ) -> ReflectionResult:
        """Generate a reflection on a tool execution.

        Args:
            state: Current agent state
            tool_name: Name of executed tool
            tool_input: Input passed to the tool
            tool_output: Output from the tool
            success: Whether tool execution succeeded

        Returns:
            ReflectionResult with analysis and suggestions
        """
        # Format the reflection prompt
        prompt = self.REFLECTION_PROMPT.format(
            task_description=state["query"],
            current_round=state["current_round"],
            max_rounds=state["max_rounds"],
            tool_name=tool_name,
            tool_input=str(tool_input),
            tool_output=str(tool_output)[:1000],  # Truncate long outputs
            success=success,
            short_term_memory=self._format_short_term(state["short_term_memory"]),
            long_term_memory=self._format_long_term(state["long_term_memory"])
        )

        # Generate reflection
        response = self.model.invoke([HumanMessage(content=prompt)])
        reflection_text = response.content

        # Parse the structured response
        entry, is_complete, confidence, suggestion = self._parse_reflection(
            reflection_text, state["current_round"]
        )

        return ReflectionResult(
            entry=entry,
            is_task_complete=is_complete,
            confidence=confidence,
            next_action_suggestion=suggestion
        )

    def _format_short_term(self, memory: List[Tuple[str, Any]]) -> str:
        """Format short-term memory for prompt.

        Args:
            memory: List of (tool_name, output) tuples

        Returns:
            Formatted string
        """
        if not memory:
            return "No previous actions in this session."

        lines = []
        for i, (tool, output) in enumerate(memory[-5:], 1):  # Last 5 actions
            output_str = str(output)[:200]  # Truncate
            lines.append(f"{i}. {tool}: {output_str}")
        return "\n".join(lines)

    def _format_long_term(self, memory: List[ReflectionEntry]) -> str:
        """Format long-term memory for prompt.

        Args:
            memory: List of ReflectionEntry objects

        Returns:
            Formatted string
        """
        if not memory:
            return "No past experiences recorded."

        lines = []
        for entry in memory[-3:]:  # Last 3 reflections
            lines.append(f"Round {entry.round}: {entry.experience}")
        return "\n".join(lines)

    def _parse_reflection(
        self,
        text: str,
        current_round: int
    ) -> Tuple[ReflectionEntry, bool, float, str]:
        """Parse structured reflection response.

        Args:
            text: Raw reflection text
            current_round: Current round number

        Returns:
            Tuple of (ReflectionEntry, is_complete, confidence, suggestion)
        """
        # Extract sections
        error_analysis = self._extract_field(text, "ERROR_ANALYSIS")
        suggestion = self._extract_field(text, "SUGGESTION")
        experience = self._extract_field(text, "EXPERIENCE")
        is_complete_str = self._extract_field(text, "IS_COMPLETE")
        confidence_str = self._extract_field(text, "CONFIDENCE")

        # Parse boolean and float
        is_complete = "true" in is_complete_str.lower()
        try:
            confidence = float(confidence_str.replace("%", "").strip())
        except (ValueError, AttributeError):
            confidence = 0.0

        entry = ReflectionEntry(
            round=current_round,
            error_analysis=error_analysis or "No issues identified",
            suggestion=suggestion or "Continue with next appropriate tool",
            experience=experience or "No specific lesson recorded"
        )

        return entry, is_complete, confidence, suggestion

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a field value from reflection text.

        Args:
            text: Full reflection text
            field_name: Name of field to extract (e.g., "ERROR_ANALYSIS")

        Returns:
            Extracted value or empty string
        """
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if field_name in line.upper():
                # Get content after the colon
                if ":" in line:
                    content = line.split(":", 1)[1].strip()
                    # If content is on next line(s)
                    if not content and i + 1 < len(lines):
                        content = lines[i + 1].strip()
                    return content
        return ""


class DualMemoryManager:
    """Manager for dual-memory system (Ms and Ml).

    Implements the memory update rules from EndoAgent:
    - Ms (short-term): Stores recent tool executions
    - Ml (long-term): Stores reflection experiences
    """

    def __init__(self, max_short_term: int = 10, max_long_term: int = 20):
        """Initialize memory manager.

        Args:
            max_short_term: Maximum entries in short-term memory
            max_long_term: Maximum entries in long-term memory
        """
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term

    def update_short_term(
        self,
        memory: List[Tuple[str, Any]],
        tool_name: str,
        output: Any
    ) -> List[Tuple[str, Any]]:
        """Update short-term memory with new tool output.

        Ms = Ms ∪ {(tool_t, output_t)}

        Args:
            memory: Current short-term memory
            tool_name: Name of executed tool
            output: Tool output

        Returns:
            Updated memory list
        """
        new_memory = memory.copy()
        new_memory.append((tool_name, output))

        # Trim if exceeds max
        if len(new_memory) > self.max_short_term:
            new_memory = new_memory[-self.max_short_term:]

        return new_memory

    def update_long_term(
        self,
        memory: List[ReflectionEntry],
        reflection: ReflectionEntry
    ) -> List[ReflectionEntry]:
        """Update long-term memory with new reflection.

        Ml = Ml ∪ {reflection_t}

        Args:
            memory: Current long-term memory
            reflection: New reflection entry

        Returns:
            Updated memory list
        """
        new_memory = memory.copy()
        new_memory.append(reflection)

        # Trim if exceeds max
        if len(new_memory) > self.max_long_term:
            new_memory = new_memory[-self.max_long_term:]

        return new_memory

    def get_relevant_experiences(
        self,
        memory: List[ReflectionEntry],
        tool_name: Optional[str] = None,
        n: int = 3
    ) -> List[ReflectionEntry]:
        """Get relevant past experiences.

        Args:
            memory: Long-term memory
            tool_name: Optional tool name to filter by
            n: Number of experiences to return

        Returns:
            List of relevant ReflectionEntry objects
        """
        if not memory:
            return []

        # For now, just return most recent
        # Future: could implement semantic similarity search
        return memory[-n:]

    def clear_short_term(self) -> List[Tuple[str, Any]]:
        """Clear short-term memory (for new session).

        Returns:
            Empty memory list
        """
        return []

    def summarize_session(
        self,
        short_term: List[Tuple[str, Any]],
        long_term: List[ReflectionEntry]
    ) -> str:
        """Generate a summary of the current session.

        Args:
            short_term: Short-term memory
            long_term: Long-term memory

        Returns:
            Summary string
        """
        lines = ["=== Session Summary ==="]
        lines.append(f"\nTools executed: {len(short_term)}")

        if short_term:
            tool_counts = {}
            for tool, _ in short_term:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            lines.append("Tool usage:")
            for tool, count in sorted(tool_counts.items()):
                lines.append(f"  - {tool}: {count}x")

        lines.append(f"\nReflections generated: {len(long_term)}")

        if long_term:
            lines.append("Key experiences:")
            for entry in long_term[-3:]:
                lines.append(f"  - Round {entry.round}: {entry.experience[:100]}...")

        return "\n".join(lines)
