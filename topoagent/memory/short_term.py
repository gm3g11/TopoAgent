"""Short-term Memory Module for TopoAgent.

Implements Ms: the short-term memory storing recent tool executions.
Ms = [(tool_1, output_1), (tool_2, output_2), ...]
"""

from typing import Any, List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ToolExecution:
    """Record of a single tool execution."""
    tool_name: str
    input_args: Dict[str, Any]
    output: Any
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    success: bool = True
    execution_time_ms: float = 0.0


class ShortTermMemory:
    """Short-term memory manager (Ms from EndoAgent).

    Stores recent tool executions for the current session.
    Used by the agent to:
    - Avoid redundant tool calls
    - Track analysis progress
    - Provide context for tool selection
    """

    def __init__(self, max_entries: int = 20):
        """Initialize short-term memory.

        Args:
            max_entries: Maximum number of entries to store
        """
        self.max_entries = max_entries
        self._memory: List[ToolExecution] = []

    def add(
        self,
        tool_name: str,
        output: Any,
        input_args: Optional[Dict] = None,
        success: bool = True,
        execution_time_ms: float = 0.0
    ) -> None:
        """Add a tool execution to memory.

        Ms = Ms ∪ {(tool_t, output_t)}

        Args:
            tool_name: Name of executed tool
            output: Tool output
            input_args: Optional input arguments
            success: Whether execution succeeded
            execution_time_ms: Execution time
        """
        entry = ToolExecution(
            tool_name=tool_name,
            input_args=input_args or {},
            output=output,
            success=success,
            execution_time_ms=execution_time_ms
        )
        self._memory.append(entry)

        # Trim if exceeds max
        if len(self._memory) > self.max_entries:
            self._memory = self._memory[-self.max_entries:]

    def get_recent(self, n: int = 5) -> List[Tuple[str, Any]]:
        """Get n most recent tool executions.

        Args:
            n: Number of entries to return

        Returns:
            List of (tool_name, output) tuples
        """
        return [(e.tool_name, e.output) for e in self._memory[-n:]]

    def get_all(self) -> List[Tuple[str, Any]]:
        """Get all tool executions.

        Returns:
            List of (tool_name, output) tuples
        """
        return [(e.tool_name, e.output) for e in self._memory]

    def get_by_tool(self, tool_name: str) -> List[ToolExecution]:
        """Get all executions of a specific tool.

        Args:
            tool_name: Name of tool

        Returns:
            List of executions
        """
        return [e for e in self._memory if e.tool_name == tool_name]

    def get_tool_sequence(self) -> List[str]:
        """Get sequence of tools executed.

        Returns:
            List of tool names in order
        """
        return [e.tool_name for e in self._memory]

    def has_executed(self, tool_name: str) -> bool:
        """Check if a tool has been executed.

        Args:
            tool_name: Name of tool

        Returns:
            True if tool was executed
        """
        return any(e.tool_name == tool_name for e in self._memory)

    def get_last_output(self, tool_name: Optional[str] = None) -> Optional[Any]:
        """Get the last output, optionally filtered by tool.

        Args:
            tool_name: Optional tool name filter

        Returns:
            Last output or None
        """
        if tool_name:
            executions = self.get_by_tool(tool_name)
            return executions[-1].output if executions else None
        return self._memory[-1].output if self._memory else None

    def format_for_prompt(self, include_details: bool = False) -> str:
        """Format memory for LLM prompt.

        Args:
            include_details: Include execution details

        Returns:
            Formatted string
        """
        if not self._memory:
            return "No previous tool executions in this session."

        lines = ["Recent tool executions:"]
        for i, entry in enumerate(self._memory, 1):
            if include_details:
                output_str = str(entry.output)[:200]
                lines.append(
                    f"{i}. {entry.tool_name} (success={entry.success}): {output_str}..."
                )
            else:
                lines.append(f"{i}. {entry.tool_name}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Statistics dictionary
        """
        if not self._memory:
            return {"total_executions": 0}

        tool_counts = {}
        for e in self._memory:
            tool_counts[e.tool_name] = tool_counts.get(e.tool_name, 0) + 1

        success_rate = sum(1 for e in self._memory if e.success) / len(self._memory)
        avg_time = sum(e.execution_time_ms for e in self._memory) / len(self._memory)

        return {
            "total_executions": len(self._memory),
            "unique_tools": len(tool_counts),
            "tool_counts": tool_counts,
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_time
        }

    def clear(self) -> None:
        """Clear all memory."""
        self._memory = []

    def to_json(self) -> str:
        """Serialize memory to JSON.

        Returns:
            JSON string
        """
        data = [
            {
                "tool_name": e.tool_name,
                "input_args": e.input_args,
                "output": str(e.output)[:500],  # Truncate
                "timestamp": e.timestamp,
                "success": e.success,
                "execution_time_ms": e.execution_time_ms
            }
            for e in self._memory
        ]
        return json.dumps(data, indent=2)

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._memory)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self._memory)
