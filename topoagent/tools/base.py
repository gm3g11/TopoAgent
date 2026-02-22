"""Base class for TopoAgent TDA tools.

All TDA tools inherit from this base class which provides:
- Consistent input/output format
- Error handling
- Logging support
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun


class TopoToolInput(BaseModel):
    """Base input schema for TDA tools."""
    pass


class TopoBaseTool(BaseTool, ABC):
    """Base class for all TopoAgent TDA tools.

    Inherits from LangChain's BaseTool for compatibility with the agent framework.
    """

    name: str = "topo_base_tool"
    description: str = "Base TDA tool"
    args_schema: Type[BaseModel] = TopoToolInput
    return_direct: bool = False

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the tool synchronously.

        Args:
            run_manager: Callback manager for tool run
            **kwargs: Tool-specific arguments

        Returns:
            Dictionary containing tool output
        """
        try:
            result = self._execute(**kwargs)
            return {
                "success": True,
                "tool_name": self.name,
                "output": result
            }
        except Exception as e:
            return {
                "success": False,
                "tool_name": self.name,
                "error": str(e)
            }

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the tool asynchronously.

        Default implementation calls synchronous version.
        Override for true async support.
        """
        return self._run(**kwargs)

    @abstractmethod
    def _execute(self, **kwargs: Any) -> Any:
        """Tool-specific execution logic.

        Override this method in subclasses.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool-specific output
        """
        pass
