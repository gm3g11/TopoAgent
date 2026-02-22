"""TopoAgent: Medical AI Agent for Topological Data Analysis.

TopoAgent combines:
- MedRAX: LangGraph workflow structure + ReAct loop
- EndoAgent: Dual-memory mechanism + reflection loop
- TDA Tools: 27 specialized topological analysis tools

Example (OpenAI):
    >>> from topoagent import create_topoagent
    >>> agent = create_topoagent(model_name="gpt-4o")
    >>> result = agent.classify("path/to/image.png")

Example (Ollama - Free Local LLM):
    >>> from topoagent import create_topoagent_ollama
    >>> # Requires: ollama serve & ollama pull llama3.1:8b
    >>> agent = create_topoagent_ollama(model_name="llama3.1:8b")
    >>> result = agent.classify("path/to/image.png")

Example (Custom Configuration):
    >>> from topoagent import TopoAgent
    >>> from langchain_openai import ChatOpenAI
    >>> model = ChatOpenAI(model="gpt-4o")
    >>> agent = TopoAgent(model=model, max_rounds=3)
    >>> result = agent.classify(
    ...     image_path="dermoscopy.png",
    ...     query="Classify this skin lesion using topological features"
    ... )
    >>> print(result["classification"], result["confidence"])
"""

__version__ = "0.1.0"
__author__ = "TopoAgent Team"

from .agent import TopoAgent, create_topoagent, create_topoagent_ollama
from .state import TopoAgentState, create_initial_state, ReflectionEntry
from .workflow import TopoAgentWorkflow, create_topoagent_workflow
from .reflection import ReflectionEngine, DualMemoryManager
from .memory import ShortTermMemory, LongTermMemory

__all__ = [
    # Main classes
    "TopoAgent",
    "create_topoagent",
    "create_topoagent_ollama",
    # State
    "TopoAgentState",
    "create_initial_state",
    "ReflectionEntry",
    # Workflow
    "TopoAgentWorkflow",
    "create_topoagent_workflow",
    # Reflection
    "ReflectionEngine",
    "DualMemoryManager",
    # Memory
    "ShortTermMemory",
    "LongTermMemory",
]
