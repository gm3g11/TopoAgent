"""TopoAgent Main Agent Class.

Combines MedRAX's Agent pattern with EndoAgent's dual-memory reflection mechanism.
This is the main interface for using TopoAgent.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file (following MedRAX pattern)
load_dotenv()

from .state import TopoAgentState, create_initial_state, ReflectionEntry
from .workflow import TopoAgentWorkflow
from .reflection import ReflectionEngine, DualMemoryManager
from .prompts import SYSTEM_PROMPT


class TopoAgent:
    """Main TopoAgent class for medical image classification using TDA.

    Combines:
    - MedRAX: LangGraph workflow structure
    - EndoAgent: Dual-memory mechanism + reflection loop
    - TDA Tools: 15 specialized topological analysis tools

    Example:
        >>> from topoagent import TopoAgent
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> model = ChatOpenAI(model="gpt-4o")
        >>> agent = TopoAgent(model=model)
        >>>
        >>> result = agent.classify(
        ...     image_path="path/to/image.png",
        ...     query="Classify this dermoscopy image"
        ... )
        >>> print(result["classification"])
        >>> print(result["confidence"])
    """

    def __init__(
        self,
        model,
        tools: Optional[Dict[str, Any]] = None,
        max_rounds: int = 4,
        log_tools: bool = True,
        log_dir: str = "topo_logs",
        system_prompt: Optional[str] = None,
        checkpointer=None,
        skills_mode: bool = False,
        agentic_mode: bool = False,
        agentic_v8: bool = False,
        agentic_v9: bool = False,
        workflow_version: str = "v4",
        verify_mode: str = "quick",
        time_limit_seconds: float = 60.0,
        # v8.1 ablation flags
        ablate_skills: bool = False,
        ablate_memory: bool = False,
        ablate_reflect: bool = False,
        ablate_analyze: bool = False,
    ):
        """Initialize TopoAgent.

        Args:
            model: LangChain chat model (e.g., ChatOpenAI, ChatAnthropic)
            tools: Dictionary of tool_name -> tool object. If None, initializes default TDA tools.
            max_rounds: Maximum reasoning rounds (default 4 for v2 pipeline)
            log_tools: Whether to log tool executions
            log_dir: Directory for logs
            system_prompt: Custom system prompt (uses default if None)
            checkpointer: LangGraph checkpointer for state persistence
            skills_mode: Enable skills-based pipeline with benchmark-validated rules
            agentic_mode: Enable v6 agentic 3-phase ReAct (LLM drives tool execution)
            agentic_v8: Enable v8 5-phase genuinely agentic pipeline
            workflow_version: "v4" (default, backward-compatible) or "v5" (reasoning-first)
            verify_mode: For v5: "quick" (LLM reasoning) or "thorough" (TabPFN on reference batch)
            time_limit_seconds: Max wall-clock time for agentic pipeline (default 60s)
            ablate_skills: C1 ablation — remove benchmark rankings from PLAN
            ablate_memory: C2 ablation — remove LTM + STM from prompts
            ablate_reflect: C3 ablation — skip REFLECT phase entirely
            ablate_analyze: C4 ablation — skip ANALYZE phase
        """
        self.model = model
        self.max_rounds = max_rounds
        self.log_tools = log_tools
        self.log_dir = Path(log_dir)
        self.skills_mode = skills_mode
        self.agentic_mode = agentic_mode
        self.agentic_v8 = agentic_v8
        self.agentic_v9 = agentic_v9
        self.workflow_version = workflow_version

        # v9 implies v8 implies agentic
        if agentic_v9:
            self.agentic_v8 = True
            agentic_v8 = True
            self.agentic_mode = True
            agentic_mode = True

        # v8 implies agentic
        if agentic_v8:
            self.agentic_mode = True
            agentic_mode = True

        if system_prompt:
            self.system_prompt = system_prompt
        elif skills_mode or agentic_mode:
            from .prompts import SKILLS_SYSTEM_PROMPT
            self.system_prompt = SKILLS_SYSTEM_PROMPT
        else:
            self.system_prompt = SYSTEM_PROMPT

        # Initialize tools
        if tools is None:
            if agentic_v9 or agentic_v8:
                self.tools = self._initialize_v8_tools()  # v9 uses same tool set as v8
            elif skills_mode or agentic_mode:
                self.tools = self._initialize_pipeline_tools()
            else:
                self.tools = self._initialize_default_tools()
        else:
            self.tools = tools

        # Initialize checkpointer
        self.checkpointer = checkpointer or MemorySaver()

        # v8/v9: Initialize long-term memory for cross-session learning
        self._long_term_memory = None
        if agentic_v9 or agentic_v8:
            from .memory.long_term import LongTermMemory
            ltm_tag = "v9" if agentic_v9 else "v8"
            ltm_path = os.path.join(log_dir, f"{ltm_tag}_reflections.json")
            self._long_term_memory = LongTermMemory(
                max_entries=200,
                persistence_path=ltm_path,
            )

        # Initialize workflow based on version
        if workflow_version == "v5":
            from .workflow import TopoAgentWorkflowV5
            self.workflow = TopoAgentWorkflowV5(
                model=model,
                tools=self.tools,
                verify_mode=verify_mode,
            )
        else:
            self.workflow = TopoAgentWorkflow(
                model=model,
                tools=self.tools,
                max_rounds=max_rounds,
                log_dir=log_dir,
                skills_mode=skills_mode,
                agentic_mode=agentic_mode,
                agentic_v8=agentic_v8,
                agentic_v9=agentic_v9,
                time_limit_seconds=time_limit_seconds,
                long_term_memory=self._long_term_memory,
                ablate_skills=ablate_skills,
                ablate_memory=ablate_memory,
                ablate_reflect=ablate_reflect,
                ablate_analyze=ablate_analyze,
            )

        # Initialize reflection engine
        self.reflection_engine = ReflectionEngine(model)

        # Initialize memory manager
        self.memory_manager = DualMemoryManager()

        # Session tracking
        self.session_id = None
        self.session_history = []

        # Setup logging
        if log_tools:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_default_tools(self) -> Dict[str, Any]:
        """Initialize default TDA tools.

        Returns:
            Dictionary of tool_name -> tool object
        """
        # Import tools (will be implemented later)
        tools = {}

        try:
            from .tools.preprocessing import (
                ImageLoaderTool,
                BinarizationTool,
                NoiseFilterTool
            )
            tools["image_loader"] = ImageLoaderTool()
            tools["binarization"] = BinarizationTool()
            tools["noise_filter"] = NoiseFilterTool()
        except ImportError:
            pass

        try:
            from .tools.filtration import (
                SublevelFiltrationTool,
                SuperlevelFiltrationTool,
                CubicalComplexTool
            )
            tools["sublevel_filtration"] = SublevelFiltrationTool()
            tools["superlevel_filtration"] = SuperlevelFiltrationTool()
            tools["cubical_complex"] = CubicalComplexTool()
        except ImportError:
            pass

        try:
            from .tools.homology import (
                ComputePHTool,
                PersistenceDiagramTool,
                PersistenceImageTool
            )
            tools["compute_ph"] = ComputePHTool()
            tools["persistence_diagram"] = PersistenceDiagramTool()
            tools["persistence_image"] = PersistenceImageTool()
        except ImportError:
            pass

        try:
            from .tools.features import (
                TopologicalFeaturesTool,
                WassersteinDistanceTool,
                BottleneckDistanceTool
            )
            tools["topological_features"] = TopologicalFeaturesTool()
            tools["wasserstein_distance"] = WassersteinDistanceTool()
            tools["bottleneck_distance"] = BottleneckDistanceTool()
        except ImportError:
            pass

        try:
            from .tools.classification import (
                KNNClassifierTool,
                MLPClassifierTool,
                EnsembleClassifierTool,
                PyTorchClassifierTool
            )
            tools["knn_classifier"] = KNNClassifierTool()
            tools["mlp_classifier"] = MLPClassifierTool()
            tools["ensemble_classifier"] = EnsembleClassifierTool()
            tools["pytorch_classifier"] = PyTorchClassifierTool()
        except ImportError:
            pass

        return tools

    def _initialize_pipeline_tools(self) -> Dict[str, Any]:
        """Initialize focused tool set for skills_mode (15 tools).

        Only includes tools needed for the descriptor selection pipeline:
        - image_loader (1 preprocessing tool)
        - compute_ph (1 homology tool)
        - 13 descriptor tools matching SUPPORTED_DESCRIPTORS

        This reduces the tool count from 28 to 15, which improves LLM
        tool selection accuracy by reducing decision complexity.

        Returns:
            Dictionary of tool_name -> tool object
        """
        from .skills.rules_data import SUPPORTED_DESCRIPTORS
        from .tools.descriptors import get_all_descriptors

        tools = {}

        # Preprocessing: image_loader only
        try:
            from .tools.preprocessing import ImageLoaderTool
            tools["image_loader"] = ImageLoaderTool()
        except ImportError:
            pass

        # Homology: compute_ph only
        try:
            from .tools.homology import ComputePHTool
            tools["compute_ph"] = ComputePHTool()
        except ImportError:
            pass

        # All supported descriptors
        descriptor_tools = get_all_descriptors()
        for desc_name in SUPPORTED_DESCRIPTORS:
            if desc_name in descriptor_tools:
                tools[desc_name] = descriptor_tools[desc_name]

        return tools

    def _initialize_v8_tools(self) -> Dict[str, Any]:
        """Initialize tool set for v8 agentic mode (21 tools).

        - image_loader, image_analyzer, noise_filter (preprocessing, 3)
        - compute_ph (homology, 1)
        - topological_features, betti_ratios (analysis, 2)
        - 15 descriptor tools (extraction)

        Returns:
            Dictionary of tool_name -> tool object
        """
        from .skills.rules_data import SUPPORTED_DESCRIPTORS
        from .tools.descriptors import get_all_descriptors

        tools = {}

        # Preprocessing tools (3)
        try:
            from .tools.preprocessing import ImageLoaderTool, ImageAnalyzerTool, NoiseFilterTool
            tools["image_loader"] = ImageLoaderTool()
            tools["image_analyzer"] = ImageAnalyzerTool()
            tools["noise_filter"] = NoiseFilterTool()
        except ImportError:
            pass

        # Homology (1)
        try:
            from .tools.homology import ComputePHTool
            tools["compute_ph"] = ComputePHTool()
        except ImportError:
            pass

        # Analysis tools (2)
        try:
            from .tools.features import TopologicalFeaturesTool
            tools["topological_features"] = TopologicalFeaturesTool()
        except ImportError:
            pass
        try:
            from .tools.morphology.betti_ratios import BettiRatiosTool
            tools["betti_ratios"] = BettiRatiosTool()
        except ImportError:
            pass

        # All 15 descriptor tools
        descriptor_tools = get_all_descriptors()
        for desc_name in SUPPORTED_DESCRIPTORS:
            if desc_name in descriptor_tools:
                tools[desc_name] = descriptor_tools[desc_name]

        return tools

    def classify(
        self,
        image_path: str,
        query: str = "Classify this medical image using topological features",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Classify a medical image using TDA.

        Args:
            image_path: Path to the input image
            query: Classification query/task description
            session_id: Optional session ID for state persistence

        Returns:
            Dictionary containing:
                - classification: The predicted class/label
                - confidence: Confidence score (0-100%)
                - evidence: List of topological evidence
                - tools_used: List of tools used
                - reasoning_trace: Step-by-step reasoning
                - rounds_used: Number of reasoning rounds
        """
        # Start session
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Run workflow
        final_state = self.workflow.invoke(query=query, image_path=image_path)

        # Extract results — handle both structured (skills v4.1) and text formats
        final_answer = final_state["final_answer"]
        if isinstance(final_answer, dict):
            # Structured output from optimized skills pipeline
            result = {
                "classification": final_answer.get("descriptor", "Unknown"),
                "confidence": final_state["confidence"],
                "evidence": final_state["evidence"],
                "tools_used": [t for t, _ in final_state["short_term_memory"]],
                "reasoning_trace": final_state["reasoning_trace"],
                "rounds_used": final_state.get("current_round", 1) - 1,
                "raw_answer": final_answer,
                "descriptor": final_answer.get("descriptor"),
                "report": final_answer,
                "llm_interactions": self._serialize_llm_interactions(final_state.get("llm_interactions", [])),
                "_reflect_history": final_state.get("_reflect_history", []),
                # Agentic v7/v8: expose LLM decisions for case study JSON
                "_observe_decisions": final_state.get("_observe_decisions"),
                "_benchmark_stance": final_state.get("_benchmark_stance"),
                "_observe_ph_interpretation": final_state.get("_observe_ph_interpretation"),
                "_ph_signals": final_state.get("_ph_signals"),
                # v8: additional fields
                "_v8_analysis_context": final_state.get("_v8_analysis_context"),
                "_v8_plan_context": final_state.get("_v8_plan_context"),
                "_v8_reflect_experience": final_state.get("_v8_reflect_experience"),
                "_perceive_decisions": final_state.get("_perceive_decisions"),
                # v9: hypothesis-first agentic fields
                "_v9_interpret_output": final_state.get("_v9_interpret_output"),
                "_v9_hypothesis": final_state.get("_v9_hypothesis"),
                "_v9_act_decision": final_state.get("_v9_act_decision"),
                "_v9_reflect_output": final_state.get("_v9_reflect_output"),
                "_v9_memory_stats": final_state.get("_v9_memory_stats"),
                # Full short_term_memory for feature vector extraction
                "short_term_memory": final_state["short_term_memory"],
            }
        else:
            # Text output from legacy workflow
            result = {
                "classification": self._extract_classification(final_answer),
                "confidence": final_state["confidence"],
                "evidence": final_state["evidence"],
                "tools_used": [t for t, _ in final_state["short_term_memory"]],
                "reasoning_trace": final_state["reasoning_trace"],
                "rounds_used": final_state.get("current_round", 1) - 1,
                "raw_answer": final_answer,
                "llm_interactions": self._serialize_llm_interactions(final_state.get("llm_interactions", []))
            }

        # Log if enabled
        if self.log_tools:
            self._log_session(result, final_state)

        return result

    async def aclassify(
        self,
        image_path: str,
        query: str = "Classify this medical image using topological features",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of classify.

        Args:
            image_path: Path to the input image
            query: Classification query/task description
            session_id: Optional session ID for state persistence

        Returns:
            Same as classify()
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        final_state = await self.workflow.ainvoke(query=query, image_path=image_path)

        result = {
            "classification": self._extract_classification(final_state["final_answer"]),
            "confidence": final_state["confidence"],
            "evidence": final_state["evidence"],
            "tools_used": [t for t, _ in final_state["short_term_memory"]],
            "reasoning_trace": final_state["reasoning_trace"],
            "rounds_used": final_state["current_round"] - 1,
            "raw_answer": final_state["final_answer"]
        }

        if self.log_tools:
            self._log_session(result, final_state)

        return result

    def _extract_classification(self, answer: Optional[str]) -> str:
        """Extract classification label from answer text.

        Args:
            answer: Raw answer text

        Returns:
            Extracted classification label
        """
        if not answer:
            return "Unknown"

        # Look for common classification patterns
        answer_lower = answer.lower()

        # Try to find classification/label mentions
        patterns = [
            ("classification:", 1),
            ("predicted class:", 2),
            ("label:", 1),
            ("result:", 1)
        ]

        for pattern, lines_after in patterns:
            if pattern in answer_lower:
                idx = answer_lower.find(pattern)
                after_pattern = answer[idx + len(pattern):].strip()
                # Get first line/phrase
                first_line = after_pattern.split("\n")[0].strip()
                if first_line:
                    return first_line

        # Fallback: return first line of answer
        return answer.split("\n")[0][:100]

    def _serialize_llm_interactions(self, interactions: List) -> List[Dict[str, Any]]:
        """Serialize LLM interactions for JSON output.

        Args:
            interactions: List of LLMInteraction objects

        Returns:
            List of serializable dictionaries with prompt/response details
        """
        serialized = []
        for interaction in interactions:
            entry = {
                "step": interaction.step,
                "round": interaction.round,
                "prompt": interaction.prompt,
                "response": interaction.response,
                "timestamp": interaction.timestamp
            }
            if interaction.tool_calls:
                entry["tool_calls"] = interaction.tool_calls
            serialized.append(entry)
        return serialized

    def _log_session(self, result: Dict[str, Any], state: TopoAgentState) -> None:
        """Log session details to file.

        Args:
            result: Classification result
            state: Final agent state
        """
        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "query": state["query"],
            "image_path": state["image_path"],
            "result": result,
            "short_term_memory": [
                {"tool": t, "output": str(o)[:500]}
                for t, o in state["short_term_memory"]
            ],
            "long_term_memory": [
                {
                    "round": e.round,
                    "error_analysis": e.error_analysis,
                    "suggestion": e.suggestion,
                    "experience": e.experience
                }
                for e in state["long_term_memory"]
            ],
            # Detailed LLM interactions (prompts and responses)
            "llm_interactions": self._serialize_llm_interactions(state.get("llm_interactions", []))
        }

        log_file = self.log_dir / f"{self.session_id}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

    def classify_direct(
        self,
        image_path: str,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Direct classification without LLM orchestration (23ms vs 30s).

        Executes the fixed TDA pipeline directly:
        1. image_loader → image_array
        2. compute_ph → persistence diagrams
        3. persistence_image → 800D feature vector
        4. pytorch_classifier → prediction + confidence

        This is 870x faster than the LLM-orchestrated version with identical
        accuracy. Use for production deployment.

        Args:
            image_path: Path to the input image
            model_path: Optional path to trained model (uses default if None)

        Returns:
            Dictionary containing:
                - predicted_class: Class name
                - class_id: Class index (0-6)
                - confidence: Confidence score (0-100%)
                - probabilities: Per-class probabilities
                - latency_ms: Inference time in milliseconds
        """
        import time
        start_time = time.time()

        try:
            # Step 1: Load image
            if "image_loader" not in self.tools:
                return {"success": False, "error": "image_loader tool not available"}

            img_result = self.tools["image_loader"]._run(image_path=image_path)
            if not img_result.get("success", False):
                return {"success": False, "error": f"Image loading failed: {img_result.get('error', 'unknown')}"}

            # Step 2: Compute persistent homology
            if "compute_ph" not in self.tools:
                return {"success": False, "error": "compute_ph tool not available"}

            ph_result = self.tools["compute_ph"]._run(image_array=img_result["image_array"])
            if not ph_result.get("success", False):
                return {"success": False, "error": f"PH computation failed: {ph_result.get('error', 'unknown')}"}

            # Step 3: Generate persistence image feature vector
            if "persistence_image" not in self.tools:
                return {"success": False, "error": "persistence_image tool not available"}

            # Extract persistence data for persistence image
            persistence_data = ph_result.get("persistence", {})

            pi_result = self.tools["persistence_image"]._run(
                persistence_data=persistence_data
            )
            if not pi_result.get("success", False):
                return {"success": False, "error": f"PI generation failed: {pi_result.get('error', 'unknown')}"}

            # Step 4: Classify using PyTorch MLP
            if "pytorch_classifier" not in self.tools:
                return {"success": False, "error": "pytorch_classifier tool not available"}

            # Prepare classifier kwargs (tool returns combined_vector)
            classifier_kwargs = {"feature_vector": pi_result["combined_vector"]}
            if model_path:
                classifier_kwargs["model_path"] = model_path

            class_result = self.tools["pytorch_classifier"]._run(**classifier_kwargs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            if class_result.get("success", False):
                return {
                    "success": True,
                    "predicted_class": class_result.get("predicted_class", "unknown"),
                    "class_id": class_result.get("class_id", -1),
                    "confidence": class_result.get("confidence", 0.0),
                    "probabilities": class_result.get("probabilities", []),
                    "latency_ms": round(latency_ms, 2),
                    "mode": "direct"
                }
            else:
                return {
                    "success": False,
                    "error": f"Classification failed: {class_result.get('error', 'unknown')}",
                    "latency_ms": round(latency_ms, 2)
                }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "latency_ms": round(latency_ms, 2)
            }

    def get_available_tools(self) -> List[str]:
        """Get list of available tools.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools.

        Returns:
            Dictionary of tool_name -> description
        """
        return {
            name: tool.description if hasattr(tool, "description") else "No description"
            for name, tool in self.tools.items()
        }

    def add_tool(self, name: str, tool: Any) -> None:
        """Add a new tool to the agent.

        Args:
            name: Tool name
            tool: Tool object (must be a LangChain BaseTool)
        """
        self.tools[name] = tool
        # Rebuild workflow with new tools
        self.workflow = TopoAgentWorkflow(
            model=self.model,
            tools=self.tools,
            max_rounds=self.max_rounds,
            log_dir=str(self.log_dir)
        )

    def get_session_summary(self) -> str:
        """Get a summary of the current/last session.

        Returns:
            Session summary string
        """
        if not self.session_history:
            return "No session history available."

        last_session = self.session_history[-1]
        return self.memory_manager.summarize_session(
            short_term=last_session.get("short_term_memory", []),
            long_term=last_session.get("long_term_memory", [])
        )


def create_topoagent(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_rounds: int = 4,
    tools: Optional[Dict[str, Any]] = None,
    skills_mode: bool = False,
    agentic_mode: bool = False,
    agentic_v8: bool = False,
    agentic_v9: bool = False,
    workflow_version: str = "v4",
    verify_mode: str = "quick",
    time_limit_seconds: float = 60.0,
    # v8.1 ablation flags
    ablate_skills: bool = False,
    ablate_memory: bool = False,
    ablate_reflect: bool = False,
    ablate_analyze: bool = False,
) -> TopoAgent:
    """Factory function to create a TopoAgent instance with OpenAI.

    Follows MedRAX pattern for OpenAI configuration. Reads from environment
    variables if parameters are not explicitly provided.

    Environment Variables:
        OPENAI_API_KEY: OpenAI API key
        OPENAI_MODEL: Model name (default: gpt-4o)
        OPENAI_BASE_URL: Optional base URL for API (for alternative providers)

    Args:
        model_name: Name of the LLM model (default: from OPENAI_MODEL or gpt-4o)
        api_key: API key (default: from OPENAI_API_KEY)
        base_url: Base URL for API (default: from OPENAI_BASE_URL)
        temperature: Model temperature (default: 0.7)
        top_p: Top-p sampling (default: 0.95)
        max_rounds: Maximum reasoning rounds (default: 4 for v2 pipeline)
        tools: Optional custom tools dict
        skills_mode: Enable skills-based pipeline with benchmark-validated rules
        agentic_mode: Enable v6 agentic 3-phase ReAct (LLM drives tool execution)
        agentic_v8: Enable v8 5-phase genuinely agentic pipeline
        agentic_v9: Enable v9 6-phase hypothesis-first agentic pipeline
        workflow_version: "v4" (default) or "v5" (reasoning-first, 2-3 LLM calls)
        verify_mode: For v5: "quick" (LLM reasoning) or "thorough" (TabPFN CV)
        time_limit_seconds: Max wall-clock time for agentic pipeline (default 60s)
        ablate_skills: C1 ablation — remove benchmark rankings from PLAN
        ablate_memory: C2 ablation — remove LTM + STM from prompts
        ablate_reflect: C3 ablation — skip REFLECT phase entirely
        ablate_analyze: C4 ablation — skip ANALYZE phase

    Returns:
        Configured TopoAgent instance

    Example:
        >>> # v8 agentic mode (5-phase genuinely agentic)
        >>> agent = create_topoagent(agentic_v8=True, time_limit_seconds=120)
        >>> result = agent.classify("blood_cell.png")

        >>> # v7 agentic mode (3-phase, backward compatible)
        >>> agent = create_topoagent(agentic_mode=True)
        >>> result = agent.classify("blood_cell.png")
    """
    from langchain_openai import ChatOpenAI

    # Get from environment if not provided (MedRAX pattern)
    if model_name is None:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL")

    # Build kwargs for ChatOpenAI
    openai_kwargs = {}
    if api_key:
        openai_kwargs["api_key"] = api_key
    if base_url:
        openai_kwargs["base_url"] = base_url

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        **openai_kwargs
    )

    return TopoAgent(
        model=model,
        tools=tools,
        max_rounds=max_rounds,
        skills_mode=skills_mode,
        agentic_mode=agentic_mode,
        agentic_v8=agentic_v8,
        agentic_v9=agentic_v9,
        workflow_version=workflow_version,
        verify_mode=verify_mode,
        time_limit_seconds=time_limit_seconds,
        ablate_skills=ablate_skills,
        ablate_memory=ablate_memory,
        ablate_reflect=ablate_reflect,
        ablate_analyze=ablate_analyze,
    )


def create_topoagent_ollama(
    model_name: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    max_rounds: int = 4,
    tools: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1
) -> TopoAgent:
    """Factory function to create a TopoAgent instance with Ollama (free local LLM).

    Uses Ollama for local LLM inference with no API costs. Requires Ollama to be
    installed and running locally.

    Installation:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3.1:8b

    Args:
        model_name: Ollama model name (default: llama3.1:8b)
            Recommended models:
            - llama3.2:3b: Fast, lightweight (2GB), good for development
            - llama3.1:8b: Balanced quality/speed (4.7GB), recommended for experiments
            - mistral:7b: Alternative, good tool support (4GB)
        base_url: Ollama server URL (default: http://localhost:11434)
        max_rounds: Maximum reasoning rounds
        tools: Optional custom tools dict
        temperature: Model temperature (default: 0.1 for consistency)

    Returns:
        Configured TopoAgent instance

    Example:
        >>> # Start Ollama first: ollama serve
        >>> agent = create_topoagent_ollama(model_name="llama3.1:8b")
        >>> result = agent.classify("image.png")
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain_ollama not installed. Install with: pip install langchain-ollama"
        )

    model = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )

    return TopoAgent(
        model=model,
        tools=tools,
        max_rounds=max_rounds
    )


def create_topoagent_claude(
    model_name: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_rounds: int = 4,
    tools: Optional[Dict[str, Any]] = None,
    skills_mode: bool = False,
    agentic_mode: bool = False,
    agentic_v8: bool = False,
    agentic_v9: bool = False,
    workflow_version: str = "v4",
    verify_mode: str = "quick",
    time_limit_seconds: float = 60.0,
    ablate_skills: bool = False,
    ablate_memory: bool = False,
    ablate_reflect: bool = False,
    ablate_analyze: bool = False,
) -> TopoAgent:
    """Factory function to create a TopoAgent instance with Anthropic Claude.

    Environment Variables:
        ANTHROPIC_API_KEY: Anthropic API key

    Args:
        model_name: Claude model name (default: claude-sonnet-4-20250514)
            Recommended models:
            - claude-sonnet-4-20250514: Best balance of quality/speed/cost
            - claude-opus-4-20250514: Highest quality, slower
            - claude-haiku-3-5-20241022: Fastest, cheapest
        api_key: API key (default: from ANTHROPIC_API_KEY)
        temperature: Model temperature (default: 0.7)
        max_rounds: Maximum reasoning rounds
        tools: Optional custom tools dict
        skills_mode: Enable skills-based pipeline
        agentic_mode: Enable v7 agentic mode
        agentic_v8: Enable v8 agentic pipeline
        workflow_version: "v4" (default) or "v5"
        verify_mode: For v5: "quick" or "thorough"
        time_limit_seconds: Max wall-clock time (default 60s)
        ablate_skills: C1 ablation
        ablate_memory: C2 ablation
        ablate_reflect: C3 ablation
        ablate_analyze: C4 ablation

    Returns:
        Configured TopoAgent instance

    Example:
        >>> agent = create_topoagent_claude(agentic_v8=True)
        >>> result = agent.classify("blood_cell.png")
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain_anthropic not installed. Install with: "
            "pip install langchain-anthropic"
        )

    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Get one at https://console.anthropic.com/settings/keys "
            "and add ANTHROPIC_API_KEY=sk-ant-... to your .env file."
        )

    model = ChatAnthropic(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=4096,
    )

    return TopoAgent(
        model=model,
        tools=tools,
        max_rounds=max_rounds,
        skills_mode=skills_mode,
        agentic_mode=agentic_mode,
        agentic_v8=agentic_v8,
        agentic_v9=agentic_v9,
        workflow_version=workflow_version,
        verify_mode=verify_mode,
        time_limit_seconds=time_limit_seconds,
        ablate_skills=ablate_skills,
        ablate_memory=ablate_memory,
        ablate_reflect=ablate_reflect,
        ablate_analyze=ablate_analyze,
    )


def create_topoagent_gemini(
    model_name: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_rounds: int = 4,
    tools: Optional[Dict[str, Any]] = None,
    skills_mode: bool = False,
    agentic_mode: bool = False,
    agentic_v8: bool = False,
    agentic_v9: bool = False,
    workflow_version: str = "v4",
    verify_mode: str = "quick",
    time_limit_seconds: float = 60.0,
    ablate_skills: bool = False,
    ablate_memory: bool = False,
    ablate_reflect: bool = False,
    ablate_analyze: bool = False,
) -> TopoAgent:
    """Factory function to create a TopoAgent instance with Google Gemini.

    Environment Variables:
        GOOGLE_API_KEY: Google AI Studio API key

    Args:
        model_name: Gemini model name (default: gemini-2.5-pro-preview-06-05)
            Recommended models:
            - gemini-2.5-pro-preview-06-05: Latest Gemini 2.5 Pro
            - gemini-2.5-flash-preview-05-20: Faster, cheaper Gemini 2.5
            - gemini-2.0-flash: Fast, production-ready
        api_key: API key (default: from GOOGLE_API_KEY)
        temperature: Model temperature (default: 0.7)
        max_rounds: Maximum reasoning rounds
        tools: Optional custom tools dict
        skills_mode: Enable skills-based pipeline
        agentic_mode: Enable v7 agentic mode
        agentic_v8: Enable v8 agentic pipeline
        workflow_version: "v4" (default) or "v5"
        verify_mode: For v5: "quick" or "thorough"
        time_limit_seconds: Max wall-clock time (default 60s)
        ablate_skills: C1 ablation
        ablate_memory: C2 ablation
        ablate_reflect: C3 ablation
        ablate_analyze: C4 ablation

    Returns:
        Configured TopoAgent instance

    Example:
        >>> agent = create_topoagent_gemini(agentic_v8=True)
        >>> result = agent.classify("blood_cell.png")
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain_google_genai not installed. Install with: "
            "pip install langchain-google-genai"
        )

    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Get one at https://aistudio.google.com/apikey "
            "and add GOOGLE_API_KEY=AI... to your .env file."
        )

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=4096,
    )

    return TopoAgent(
        model=model,
        tools=tools,
        max_rounds=max_rounds,
        skills_mode=skills_mode,
        agentic_mode=agentic_mode,
        agentic_v8=agentic_v8,
        agentic_v9=agentic_v9,
        workflow_version=workflow_version,
        verify_mode=verify_mode,
        time_limit_seconds=time_limit_seconds,
        ablate_skills=ablate_skills,
        ablate_memory=ablate_memory,
        ablate_reflect=ablate_reflect,
        ablate_analyze=ablate_analyze,
    )


# ---- Provider auto-detection ------------------------------------------------

# Mapping of model name prefixes to providers
_MODEL_PROVIDER_MAP = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
}


def detect_provider(model_name: str) -> str:
    """Detect LLM provider from model name.

    Returns one of: 'openai', 'anthropic', 'google'.
    Raises ValueError if the provider cannot be determined.
    """
    for prefix, provider in _MODEL_PROVIDER_MAP.items():
        if model_name.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot detect provider for model '{model_name}'. "
        f"Known prefixes: {list(_MODEL_PROVIDER_MAP.keys())}. "
        f"Use create_topoagent(), create_topoagent_claude(), or "
        f"create_topoagent_gemini() directly."
    )


def create_topoagent_auto(
    model_name: str = "gpt-4o",
    **kwargs,
) -> TopoAgent:
    """Auto-detect provider from model name and create a TopoAgent.

    Convenience wrapper that routes to the correct factory function
    based on the model name prefix.

    Args:
        model_name: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514",
                    "gemini-2.5-pro-preview-06-05")
        **kwargs: Passed to the underlying factory function.

    Returns:
        Configured TopoAgent instance

    Example:
        >>> agent = create_topoagent_auto("claude-sonnet-4-20250514", agentic_v8=True)
        >>> agent = create_topoagent_auto("gemini-2.5-pro-preview-06-05", agentic_v8=True)
        >>> agent = create_topoagent_auto("gpt-4o", agentic_v8=True)
    """
    provider = detect_provider(model_name)

    if provider == "openai":
        return create_topoagent(model_name=model_name, **kwargs)
    elif provider == "anthropic":
        return create_topoagent_claude(model_name=model_name, **kwargs)
    elif provider == "google":
        return create_topoagent_gemini(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")