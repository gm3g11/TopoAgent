"""TopoAgent LangGraph Workflow.

Implements the ReAct + Reflection workflow combining:
- MedRAX: LangGraph StateGraph pattern
- EndoAgent: Algorithm 1 with dual-memory and reflection loop

Workflow:
    for t = 1 to N (max 3):
        tool_t <- SelectTool(context, Ms, Ml, T)
        output_t <- tool_t.invoke(context)
        Ms <- Ms ∪ {(tool_t, output_t)}
        reflection_t <- LLM_reflection(context, Ms, Ml)
        Ml <- Ml ∪ {reflection_t}
        if IsTaskComplete: return output_t
        context <- UpdateContext(context, output_t, reflection_t)
"""

from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .state import (
    TopoAgentState,
    TopoAgentStateV5,
    AgentReport,
    ReflectionEntry,
    LLMInteraction,
    create_initial_state,
    create_initial_state_v5,
    update_short_term_memory,
    update_long_term_memory,
    format_short_term_memory,
    format_long_term_memory,
    format_skill_context,
    truncate_output_for_prompt,
    # v2 Auto-injection helpers
    get_persistence_data,
    get_feature_vector,
    get_image_array,
    # v3 Adaptive helpers
    get_image_analyzer_results,
    get_recommended_filtration,
    get_recommended_sigma,
)
from .prompts import (
    SYSTEM_PROMPT,
    TOOL_SELECTION_PROMPT,
    REFLECTION_PROMPT,
    COMPLETION_CHECK_PROMPT,
    FINAL_ANSWER_PROMPT,
    ADAPTIVE_TOOL_SELECTION_PROMPT,  # v3
    ADAPTIVE_REFLECTION_PROMPT,  # v3
    SKILLS_SYSTEM_PROMPT,  # v4 Skills (reasoning-based)
    SKILLS_TOOL_SELECTION_PROMPT,  # v4 Skills (reasoning-based)
    SKILLS_REFLECTION_PROMPT,  # v4 Skills (reasoning-based)
    SKILLS_PLAN_PROMPT,  # v4.1 optimized skills pipeline
    SKILLS_VERIFY_PROMPT,  # v4.1 optimized skills pipeline
    ANALYZE_AND_PLAN_PROMPT,  # v5
    VERIFY_REASONING_PROMPT,  # v5
    OUTPUT_REPORT_PROMPT,  # v5
    AGENTIC_OBSERVE_R1_PROMPT,  # v7 agentic (JSON decisions)
    AGENTIC_OBSERVE_R2_PROMPT,  # v7 agentic (PH interpretation)
    AGENTIC_ACT_PROMPT,  # v7 agentic (FOLLOW/DEVIATE)
    AGENTIC_REFLECT_PROMPT,  # v7 agentic (with decisions header)
    V8_PERCEIVE_PROMPT,  # v8 agentic (multi-turn perception)
    V8_PERCEIVE_DECIDE_PROMPT,  # v8.1 agentic (LLM decides filtration + denoising)
    V8_ANALYZE_PROMPT,   # v8 agentic (synthesize perception)
    V8_PLAN_PROMPT,      # v8 agentic (descriptor selection with LTM)
    V8_REFLECT_PROMPT,   # v8 agentic (LLM-driven reflection + memory write)
    V9_INTERPRET_PROMPT,   # v9 agentic (blind interpretation)
    V9_ANALYZE_PROMPT,     # v9 agentic (hypothesis formation)
    V9_ACT_PROMPT,         # v9 agentic (reconcile and decide)
    V9_REFLECT_PROMPT,     # v9 agentic (evaluate and learn)
    format_tool_descriptions,
)
from .skills.rules_data import (
    SUPPORTED_DESCRIPTORS, ALL_DESCRIPTORS, TOP_PERFORMERS, get_descriptor_dim,
    get_expected_quality, build_quality_assessment_text,
    # v9 helpers
    OBJECT_TYPE_TAXONOMY, PARAMETER_REASONING_GUIDE,
    build_descriptor_properties_only, build_ph_signal_observations,
    build_reasoning_principles, build_parameter_reasoning_text,
    build_benchmark_advisory, build_reference_quality_ranges,
    # v9 helpers (tiered advisory, domain context)
    build_tiered_benchmark_advisory, extract_domain_context,
)


class TopoAgentWorkflow:
    """LangGraph workflow for TopoAgent.

    Implements the dual-memory ReAct + Reflection pattern from EndoAgent.
    """

    def __init__(
        self,
        model,
        tools: Dict[str, Any],
        max_rounds: int = 4,
        log_dir: Optional[str] = None,
        enable_reflection: bool = True,
        enable_short_memory: bool = True,
        enable_long_memory: bool = True,
        adaptive_mode: bool = False,  # v3: Enable adaptive pipeline
        skills_mode: bool = False,    # v4: Enable skills-based pipeline
        agentic_mode: bool = False,   # v6: Enable agentic 3-phase ReAct
        agentic_v8: bool = False,     # v8: Enable 5-phase genuinely agentic
        agentic_v9: bool = False,     # v9: Enable 6-phase hypothesis-first agentic
        verbose: bool = False,        # Print all LLM interactions to stdout
        time_limit_seconds: float = 60.0,  # Max wall-clock time for agentic pipeline
        long_term_memory=None,        # v8: Shared LongTermMemory instance
        # v8.1 ablation flags
        ablate_skills: bool = False,  # C1: remove benchmark rankings from PLAN
        ablate_memory: bool = False,  # C2: remove LTM + STM from prompts
        ablate_reflect: bool = False, # C3: skip REFLECT phase entirely
        ablate_analyze: bool = False, # C4: skip ANALYZE phase
    ):
        """Initialize the workflow.

        Args:
            model: LangChain chat model (e.g., ChatOpenAI)
            tools: Dictionary of tool_name -> tool object
            max_rounds: Maximum reasoning rounds (default 4 for v2, 5 for v3)
            log_dir: Optional directory for logging
            enable_reflection: Enable reflection mechanism (ablation flag)
            enable_short_memory: Enable short-term memory (ablation flag)
            enable_long_memory: Enable long-term memory (ablation flag)
            adaptive_mode: Enable v3 adaptive pipeline with image_analyzer
            skills_mode: Enable v4 skills-based pipeline with benchmark rules
            agentic_mode: Enable v6 agentic 3-phase ReAct (LLM drives tool execution)
            agentic_v8: Enable v8 5-phase genuinely agentic pipeline
            agentic_v9: Enable v9 6-phase hypothesis-first agentic pipeline
            verbose: If True, print full prompts, responses, and tool outputs to stdout
            time_limit_seconds: Max wall-clock time for agentic pipeline (default 60s)
            long_term_memory: Optional LongTermMemory instance for cross-session learning
        """
        self.model = model
        self.tools = tools
        self.max_rounds = max_rounds if not adaptive_mode else max(max_rounds, 5)
        self.log_dir = log_dir
        self.verbose = verbose

        # Ablation flags
        self.enable_reflection = enable_reflection
        self.enable_short_memory = enable_short_memory
        self.enable_long_memory = enable_long_memory

        # v3: Adaptive mode
        self.adaptive_mode = adaptive_mode

        # v4: Skills mode
        self.skills_mode = skills_mode

        # v6: Agentic mode
        self.agentic_mode = agentic_mode
        self.agentic_v8 = agentic_v8
        if agentic_v8:
            agentic_mode = True
            self.agentic_mode = True
        self.agentic_v9 = agentic_v9
        if agentic_v9:
            agentic_mode = True
            self.agentic_mode = True
        self.time_limit_seconds = time_limit_seconds

        # v8: Long-term memory for cross-session learning
        self.long_term_memory = long_term_memory

        # v8.1 ablation flags
        self.ablate_skills = ablate_skills
        self.ablate_memory = ablate_memory
        self.ablate_reflect = ablate_reflect
        self.ablate_analyze = ablate_analyze

        # Skills registry needed for both skills_mode and agentic_mode
        if skills_mode or agentic_mode:
            from .skills import SkillRegistry
            self.skill_registry = SkillRegistry()

        # Bind tools to model (only needed for non-skills-mode workflow).
        # The optimized skills pipeline uses plain LLM calls (no tool binding).
        self.model_with_tools = model.bind_tools(list(tools.values()))

        # v6: Per-phase tool binding for agentic mode (v7 backward compat)
        if agentic_mode and not agentic_v8:
            observe_tool_list = [tools[t] for t in ["image_loader", "compute_ph"]
                                 if t in tools]
            act_tool_list = [tools[t] for t in SUPPORTED_DESCRIPTORS
                             if t in tools]
            self.observe_model = model.bind_tools(observe_tool_list)
            self.act_model = model.bind_tools(act_tool_list)

        # v8: Rich per-phase tool binding
        if agentic_v8:
            perceive_tool_names = [
                "image_loader", "image_analyzer", "noise_filter", "compute_ph",
                "topological_features", "persistence_diagram", "betti_ratios",
                "euler_characteristic", "total_persistence_stats",
            ]
            perceive_tool_list = [tools[t] for t in perceive_tool_names if t in tools]
            extract_tool_list = [tools[t] for t in SUPPORTED_DESCRIPTORS if t in tools]
            self.perceive_model = model.bind_tools(perceive_tool_list)
            self.extract_model = model.bind_tools(extract_tool_list)

        # Build the workflow graph
        if agentic_v9:
            self.workflow = self._build_agentic_v9_workflow()
        elif agentic_v8:
            self.workflow = self._build_agentic_v8_workflow()
        elif agentic_mode:
            self.workflow = self._build_agentic_workflow()
        elif skills_mode:
            self.workflow = self._build_skills_workflow()
        else:
            self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Supports ablation by conditionally including/excluding nodes.
        In skills_mode, knowledge is injected into _select_tool prompts
        (no separate _apply_skills node needed).

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(TopoAgentState)

        # Add core nodes (always present)
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("select_tool", self._select_tool)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("update_short_memory", self._update_short_memory)
        workflow.add_node("check_completion", self._check_completion)
        workflow.add_node("generate_answer", self._generate_answer)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        # Edges (same for all modes — skills knowledge injected in _select_tool)
        workflow.add_edge("analyze_query", "select_tool")

        workflow.add_edge("select_tool", "execute_tool")
        workflow.add_edge("execute_tool", "update_short_memory")

        # Conditional: Add reflection nodes if enabled
        if self.enable_reflection:
            workflow.add_node("reflect", self._reflect)
            workflow.add_node("update_long_memory", self._update_long_memory)
            workflow.add_node("handle_recovery", self._handle_recovery)

            # Full workflow with reflection and error recovery
            workflow.add_edge("update_short_memory", "reflect")
            workflow.add_edge("reflect", "update_long_memory")

            # Conditional: Check if recovery action is needed
            workflow.add_conditional_edges(
                "update_long_memory",
                self._should_recover,
                {
                    "recover": "handle_recovery",
                    "continue": "check_completion"
                }
            )
            workflow.add_edge("handle_recovery", "check_completion")
        else:
            # Simplified workflow without reflection
            workflow.add_edge("update_short_memory", "check_completion")

        # Conditional edges for continuation
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "select_tool",
                "finish": "generate_answer"
            }
        )

        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # =========================================================================
    # v6 Agentic Pipeline: 3-Phase ReAct (Observe → Act → Reflect)
    # =========================================================================

    def _build_agentic_workflow(self) -> Any:
        """Build the agentic 3-phase LangGraph workflow (v7: genuinely agentic).

        Architecture (3-5 LLM calls):
            OBSERVE R1 (LLM returns JSON decisions, no tools)
            → execute_observe_tools (programmatic image_loader + compute_ph)
            → OBSERVE R2 (LLM interprets PH, no tools)
            → ACT (LLM selects descriptor with FOLLOW/DEVIATE stance)
            → act_tools (execute chosen descriptor)
            → REFLECT (LLM evaluates quality)
            → END or retry back to ACT/OBSERVE

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(TopoAgentState)

        # Phase 1: Observe (2 rounds: JSON decisions → PH interpretation)
        workflow.add_node("observe", self._agentic_observe)
        workflow.add_node("execute_observe_tools", self._agentic_execute_observe_tools)

        # Phase 2: Act
        workflow.add_node("act", self._agentic_act)
        workflow.add_node("act_tools", self._agentic_execute_tools)

        # Phase 3: Reflect
        workflow.add_node("reflect", self._agentic_reflect)

        # Phase 1 edges
        workflow.set_entry_point("observe")
        workflow.add_conditional_edges("observe", self._route_observe,
            {"execute_observe_tools": "execute_observe_tools", "act": "act"})
        workflow.add_edge("execute_observe_tools", "observe")  # back to observe R2

        # Phase 2 edges
        workflow.add_conditional_edges("act", self._route_act,
            {"tools": "act_tools", "next": "reflect"})
        workflow.add_conditional_edges("act_tools", self._route_after_act_tools,
            {"act": "act", "reflect": "reflect"})

        # Phase 3 edges
        workflow.add_conditional_edges("reflect", self._route_reflect,
            {"complete": END, "retry_descriptor": "act", "retry_ph": "observe"})

        return workflow.compile()

    def _agentic_observe(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 1: LLM makes decisions via JSON (R1) then interprets PH (R2).

        Round 1 (observe_turns==1): LLM returns JSON with object_type, color_mode,
        filtration_type. No tools bound — decisions only.

        Round 2 (observe_turns==2): After programmatic tool execution, LLM interprets
        PH results and gives pre-benchmark descriptor intuition. No tools.
        """
        import time
        import json as json_mod
        from .skills.rules_data import (
            build_color_mode_advisory,
            DATASET_TO_OBJECT_TYPE,
        )

        if not hasattr(self, '_workflow_start_time'):
            self._workflow_start_time = time.time()

        observe_turns = state.get("_observe_turns", 0) + 1
        self._vprint_header("OBSERVE", round_num=observe_turns)

        new_interactions = list(state.get("llm_interactions", []))

        if observe_turns == 1:
            # === OBSERVE R1: JSON decisions only (no tools bound) ===

            # Build retry context if coming back from reflect
            retry_context = ""
            retry_feedback = state.get("_retry_feedback")
            if retry_feedback:
                retry_context = f"## Previous Attempt Feedback\n{retry_feedback}\n"

            # Determine n_channels from image path
            n_channels = 3  # default
            try:
                from PIL import Image as PILImage
                img = PILImage.open(state["image_path"])
                n_channels = len(img.getbands())
            except Exception:
                pass

            color_advisory = build_color_mode_advisory(n_channels)

            prompt = AGENTIC_OBSERVE_R1_PROMPT.format(
                query=state["query"],
                image_path=state["image_path"],
                retry_context=retry_context,
                n_channels=n_channels,
                color_mode_advisory=color_advisory,
            )

            prompt_msg = HumanMessage(content=prompt)
            self._vprint_prompt(prompt, max_chars=3000)

            # Call LLM WITHOUT tools — JSON response only
            response = self.model.invoke([prompt_msg])
            response_text = response.content if hasattr(response, "content") else str(response)

            self._vprint_response(response_text)

            # Parse JSON decisions
            decisions = self._parse_observe_decisions(response_text)
            decisions["_n_channels"] = n_channels

            # Soft validation: check LLM's object_type vs ground truth
            gt_object_type = None
            for ds_name, obj_type in DATASET_TO_OBJECT_TYPE.items():
                if ds_name.lower() in state["query"].lower():
                    gt_object_type = obj_type
                    break
            decisions["_gt_object_type"] = gt_object_type
            decisions["_object_type_correct"] = (
                decisions.get("object_type") == gt_object_type if gt_object_type else None
            )
            if gt_object_type and decisions.get("object_type") != gt_object_type:
                if self.verbose:
                    print(f"\n  [NOTE] LLM chose object_type='{decisions['object_type']}' "
                          f"but ground truth is '{gt_object_type}'")

            # Log interaction
            interaction = LLMInteraction(
                step="agentic_observe_r1",
                round=1,
                prompt=prompt,
                response=response_text,
            )
            new_interactions.append(interaction)

            return {
                "messages": [prompt_msg, response],
                "_agentic_phase": "observe",
                "_observe_turns": observe_turns,
                "_observe_decisions": decisions,
                "llm_interactions": new_interactions,
                "reasoning_trace": state["reasoning_trace"] + [
                    f"Observe R1: object_type={decisions.get('object_type', '?')}, "
                    f"color_mode={decisions.get('color_mode', '?')}, "
                    f"filtration={decisions.get('filtration_type', '?')}"
                ],
            }

        elif observe_turns == 2:
            # === OBSERVE R2: PH interpretation (no tools) ===

            decisions = state.get("_observe_decisions") or {}
            ph_summary = self._build_ph_summary(state)

            # Count H0/H1 from ph_stats
            ph_stats = state.get("_ph_stats") or {}
            h0_count = ph_stats.get("H0_count", 0)
            h1_count = ph_stats.get("H1_count", 0)

            prompt = AGENTIC_OBSERVE_R2_PROMPT.format(
                object_type=decisions.get("object_type", "unknown"),
                ph_summary=ph_summary,
                color_mode=decisions.get("color_mode", "grayscale"),
                h0_count=h0_count,
                h1_count=h1_count,
            )

            prompt_msg = HumanMessage(content=prompt)
            self._vprint_prompt(prompt, max_chars=3000)

            # Call LLM without tools — interpretation only
            response = self.model.invoke([prompt_msg])
            response_text = response.content if hasattr(response, "content") else str(response)

            self._vprint_response(response_text)

            # Log interaction
            interaction = LLMInteraction(
                step="agentic_observe_r2",
                round=2,
                prompt=prompt,
                response=response_text,
            )
            new_interactions.append(interaction)

            return {
                "messages": [prompt_msg, response],
                "_agentic_phase": "observe",
                "_observe_turns": observe_turns,
                "_observe_ph_interpretation": response_text,
                "llm_interactions": new_interactions,
                "reasoning_trace": state["reasoning_trace"] + [
                    f"Observe R2: PH interpretation ({len(response_text)} chars)"
                ],
            }

        else:
            # Safety fallback: advance to act
            return {
                "_observe_turns": observe_turns,
                "reasoning_trace": state["reasoning_trace"] + [
                    "Observe: max turns reached, advancing to ACT"
                ],
            }

    def _parse_observe_decisions(self, text: str) -> Dict[str, Any]:
        """Parse the OBSERVE R1 JSON decisions from LLM response."""
        import json as json_mod

        defaults = {
            "object_type": "surface_lesions",
            "object_type_reasoning": "",
            "color_mode": "grayscale",
            "color_mode_reasoning": "",
            "filtration_type": "sublevel",
            "filtration_reasoning": "",
        }

        # Try direct JSON parse
        try:
            parsed = json_mod.loads(text.strip())
            for k, v in defaults.items():
                if k not in parsed:
                    parsed[k] = v
            return parsed
        except json_mod.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        for marker in ["```json", "```"]:
            if marker in text:
                try:
                    start = text.index(marker) + len(marker)
                    end = text.index("```", start)
                    parsed = json_mod.loads(text[start:end].strip())
                    for k, v in defaults.items():
                        if k not in parsed:
                            parsed[k] = v
                    return parsed
                except (json_mod.JSONDecodeError, ValueError):
                    pass

        # Try finding JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                parsed = json_mod.loads(text[brace_start:brace_end + 1])
                for k, v in defaults.items():
                    if k not in parsed:
                        parsed[k] = v
                return parsed
            except json_mod.JSONDecodeError:
                pass

        # Fallback: return defaults
        if self.verbose:
            print(f"\n  [WARN] Could not parse OBSERVE R1 JSON, using defaults")
        return defaults

    def _agentic_execute_observe_tools(self, state: TopoAgentState) -> Dict[str, Any]:
        """Programmatically execute image_loader + compute_ph after OBSERVE R1.

        Uses the LLM's decisions (filtration_type, color_mode) to drive tool
        execution. When color_mode is per_channel, also loads RGB image, splits
        into R/G/B channels, and computes PH on each channel separately for
        downstream per-channel vectorization.
        """
        import numpy as np
        from PIL import Image as PILImage

        decisions = state.get("_observe_decisions") or {}
        filtration_type = decisions.get("filtration_type", "sublevel")
        color_mode = decisions.get("color_mode", "grayscale")

        self._vprint_header("EXECUTE OBSERVE TOOLS")

        new_memory = list(state["short_term_memory"])
        ph_stats = None
        per_channel_persistence = None
        per_channel_images = None

        # 1. Execute image_loader programmatically (grayscale for PH summary)
        has_img = any(name == "image_loader" for name, _ in new_memory)
        if not has_img and "image_loader" in self.tools:
            tool_args = {
                "image_path": state["image_path"],
                "normalize": True,
                "grayscale": True,
            }
            result = self.tools["image_loader"].invoke(tool_args)
            self._vprint_tool_output("image_loader", result, result.get("success", False))
            new_memory.append(("image_loader", result))

        # 2. Execute compute_ph with LLM's chosen filtration (grayscale summary)
        has_ph = any(name == "compute_ph" for name, _ in new_memory)
        if not has_ph and "compute_ph" in self.tools:
            image_array = None
            for name, output in reversed(new_memory):
                if name == "image_loader" and isinstance(output, dict) and output.get("success"):
                    image_array = output.get("image_array")
                    break

            if image_array is not None:
                tool_args = {
                    "image_array": image_array,
                    "filtration_type": filtration_type,
                    "max_dimension": 1,
                }
                if self.verbose:
                    print(f"\n  Computing PH with LLM's chosen filtration: {filtration_type}")
                result = self.tools["compute_ph"].invoke(tool_args)
                self._vprint_tool_output("compute_ph", result, result.get("success", False))
                new_memory.append(("compute_ph", result))

                # Extract PH stats
                if result.get("success"):
                    stats = result.get("statistics", {})
                    persistence_data = result.get("persistence", {})
                    h0_pers = self._extract_persistence_values(persistence_data.get("H0", []))
                    h1_pers = self._extract_persistence_values(persistence_data.get("H1", []))
                    h0_avg = float(np.mean(h0_pers)) if h0_pers else 0.0
                    h1_avg = float(np.mean(h1_pers)) if h1_pers else 0.0
                    ph_stats = {
                        "H0_count": stats.get("H0", {}).get("count", 0),
                        "H1_count": stats.get("H1", {}).get("count", 0),
                        "H0_max_persistence": max(h0_pers) if h0_pers else 0.0,
                        "H1_max_persistence": max(h1_pers) if h1_pers else 0.0,
                        "H0_avg_persistence": h0_avg,
                        "H1_avg_persistence": h1_avg,
                        "filtration": filtration_type,
                    }

        # 3. Per-channel PH: when LLM chose per_channel, compute PH on R/G/B
        if color_mode == "per_channel" and "compute_ph" in self.tools:
            try:
                rgb_img = np.array(PILImage.open(state["image_path"]).convert("RGB"),
                                   dtype=np.float32) / 255.0
                channels = {"R": rgb_img[:, :, 0], "G": rgb_img[:, :, 1], "B": rgb_img[:, :, 2]}
                per_channel_persistence = {}
                per_channel_images = {}

                if self.verbose:
                    print(f"\n  Per-channel PH: computing on R, G, B channels separately")

                for ch_name, ch_array in channels.items():
                    per_channel_images[ch_name] = ch_array.tolist()
                    ch_result = self.tools["compute_ph"].invoke({
                        "image_array": ch_array.tolist(),
                        "filtration_type": filtration_type,
                        "max_dimension": 1,
                    })
                    if ch_result.get("success"):
                        per_channel_persistence[ch_name] = ch_result.get("persistence", {})
                        if self.verbose:
                            ch_stats = ch_result.get("statistics", {})
                            h0c = ch_stats.get("H0", {}).get("count", 0)
                            h1c = ch_stats.get("H1", {}).get("count", 0)
                            print(f"    {ch_name}: H0={h0c}, H1={h1c}")
                    else:
                        per_channel_persistence[ch_name] = {"H0": [], "H1": []}

            except Exception as e:
                if self.verbose:
                    print(f"\n  [WARNING] Per-channel PH failed: {e} — falling back to grayscale")
                per_channel_persistence = None
                per_channel_images = None

        # 4. Compute PH-based signals for ACT phase descriptor selection
        ph_signals = None
        if ph_stats is not None:
            from .skills.rules_data import compute_ph_signals
            ph_signals = compute_ph_signals(
                h0_count=ph_stats["H0_count"],
                h1_count=ph_stats["H1_count"],
                h0_avg_persistence=ph_stats.get("H0_avg_persistence", 0.0),
                h1_avg_persistence=ph_stats.get("H1_avg_persistence", 0.0),
                h0_max_persistence=ph_stats.get("H0_max_persistence", 1.0),
                h1_max_persistence=ph_stats.get("H1_max_persistence", 1.0),
            )
            if self.verbose and ph_signals:
                print(f"\n  PH Signals triggered: {[s['name'] for s in ph_signals]}")

        updates = {
            "short_term_memory": new_memory,
        }
        if ph_stats is not None:
            updates["_ph_stats"] = ph_stats
        if per_channel_persistence is not None:
            updates["_per_channel_persistence"] = per_channel_persistence
        if per_channel_images is not None:
            updates["_per_channel_images"] = per_channel_images
        if ph_signals is not None:
            updates["_ph_signals"] = ph_signals

        return updates

    def _agentic_act(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 2: LLM selects and calls a descriptor tool with FOLLOW/DEVIATE stance.

        Uses the LLM's OBSERVE decisions (object_type, color_mode) — NOT silent
        overrides from skill_registry. Shows benchmark rankings as advisory with
        explicit parameter table. Tracks FOLLOW/DEVIATE stance.
        """
        from .skills.rules_data import (
            build_descriptor_knowledge_text,
            build_parameter_table,
            build_ph_signals_text,
            get_top_recommendation,
            DESCRIPTOR_PROPERTIES,
            REASONING_CHAINS,
            SUPPORTED_TOP_PERFORMERS,
        )

        act_turns = state.get("_act_turns", 0) + 1
        self._vprint_header("ACT", round_num=act_turns)

        # Use LLM's decisions from OBSERVE (NOT silent overrides)
        decisions = state.get("_observe_decisions") or {}
        object_type_hint = decisions.get("object_type", None)
        color_mode = decisions.get("color_mode", "grayscale")

        # Extract PH summary from short_term_memory
        ph_summary = self._build_ph_summary(state)

        # Build descriptor properties text
        descriptor_props_lines = []
        for name, props in DESCRIPTOR_PROPERTIES.items():
            if name in ("ATOL", "persistence_codebook"):
                continue  # Skip training-based descriptors
            descriptor_props_lines.append(
                f"**{name}** ({props['input']}): {props['captures']}\n"
                f"  Best when: {props['best_when']}\n"
                f"  Weakness: {props['weakness']}"
            )
        descriptor_properties = "\n\n".join(descriptor_props_lines)

        # Build reasoning chains text
        chain_lines = []
        for chain_id, chain in REASONING_CHAINS.items():
            chain_lines.append(
                f"**{chain_id}**: {chain['condition']}\n"
                f"  Recommended: {', '.join(chain.get('recommended', []))}"
            )
        reasoning_chains = "\n\n".join(chain_lines)

        # Build benchmark rankings text
        obj_type_for_ranking = object_type_hint or "surface_lesions"
        performers = SUPPORTED_TOP_PERFORMERS.get(obj_type_for_ranking, [])
        ranking_lines = []
        for i, entry in enumerate(performers, 1):
            ranking_lines.append(f"  {i}. {entry['descriptor']} ({entry['accuracy']:.1%})")
        benchmark_rankings = "\n".join(ranking_lines) if ranking_lines else "No rankings available."

        # Build parameter table
        parameter_table = build_parameter_table(obj_type_for_ranking)

        # Get top recommendation
        top_ranked, top_accuracy = get_top_recommendation(obj_type_for_ranking)

        # Build learned rules context (advisory, not authoritative)
        learned_rules = ""
        try:
            learned_rules = self.skill_registry.skill_memory.get_learned_context(
                object_type_hint
            )
            if learned_rules and learned_rules != "No learned rules or reflections yet.":
                learned_rules = f"Learned rules (from past experience):\n{learned_rules}"
            else:
                learned_rules = ""
        except Exception:
            pass

        # Build PH signals text
        ph_signals = state.get("_ph_signals") or []
        ph_signals_text = build_ph_signals_text(ph_signals, top_ranked)

        # Build retry context
        retry_context = ""
        retry_feedback = state.get("_retry_feedback")
        if retry_feedback:
            retry_context = f"## Previous Attempt Feedback\n{retry_feedback}\n"

        # Build prompt with all LLM decisions visible
        prompt = AGENTIC_ACT_PROMPT.format(
            object_type=object_type_hint or "unknown",
            object_type_reasoning=decisions.get("object_type_reasoning", ""),
            color_mode=color_mode,
            filtration_type=decisions.get("filtration_type", "sublevel"),
            ph_summary=ph_summary,
            ph_signals_text=ph_signals_text,
            descriptor_properties=descriptor_properties,
            reasoning_chains=reasoning_chains,
            benchmark_rankings=benchmark_rankings,
            parameter_table=parameter_table,
            top_ranked=top_ranked,
            top_accuracy=top_accuracy,
            learned_rules=learned_rules,
            retry_context=retry_context,
        )

        prompt_msg = HumanMessage(content=prompt)

        # Fresh message list — don't carry observe-phase messages
        messages_to_send = [prompt_msg]

        self._vprint_prompt(prompt, max_chars=3000)

        # Call LLM with descriptor tools
        response = self.act_model.invoke(messages_to_send)

        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            self._vprint_tool_calls([
                {"name": tc.get("name", "?"), "args": tc.get("args", {})}
                for tc in tool_calls
            ])

        # Track FOLLOW/DEVIATE stance
        stance = "FOLLOW"
        response_lower = response_text.lower() if response_text else ""
        if "deviate" in response_lower:
            stance = "DEVIATE"

        # Log LLM interaction
        interaction = LLMInteraction(
            step="agentic_act",
            round=act_turns,
            prompt=prompt,
            response=response_text,
            tool_calls=[{"name": tc.get("name", "?"), "args": tc.get("args", {})}
                        for tc in tool_calls] if tool_calls else None,
        )
        new_interactions = list(state.get("llm_interactions", []))
        new_interactions.append(interaction)

        return {
            "messages": [prompt_msg, response],
            "_agentic_phase": "act",
            "_act_turns": act_turns,
            "_color_mode": color_mode,
            "_benchmark_stance": stance,
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Act turn {act_turns}: {'tool call' if tool_calls else 'reasoning'} "
                f"(stance={stance})"
            ],
        }

    def _agentic_execute_tools(self, state: TopoAgentState) -> Dict[str, Any]:
        """Execute tools called by the LLM in observe or act phase.

        Shared between observe_tools and act_tools nodes. Extracts tool_calls
        from the last AIMessage, executes them with auto-injection, and creates
        ToolMessages for the conversation history.

        If no tool_calls but forced execution is needed (e.g. LLM didn't call
        compute_ph), injects the missing tool call automatically.
        """
        import numpy as np

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []

        # Force compute_ph if LLM didn't call it but image is loaded
        if not tool_calls:
            has_img = any(name == "image_loader" for name, _ in state["short_term_memory"])
            has_ph = any(name == "compute_ph" for name, _ in state["short_term_memory"])
            if has_img and not has_ph:
                # Use LLM's chosen filtration from OBSERVE decisions
                decisions = state.get("_observe_decisions") or {}
                filt = decisions.get("filtration_type", "sublevel")
                retry_fb = state.get("_retry_feedback", "") or ""
                if "superlevel" in retry_fb.lower():
                    filt = "superlevel"
                tool_calls = [{"name": "compute_ph", "args": {
                    "filtration_type": filt, "max_dimension": 1,
                }, "id": "forced_compute_ph"}]
                if self.verbose:
                    print(f"\n  [SAFETY NET] compute_ph (filtration={filt})")

        # Force descriptor if LLM didn't call one but PH is ready
        if not tool_calls:
            has_ph = any(name == "compute_ph" for name, _ in state["short_term_memory"])
            has_descriptor = any(
                name in SUPPORTED_DESCRIPTORS for name, _ in state["short_term_memory"]
            )
            if has_ph and not has_descriptor:
                # Try to extract descriptor from LLM's reasoning text
                last_msg = state["messages"][-1] if state["messages"] else None
                reasoning = ""
                if last_msg and hasattr(last_msg, "content") and last_msg.content:
                    reasoning = last_msg.content

                parsed_desc = None
                for desc_name in SUPPORTED_DESCRIPTORS:
                    if desc_name in reasoning:
                        parsed_desc = desc_name
                        break

                if parsed_desc:
                    desc_name = parsed_desc
                    label = "SAFETY NET — executing LLM's stated choice"
                else:
                    desc_name = self._get_forced_descriptor_for_retry(state)
                    label = "SAFETY NET — top-ranked fallback"

                # Inject optimal parameters from benchmark rules
                decisions = state.get("_observe_decisions") or {}
                obj_type = decisions.get("object_type", "surface_lesions")
                from .skills.rules_data import get_optimal_params
                optimal = get_optimal_params(desc_name, obj_type)
                safety_args = {k: v for k, v in optimal.items()
                               if k not in ("dim", "total_dim", "classifier", "color_mode")}

                tool_calls = [{"name": desc_name, "args": safety_args,
                               "id": f"safety_net_{desc_name}"}]
                if self.verbose:
                    print(f"\n  [{label}] {desc_name} (params: {safety_args})")

        if not tool_calls:
            return {}

        new_memory = list(state["short_term_memory"])
        tool_messages = []
        ph_stats = state.get("_ph_stats")
        feature_quality = state.get("_feature_quality")

        # Sort tool_calls so dependencies run first:
        # image_loader before compute_ph, compute_ph before descriptors
        TOOL_ORDER = {"image_loader": 0, "compute_ph": 1}
        sorted_calls = sorted(tool_calls,
                              key=lambda tc: TOOL_ORDER.get(tc.get("name", ""), 2))

        # Running state for auto-injection so earlier results in this batch
        # are available for later tools (e.g. image_loader → compute_ph)
        running_state = dict(state)
        running_state["short_term_memory"] = new_memory

        # Determine per-channel state
        decisions = state.get("_observe_decisions") or {}
        color_mode = decisions.get("color_mode", "grayscale")
        per_ch_ph = state.get("_per_channel_persistence")
        per_ch_imgs = state.get("_per_channel_images")

        # PH-based vs image-based descriptor sets
        IMAGE_BASED = {"minkowski_functionals", "euler_characteristic_curve",
                       "euler_characteristic_transform", "edge_histogram", "lbp_texture"}

        for tc in sorted_calls:
            tool_name = tc.get("name", "")
            tool_args = (tc.get("args") or {}).copy()
            tc_id = tc.get("id") or tool_name

            # Per-channel descriptor execution: run descriptor 3 times (R, G, B)
            # and concatenate feature vectors for 3x dimension.
            is_descriptor = tool_name in SUPPORTED_DESCRIPTORS
            use_per_channel = (is_descriptor and color_mode == "per_channel"
                               and (per_ch_ph or per_ch_imgs))

            if use_per_channel and tool_name in self.tools:
                try:
                    channel_vectors = []
                    is_image_based = tool_name in IMAGE_BASED

                    for ch_name in ["R", "G", "B"]:
                        ch_args = tool_args.copy()
                        if is_image_based and per_ch_imgs and ch_name in per_ch_imgs:
                            ch_args["image_array"] = per_ch_imgs[ch_name]
                        elif not is_image_based and per_ch_ph and ch_name in per_ch_ph:
                            ch_args["persistence_data"] = per_ch_ph[ch_name]
                        else:
                            # Fallback: use auto-injected grayscale data
                            ch_args = self._auto_inject_args(tool_name, ch_args, running_state)

                        ch_result = self.tools[tool_name].invoke(ch_args)
                        fv = ch_result.get("combined_vector") or ch_result.get("feature_vector")
                        if fv is not None:
                            channel_vectors.append(np.asarray(fv, dtype=np.float64))

                    if channel_vectors:
                        concat_vector = np.concatenate(channel_vectors).tolist()
                        # Build a merged result dict
                        result = {
                            "success": True,
                            "combined_vector": concat_vector,
                            "vector_length": len(concat_vector),
                            "tool_name": tool_name,
                            "color_mode": "per_channel",
                            "per_channel_dims": [len(v) for v in channel_vectors],
                        }
                        # Carry over descriptor-specific params
                        for k in ["n_templates", "template_type", "resolution",
                                   "sigma", "n_bins", "n_layers", "n_thresholds",
                                   "n_directions", "n_heights", "power"]:
                            if k in tool_args:
                                result[k] = tool_args[k]
                    else:
                        result = {"success": False, "error": "No per-channel features extracted"}

                    self._vprint_tool_output(tool_name, result, result.get("success", False))
                    if self.verbose and result.get("success"):
                        dims = result.get("per_channel_dims", [])
                        print(f"  per_channel: R={dims[0] if dims else '?'}D + "
                              f"G={dims[1] if len(dims)>1 else '?'}D + "
                              f"B={dims[2] if len(dims)>2 else '?'}D = {result['vector_length']}D")

                    new_memory.append((tool_name, result))
                    running_state["short_term_memory"] = new_memory

                    # Extract feature quality from concatenated vector
                    if result.get("success"):
                        fv = result.get("combined_vector")
                        if fv is not None:
                            arr = np.asarray(fv, dtype=np.float64)
                            feature_quality = {
                                "dimension": len(arr),
                                "sparsity": float((np.abs(arr) < 1e-10).mean()) * 100,
                                "variance": float(np.var(arr)),
                                "nan_count": int(np.isnan(arr).sum()),
                            }

                    tool_messages.append(ToolMessage(
                        content=str(truncate_output_for_prompt(result, max_chars=500)),
                        tool_call_id=tc_id,
                    ))
                except Exception as e:
                    new_memory.append((tool_name, {"success": False, "error": str(e)}))
                    tool_messages.append(ToolMessage(
                        content=f"Error: {str(e)[:300]}",
                        tool_call_id=tc_id,
                    ))
                    self._vprint_tool_output(tool_name, str(e), False)
                continue  # Skip normal execution path

            # Normal (non-per-channel) execution
            # Auto-inject args using running state
            tool_args = self._auto_inject_args(tool_name, tool_args, running_state)

            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name].invoke(tool_args)
                    self._vprint_tool_output(tool_name, result, result.get("success", False))

                    new_memory.append((tool_name, result))
                    running_state["short_term_memory"] = new_memory

                    # Extract PH stats if compute_ph was called
                    if tool_name == "compute_ph" and result.get("success", False):
                        stats = result.get("statistics", {})
                        persistence_data = result.get("persistence", {})
                        h0_pers = self._extract_persistence_values(persistence_data.get("H0", []))
                        h1_pers = self._extract_persistence_values(persistence_data.get("H1", []))
                        ph_stats = {
                            "H0_count": stats.get("H0", {}).get("count", 0),
                            "H1_count": stats.get("H1", {}).get("count", 0),
                            "H0_max_persistence": max(h0_pers) if h0_pers else 0.0,
                            "H1_max_persistence": max(h1_pers) if h1_pers else 0.0,
                            "filtration": result.get("filtration_type", "sublevel"),
                        }

                    # Extract feature quality if a descriptor was called
                    if tool_name in SUPPORTED_DESCRIPTORS and result.get("success", False):
                        fv = result.get("combined_vector") or result.get("feature_vector")
                        if fv is not None:
                            arr = np.asarray(fv, dtype=np.float64)
                            feature_quality = {
                                "dimension": len(arr),
                                "sparsity": float((np.abs(arr) < 1e-10).mean()) * 100,
                                "variance": float(np.var(arr)),
                                "nan_count": int(np.isnan(arr).sum()),
                            }

                    # Truncate output for ToolMessage
                    tool_messages.append(ToolMessage(
                        content=str(truncate_output_for_prompt(result, max_chars=500)),
                        tool_call_id=tc_id,
                    ))
                except Exception as e:
                    new_memory.append((tool_name, {"success": False, "error": str(e)}))
                    tool_messages.append(ToolMessage(
                        content=f"Error: {str(e)[:300]}",
                        tool_call_id=tc_id,
                    ))
                    self._vprint_tool_output(tool_name, str(e), False)
            else:
                tool_messages.append(ToolMessage(
                    content=f"Tool '{tool_name}' not found.",
                    tool_call_id=tc_id,
                ))

        updates = {
            "messages": tool_messages,
            "short_term_memory": new_memory,
        }
        if ph_stats is not None:
            updates["_ph_stats"] = ph_stats
        if feature_quality is not None:
            updates["_feature_quality"] = feature_quality

        return updates

    def _agentic_reflect(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 3: LLM evaluates feature quality and decides next action.

        No tools bound — pure evaluation. Includes the LLM's earlier decisions
        for trace completeness. Parses JSON decision:
        COMPLETE / RETRY_DESCRIPTOR / RETRY_PH
        """
        import json as json_mod
        import time

        self._vprint_header("REFLECT")

        ph_stats = state.get("_ph_stats") or {}
        feature_quality = state.get("_feature_quality") or {}
        decisions = state.get("_observe_decisions") or {}

        # Find descriptor name from short_term_memory
        descriptor_name = "unknown"
        descriptor_params = {}
        for tool_name, output in reversed(state["short_term_memory"]):
            if tool_name in SUPPORTED_DESCRIPTORS:
                descriptor_name = tool_name
                # Try to extract params from tool args (they're in the output)
                if isinstance(output, dict):
                    for k in ["resolution", "n_bins", "n_layers", "n_templates",
                              "n_thresholds", "sigma", "power", "n_directions",
                              "n_heights", "n_scales", "n_spatial_cells", "max_terms"]:
                        if k in output:
                            descriptor_params[k] = output[k]
                break

        # Image info
        image_info = state["image_path"]
        for tool_name, output in state["short_term_memory"]:
            if tool_name == "image_loader" and isinstance(output, dict):
                image_info = f"{state['image_path']} (shape={output.get('shape', '?')})"
                break

        # Use LLM's object_type from OBSERVE decisions (not silent override)
        object_type_hint = decisions.get("object_type")
        if not object_type_hint:
            # Fallback: use skill_registry for non-agentic paths
            object_type_hint = self.skill_registry.infer_object_type_hint(
                query=state["query"],
                image_path=state["image_path"],
            )
        reference_stats = self._build_reference_stats(
            object_type_hint or "surface_lesions", descriptor_name
        )

        # Build prompt with LLM's decisions in header
        object_type_correct = decisions.get("_object_type_correct")
        if object_type_correct is True:
            ot_correct_str = "YES"
        elif object_type_correct is False:
            ot_correct_str = f"NO (ground truth: {decisions.get('_gt_object_type', '?')})"
        else:
            ot_correct_str = "unknown"

        prompt = AGENTIC_REFLECT_PROMPT.format(
            object_type=decisions.get("object_type", "unknown"),
            object_type_correct=ot_correct_str,
            color_mode=decisions.get("color_mode", "grayscale"),
            filtration_type=decisions.get("filtration_type", "sublevel"),
            benchmark_stance=state.get("_benchmark_stance", "unknown"),
            image_info=image_info,
            h0_count=ph_stats.get("H0_count", 0),
            h1_count=ph_stats.get("H1_count", 0),
            filtration=ph_stats.get("filtration", "sublevel"),
            descriptor_name=descriptor_name,
            dim=feature_quality.get("dimension", 0),
            sparsity=f"{feature_quality.get('sparsity', 0):.1f}",
            variance=f"{feature_quality.get('variance', 0):.6f}",
            nan_count=feature_quality.get("nan_count", 0),
            params=json_mod.dumps(descriptor_params) if descriptor_params else "defaults",
            reference_stats=reference_stats,
        )

        self._vprint_prompt(prompt)

        # Call LLM (no tools)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        self._vprint_response(response_text)

        # Log LLM interaction
        interaction = LLMInteraction(
            step="agentic_reflect",
            round=len(state.get("_reflect_history", [])) + 1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions = list(state.get("llm_interactions", []))
        new_interactions.append(interaction)

        # Parse JSON decision
        decision = self._parse_reflect_decision(response_text)

        # Code-level override: LLM sometimes hallucinates "all checks pass"
        # when dim=0. Force RETRY_DESCRIPTOR if dimension is 0.
        feat_dim = feature_quality.get("dimension", 0)
        if feat_dim == 0 and decision.get("decision", "COMPLETE").upper() == "COMPLETE":
            if self.verbose:
                print(f"\n  [OVERRIDE] LLM said COMPLETE but dim=0 — forcing RETRY_DESCRIPTOR")
            decision["decision"] = "RETRY_DESCRIPTOR"
            decision["quality_ok"] = False
            decision["reasoning"] = (
                f"Code override: feature dimension is 0 (descriptor '{descriptor_name}' "
                f"produced no features). Retrying with a different descriptor."
            )
            decision["retry_suggestion"] = "Try a different descriptor."

        if self.verbose:
            print(f"\n  Decision: {decision.get('decision', 'COMPLETE')}")
            print(f"  Quality OK: {decision.get('quality_ok', True)}")
            print(f"  Reasoning: {decision.get('reasoning', '')[:100]}")

        elapsed = time.time() - getattr(self, "_workflow_start_time", time.time())

        # Build result based on decision
        result_decision = decision.get("decision", "COMPLETE").upper()

        # Build reflect record for history
        reflect_record = {
            "round": len(state.get("_reflect_history", [])) + 1,
            "descriptor": descriptor_name,
            "descriptor_params": descriptor_params,
            "ph_stats": {
                "H0_count": ph_stats.get("H0_count", 0),
                "H1_count": ph_stats.get("H1_count", 0),
                "filtration": ph_stats.get("filtration", "sublevel"),
            },
            "feature_quality": {
                "dimension": feature_quality.get("dimension", 0),
                "sparsity": feature_quality.get("sparsity", 0.0),
                "variance": feature_quality.get("variance", 0.0),
                "nan_count": feature_quality.get("nan_count", 0),
            },
            "quality_ok": decision.get("quality_ok", True),
            "decision": result_decision,
            "reasoning": decision.get("reasoning", ""),
            "retry_suggestion": decision.get("retry_suggestion", ""),
        }

        updates = {
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Reflect: {result_decision} - {decision.get('reasoning', '')[:80]}"
            ],
            "_reflect_history": list(state.get("_reflect_history", [])) + [reflect_record],
        }

        if result_decision in ("RETRY_DESCRIPTOR", "RETRY_PH"):
            # Time limit or max retry: force COMPLETE
            if state.get("_retry_count", 0) >= 1 or self._is_time_exceeded():
                result_decision = "COMPLETE"
            else:
                updates["_retry_count"] = state.get("_retry_count", 0) + 1

                if result_decision == "RETRY_DESCRIPTOR":
                    # Clear descriptor entries from short_term_memory so routing
                    # doesn't skip directly to reflect on retry
                    cleaned_memory = [
                        (name, out) for name, out in state["short_term_memory"]
                        if name not in SUPPORTED_DESCRIPTORS
                    ]
                    updates["short_term_memory"] = cleaned_memory
                    updates["_act_turns"] = 0
                    updates["_feature_quality"] = None

                    # Tell the LLM which descriptor failed and to try something else
                    suggestion = decision.get("retry_suggestion", "")
                    updates["_retry_feedback"] = (
                        f"Previous descriptor '{descriptor_name}' had quality issues. "
                        f"Do NOT use {descriptor_name} again. Choose a DIFFERENT descriptor. "
                        f"{suggestion}"
                    )

                elif result_decision == "RETRY_PH":
                    # Clear ALL tool entries from short_term_memory so observe
                    # re-runs image_loader + compute_ph from scratch
                    updates["short_term_memory"] = []
                    updates["_observe_turns"] = 0
                    updates["_act_turns"] = 0
                    updates["_feature_quality"] = None
                    updates["_ph_stats"] = None

                    # Tell the LLM to try different filtration parameters
                    prev_filt = ph_stats.get("filtration", "sublevel")
                    alt_filt = "superlevel" if prev_filt == "sublevel" else "sublevel"
                    suggestion = decision.get("retry_suggestion", "")
                    updates["_retry_feedback"] = (
                        f"Previous PH used {prev_filt} filtration. "
                        f"Try {alt_filt} filtration instead. {suggestion}"
                    )

        if result_decision == "COMPLETE":
            # Build AgentReport for final_answer
            n_tools = len(state["short_term_memory"])
            n_llm_calls = len(new_interactions)

            agent_report = AgentReport(
                descriptor=descriptor_name,
                object_type=object_type_hint or "unknown",
                reasoning_chain=decision.get("reasoning", ""),
                image_analysis="",
                descriptor_rationale=decision.get("reasoning", ""),
                alternatives_considered="",
                parameters=descriptor_params,
                feature_dimension=feature_quality.get("dimension", 0),
                color_mode=state.get("_color_mode") or "grayscale",
                actual_dimension=feature_quality.get("dimension", 0),
                sparsity_pct=feature_quality.get("sparsity", 0.0),
                variance=feature_quality.get("variance", 0.0),
                nan_count=feature_quality.get("nan_count", 0),
                ph_confirms_object_type=True,
                dimension_correct=True,
                quality_ok=decision.get("quality_ok", True),
                issues=[],
                error_analysis="",
                experience=decision.get("reasoning", ""),
                # Agentic v7: LLM's genuine decisions
                observe_object_type=decisions.get("object_type", ""),
                observe_color_mode=decisions.get("color_mode", ""),
                observe_filtration=decisions.get("filtration_type", ""),
                observe_reasoning=decisions.get("object_type_reasoning", ""),
                object_type_correct=decisions.get("_object_type_correct"),
                benchmark_stance=state.get("_benchmark_stance", ""),
                ph_interpretation=state.get("_observe_ph_interpretation", ""),
                n_llm_calls=n_llm_calls,
                n_tools_executed=n_tools,
                retry_used=state.get("_retry_count", 0) > 0,
                total_time_seconds=round(elapsed, 2),
            )

            report = agent_report.to_dict()
            quality_ok = decision.get("quality_ok", True)
            confidence = 85.0 if quality_ok else 50.0

            updates["final_answer"] = report
            updates["confidence"] = confidence
            updates["evidence"] = [str(item) for item in state["short_term_memory"]]
            updates["task_complete"] = True

        # Store the decision for routing
        updates["_reflect_decision"] = result_decision

        return updates

    def _parse_reflect_decision(self, text: str) -> Dict[str, Any]:
        """Parse the reflect phase JSON decision from LLM response."""
        import json as json_mod

        # Try direct JSON parse
        try:
            return json_mod.loads(text.strip())
        except json_mod.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        for marker in ["```json", "```"]:
            if marker in text:
                try:
                    start = text.index(marker) + len(marker)
                    end = text.index("```", start)
                    return json_mod.loads(text[start:end].strip())
                except (json_mod.JSONDecodeError, ValueError):
                    pass

        # Try finding JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json_mod.loads(text[brace_start:brace_end + 1])
            except json_mod.JSONDecodeError:
                pass

        # Fallback: parse decision from text
        text_upper = text.upper()
        if "RETRY_PH" in text_upper:
            return {"quality_ok": False, "decision": "RETRY_PH",
                    "reasoning": "Parsed from text", "retry_suggestion": "Try different filtration"}
        elif "RETRY_DESCRIPTOR" in text_upper:
            return {"quality_ok": False, "decision": "RETRY_DESCRIPTOR",
                    "reasoning": "Parsed from text", "retry_suggestion": "Try different descriptor"}
        else:
            return {"quality_ok": True, "decision": "COMPLETE",
                    "reasoning": "Defaulted to COMPLETE", "retry_suggestion": ""}

    # --- Routing functions for agentic workflow ---

    def _is_time_exceeded(self) -> bool:
        """Check if the agentic pipeline has exceeded its time limit."""
        import time
        if not hasattr(self, '_workflow_start_time'):
            return False
        elapsed = time.time() - self._workflow_start_time
        return elapsed > self.time_limit_seconds

    def _route_observe(self, state: TopoAgentState) -> Literal["execute_observe_tools", "act"]:
        """Route after observe based on observe_turns.

        Round 1 (JSON decisions): route to execute_observe_tools for programmatic
        image_loader + compute_ph execution.
        Round 2 (PH interpretation): advance to ACT.
        """
        observe_turns = state.get("_observe_turns", 0)

        if observe_turns == 1:
            # After R1 (JSON decisions): programmatically run tools
            return "execute_observe_tools"
        elif observe_turns >= 2:
            # After R2 (PH interpretation): advance to ACT
            return "act"
        # Safety fallback
        return "act"

    def _route_act(self, state: TopoAgentState) -> Literal["tools", "next"]:
        """Route after act: if tool_calls → tools, else → reflect.

        Critical: when LLM produces text reasoning but no tool_calls, we still
        route to 'tools' so the safety net in _agentic_execute_tools() can
        parse the LLM's stated descriptor choice and execute it.
        """
        last_msg = state["messages"][-1] if state["messages"] else None
        if last_msg is None:
            return "next"

        tool_calls = getattr(last_msg, "tool_calls", []) or []
        if tool_calls:
            return "tools"

        # Check if a descriptor has been computed (required to advance)
        has_descriptor = any(
            name in SUPPORTED_DESCRIPTORS for name, _ in state["short_term_memory"]
        )
        if has_descriptor:
            return "next"

        # No tool_calls AND no descriptor yet: route to tools so the safety
        # net can parse the LLM's reasoning and execute the descriptor.
        has_ph = any(
            name == "compute_ph" for name, _ in state["short_term_memory"]
        )
        if has_ph:
            return "tools"

        # Safety: max 3 act turns
        if state.get("_act_turns", 0) >= 3:
            return "next"

        return "next"

    def _route_reflect(self, state: TopoAgentState) -> Literal["complete", "retry_descriptor", "retry_ph"]:
        """Route after reflect based on the LLM's decision."""
        decision = state.get("_reflect_decision", "COMPLETE").upper()

        # Time limit: force COMPLETE regardless of LLM decision
        if self._is_time_exceeded() and decision != "COMPLETE":
            if self.verbose:
                print(f"\n  [TIME LIMIT] Exceeded {self.time_limit_seconds}s — forcing COMPLETE (was {decision})")
            return "complete"

        if decision == "RETRY_DESCRIPTOR":
            return "retry_descriptor"
        elif decision == "RETRY_PH":
            return "retry_ph"
        else:
            return "complete"

    # _route_after_observe_tools removed in v7 — execute_observe_tools
    # edges directly back to observe (for R2 PH interpretation).

    def _route_after_act_tools(self, state: TopoAgentState) -> Literal["act", "reflect"]:
        """Route after act_tools: go to reflect if descriptor was computed.

        If a descriptor tool ran successfully, proceed directly to reflect
        for quality evaluation. Otherwise loop back to act for retry.
        """
        has_descriptor = any(
            name in SUPPORTED_DESCRIPTORS for name, _ in state["short_term_memory"]
        )
        if has_descriptor:
            return "reflect"
        return "act"

    # =========================================================================
    # v8 Agentic Pipeline: 5-Phase (PERCEIVE → ANALYZE → PLAN → EXTRACT → REFLECT)
    # =========================================================================

    def _build_agentic_v8_workflow(self) -> Any:
        """Build the v8.1 7-phase LangGraph workflow.

        Architecture (v8.1):
            PERCEIVE_BASE (deterministic: image_loader + image_analyzer)
            → PERCEIVE_DECIDE (LLM #1: filtration type, denoising, PH parameters)
            → PERCEIVE_EXECUTE (deterministic: [noise_filter?] + compute_ph + topo_features + betti_ratios)
            → ANALYZE (LLM #2: synthesize perception, classify object type + color mode)
            → PLAN (LLM #3: select descriptor with LTM consultation)
            → EXTRACT (deterministic: run descriptor tool)
            → REFLECT (LLM #4: evaluate quality with descriptor-specific ranges, learn, decide complete/retry)
            → END or retry back to PLAN (LLM #5-6 if retry)

        LLM calls: 4 minimum (no retry), 6 maximum (1 retry). Variable per image.

        Key changes from v8.0:
        - PERCEIVE split into 3 sub-phases (BASE → DECIDE → EXECUTE)
        - PERCEIVE_DECIDE: LLM chooses filtration_type + denoising (was hardcoded)
        - noise_filter actually called when LLM decides SNR is too low
        - REFLECT uses descriptor-specific expected quality ranges

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(TopoAgentState)

        # Phase 1a: PERCEIVE_BASE (deterministic: image_loader + image_analyzer)
        workflow.add_node("v8_perceive_base", self._v8_perceive_base)

        # Phase 1b: PERCEIVE_DECIDE (LLM #1: filtration + denoising decisions)
        workflow.add_node("v8_perceive_decide", self._v8_perceive_decide)

        # Phase 1c: PERCEIVE_EXECUTE (deterministic: uses LLM decisions)
        workflow.add_node("v8_perceive_execute", self._v8_perceive_execute)

        # Phase 2: ANALYZE (LLM #2 or deterministic bypass for C4)
        if self.ablate_analyze:
            workflow.add_node("v8_analyze", self._v8_analyze_bypass)
        else:
            workflow.add_node("v8_analyze", self._v8_analyze)

        # Phase 3: PLAN (LLM #3: descriptor selection)
        workflow.add_node("v8_plan", self._v8_plan)

        # Phase 4: EXTRACT (deterministic: execute descriptor)
        if self.ablate_reflect:
            # C3: _v8_extract also builds final report (no REFLECT phase)
            workflow.add_node("v8_extract", self._v8_extract_with_report)
        else:
            workflow.add_node("v8_extract", self._v8_extract)

        # 7-phase pipeline
        workflow.set_entry_point("v8_perceive_base")
        workflow.add_edge("v8_perceive_base", "v8_perceive_decide")   # LLM #1
        workflow.add_edge("v8_perceive_decide", "v8_perceive_execute")
        workflow.add_edge("v8_perceive_execute", "v8_analyze")         # LLM #2 (or bypass)
        workflow.add_edge("v8_analyze", "v8_plan")                     # LLM #3
        workflow.add_edge("v8_plan", "v8_extract")

        if self.ablate_reflect:
            # C3: EXTRACT → END directly (no REFLECT)
            workflow.add_edge("v8_extract", END)
        else:
            # Phase 5: REFLECT (LLM #4: evaluation + learning)
            workflow.add_node("v8_reflect", self._v8_reflect)
            workflow.add_edge("v8_extract", "v8_reflect")              # LLM #4
            # REFLECT → END, retry via PLAN (LLM), or retry_forced (skip PLAN, use backup)
            workflow.add_conditional_edges("v8_reflect", self._v8_route_reflect,
                {"complete": END, "retry": "v8_plan", "retry_forced": "v8_extract"})

        return workflow.compile()

    def _v8_perceive_base(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 1a: PERCEIVE_BASE — load image and analyze properties.

        Deterministic: always runs image_loader + image_analyzer.
        These two always run regardless of image content.
        """
        import time

        if not hasattr(self, '_workflow_start_time'):
            self._workflow_start_time = time.time()

        self._vprint_header("V8 PERCEIVE_BASE (deterministic)")

        new_memory = list(state["short_term_memory"])
        tools_used = []
        image_analysis = None
        reasoning_trace = list(state["reasoning_trace"])

        # Helper to execute a tool with auto-injection
        def _exec_tool(name, extra_args=None):
            if name not in self.tools:
                if self.verbose:
                    print(f"  [PERCEIVE_BASE] Tool '{name}' not available, skipping")
                return None
            args = (extra_args or {}).copy()
            args = self._auto_inject_args(name, args,
                                          {**state, "short_term_memory": new_memory})
            try:
                result = self.tools[name].invoke(args)
                self._vprint_tool_output(name, result, result.get("success", False))
                new_memory.append((name, result))
                tools_used.append(name)
                return result
            except Exception as e:
                if self.verbose:
                    print(f"  [PERCEIVE_BASE] {name} failed: {e}")
                new_memory.append((name, {"success": False, "error": str(e)}))
                tools_used.append(name)
                return None

        # Detect original image channel count from file (before grayscale conversion)
        import numpy as np
        from PIL import Image as PILImage
        original_n_channels = 1
        try:
            pil_img = PILImage.open(state["image_path"])
            if pil_img.mode == "RGB":
                original_n_channels = 3
            elif pil_img.mode == "RGBA":
                original_n_channels = 3  # treat as RGB
            original_shape = list(np.array(pil_img).shape)
        except Exception:
            original_shape = []

        # 1. Load image (converts to grayscale for PH computation)
        _exec_tool("image_loader", {"image_path": state["image_path"]})

        # 2. Image analyzer (quantitative stats: SNR, contrast, edges, etc.)
        img_analysis_result = _exec_tool("image_analyzer",
                                         {"image_path": state["image_path"]})
        if img_analysis_result and img_analysis_result.get("success", False):
            image_analysis = img_analysis_result
            # Inject original channel info into image_analysis so ANALYZE sees it
            if "image_statistics" not in image_analysis:
                image_analysis["image_statistics"] = {}
            image_analysis["image_statistics"]["original_n_channels"] = original_n_channels
            image_analysis["image_statistics"]["original_shape"] = original_shape

        reasoning_trace.append(
            f"V8 Perceive Base: executed {len(tools_used)} tools: {tools_used}"
        )

        updates = {
            "short_term_memory": new_memory,
            "_v8_mode": True,
            "_v8_perceive_turns": len(tools_used),
            "reasoning_trace": reasoning_trace,
        }
        if image_analysis is not None:
            updates["_v8_image_analysis"] = image_analysis

        return updates

    def _v8_perceive_decide(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 1b: PERCEIVE_DECIDE — LLM decides filtration + denoising.

        LLM call #1: sees image_analyzer output and makes 3 decisions:
        1. filtration_type: sublevel or superlevel
        2. apply_denoising: whether to run noise_filter
        3. max_dimension: 1 (default) or 2

        Text-in/JSON-out call (no tool-calling) to avoid GPT-4o reliability issues.
        """
        import json as json_mod

        self._vprint_header("V8 PERCEIVE_DECIDE (LLM #1)")
        new_interactions = list(state.get("llm_interactions", []))

        # Extract image analysis for prompt
        image_analysis = state.get("_v8_image_analysis") or {}
        if image_analysis:
            metrics = image_analysis.get("image_statistics", {})
            recs = image_analysis.get("recommendations", {})
            img_summary = (
                f"SNR={metrics.get('snr_estimate', 'N/A')}, "
                f"contrast={metrics.get('contrast', 'N/A')}, "
                f"edge_density={metrics.get('edge_density', 'N/A')}, "
                f"bright_ratio={metrics.get('bright_ratio', 'N/A')}, "
                f"dark_ratio={metrics.get('dark_ratio', 'N/A')}, "
                f"variance={metrics.get('variance', 'N/A')}"
            )
            recommended_filtration = recs.get("filtration_type", "sublevel")
            filtration_reason = recs.get("filtration_reason", "default")
            recommended_noise = recs.get("noise_filter", "none")
            noise_reason = recs.get("noise_reason", "not recommended")
        else:
            img_summary = "image_analyzer was not called."
            recommended_filtration = "sublevel"
            filtration_reason = "default (no image analysis available)"
            recommended_noise = "none"
            noise_reason = "no image analysis available"

        dataset_context = state.get("query", "")[:300]

        prompt = V8_PERCEIVE_DECIDE_PROMPT.format(
            image_analysis_summary=img_summary,
            recommended_filtration=recommended_filtration,
            filtration_reason=filtration_reason,
            recommended_noise_filter=recommended_noise,
            noise_reason=noise_reason,
            dataset_context=dataset_context,
        )

        self._vprint_prompt(prompt, max_chars=2000)

        # Call LLM (no tools — text-in/JSON-out)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse JSON
        decisions = self._parse_json_response(response_text, defaults={
            "filtration_type": recommended_filtration,
            "filtration_reasoning": f"Following image analyzer: {filtration_reason}",
            "apply_denoising": False,
            "denoising_method": None,
            "denoising_reasoning": "Default: no denoising",
            "max_dimension": 1,
        })

        # Validate filtration_type
        if decisions.get("filtration_type") not in ("sublevel", "superlevel"):
            decisions["filtration_type"] = "sublevel"

        # Log interaction
        interaction = LLMInteraction(
            step="v8_perceive_decide",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        if self.verbose:
            print(f"\n  Filtration: {decisions['filtration_type']} ({decisions.get('filtration_reasoning', '')[:80]})")
            print(f"  Denoising: {decisions.get('apply_denoising', False)} ({decisions.get('denoising_reasoning', '')[:80]})")
            print(f"  Max dimension: {decisions.get('max_dimension', 1)}")

        return {
            "_perceive_decisions": decisions,
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V8 Perceive Decide: filtration={decisions['filtration_type']}, "
                f"denoise={decisions.get('apply_denoising', False)}, "
                f"max_dim={decisions.get('max_dimension', 1)}"
            ],
        }

    def _v8_perceive_execute(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 1c: PERCEIVE_EXECUTE — execute PH tools using LLM decisions.

        Deterministic execution using PERCEIVE_DECIDE's output:
        1. Optional noise_filter (if LLM decided apply_denoising=True)
        2. compute_ph with LLM-chosen filtration_type
        3. topological_features (always)
        4. betti_ratios (always)
        """
        import numpy as np

        self._vprint_header("V8 PERCEIVE_EXECUTE (deterministic, uses LLM decisions)")

        decisions = state.get("_perceive_decisions") or {}
        new_memory = list(state["short_term_memory"])
        tools_used = []
        ph_stats = None
        reasoning_trace = list(state["reasoning_trace"])

        # Helper to execute a tool with auto-injection
        def _exec_tool(name, extra_args=None):
            if name not in self.tools:
                if self.verbose:
                    print(f"  [PERCEIVE_EXECUTE] Tool '{name}' not available, skipping")
                return None
            args = (extra_args or {}).copy()
            args = self._auto_inject_args(name, args,
                                          {**state, "short_term_memory": new_memory})
            try:
                result = self.tools[name].invoke(args)
                self._vprint_tool_output(name, result, result.get("success", False))
                new_memory.append((name, result))
                tools_used.append(name)
                return result
            except Exception as e:
                if self.verbose:
                    print(f"  [PERCEIVE_EXECUTE] {name} failed: {e}")
                new_memory.append((name, {"success": False, "error": str(e)}))
                tools_used.append(name)
                return None

        # 1. Optional denoising (based on LLM decision)
        if decisions.get("apply_denoising"):
            method = decisions.get("denoising_method", "median")
            if method not in ("median", "gaussian", "bilateral"):
                method = "median"
            _exec_tool("noise_filter", {"method": method})
            if self.verbose:
                print(f"  [PERCEIVE_EXECUTE] Applied {method} denoising (LLM decision)")

        # 2. Compute PH with LLM-chosen filtration
        filtration_type = decisions.get("filtration_type", "sublevel")
        max_dimension = decisions.get("max_dimension", 1)
        ph_result = _exec_tool("compute_ph", {
            "filtration_type": filtration_type,
            "max_dimension": max_dimension,
        })

        if ph_result and ph_result.get("success", False):
            stats = ph_result.get("statistics", {})
            persistence_data = ph_result.get("persistence", {})
            h0_pers = self._extract_persistence_values(persistence_data.get("H0", []))
            h1_pers = self._extract_persistence_values(persistence_data.get("H1", []))
            h0_avg = float(np.mean(h0_pers)) if h0_pers else 0.0
            h1_avg = float(np.mean(h1_pers)) if h1_pers else 0.0
            ph_stats = {
                "H0_count": stats.get("H0", {}).get("count", 0),
                "H1_count": stats.get("H1", {}).get("count", 0),
                "H0_max_persistence": max(h0_pers) if h0_pers else 0.0,
                "H1_max_persistence": max(h1_pers) if h1_pers else 0.0,
                "H0_avg_persistence": h0_avg,
                "H1_avg_persistence": h1_avg,
                "filtration": ph_result.get("filtration_type", filtration_type),
            }

        # 3. Topological features (62 stats)
        _exec_tool("topological_features")

        # 4. Betti ratios
        _exec_tool("betti_ratios")

        reasoning_trace.append(
            f"V8 Perceive Execute: {len(tools_used)} tools "
            f"(filtration={filtration_type}, denoise={decisions.get('apply_denoising', False)}): {tools_used}"
        )

        updates = {
            "short_term_memory": new_memory,
            "_v8_perceive_turns": state.get("_v8_perceive_turns", 0) + len(tools_used),
            "reasoning_trace": reasoning_trace,
        }
        if ph_stats is not None:
            updates["_ph_stats"] = ph_stats

        return updates

    def _v8_perceive_tools(self, state: TopoAgentState) -> Dict[str, Any]:
        """Execute tool called by LLM in PERCEIVE phase.

        Executes one tool at a time, with auto-injection of dependencies.
        Stores results in short-term memory and extracts PH stats / image analysis.

        Includes forced progression: if the LLM keeps calling a tool that's
        already been executed, replace it with the next required tool to
        ensure the pipeline advances (image_loader → compute_ph minimum).
        """
        import numpy as np

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []

        if not tool_calls:
            return {}

        new_memory = list(state["short_term_memory"])
        tool_messages = []
        ph_stats = state.get("_ph_stats")
        image_analysis = state.get("_v8_image_analysis")

        # Track which tools have already been called (for forced progression)
        tools_already_called = {name for name, _ in state.get("short_term_memory", [])}

        # Forced progression pipeline: image_loader → compute_ph → done
        # If LLM keeps calling the same tool, replace with the next required one
        _PERCEIVE_PIPELINE = ["image_loader", "compute_ph", "topological_features", "betti_ratios"]

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = (tc.get("args") or {}).copy()
            tc_id = tc.get("id") or tool_name

            # Forced progression: if this tool was already called, advance to
            # the next uncalled tool in the pipeline
            if tool_name in tools_already_called:
                original_name = tool_name
                replaced = False
                for next_tool in _PERCEIVE_PIPELINE:
                    if next_tool not in tools_already_called and next_tool in self.tools:
                        tool_name = next_tool
                        tool_args = {}  # reset args; auto-inject will fill them
                        replaced = True
                        break
                if replaced:
                    if self.verbose:
                        print(f"  [FORCED PROGRESSION] {original_name} already called → forcing {tool_name}")
                elif self.verbose:
                    print(f"  [FORCED PROGRESSION] All pipeline tools done, skipping duplicate {tool_name}")

            # Auto-inject args
            tool_args = self._auto_inject_args(tool_name, tool_args,
                                                {**state, "short_term_memory": new_memory})

            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name].invoke(tool_args)
                    self._vprint_tool_output(tool_name, result, result.get("success", False))

                    new_memory.append((tool_name, result))
                    tools_already_called.add(tool_name)

                    # Extract PH stats
                    if tool_name == "compute_ph" and result.get("success", False):
                        stats = result.get("statistics", {})
                        persistence_data = result.get("persistence", {})
                        h0_pers = self._extract_persistence_values(persistence_data.get("H0", []))
                        h1_pers = self._extract_persistence_values(persistence_data.get("H1", []))
                        h0_avg = float(np.mean(h0_pers)) if h0_pers else 0.0
                        h1_avg = float(np.mean(h1_pers)) if h1_pers else 0.0
                        ph_stats = {
                            "H0_count": stats.get("H0", {}).get("count", 0),
                            "H1_count": stats.get("H1", {}).get("count", 0),
                            "H0_max_persistence": max(h0_pers) if h0_pers else 0.0,
                            "H1_max_persistence": max(h1_pers) if h1_pers else 0.0,
                            "H0_avg_persistence": h0_avg,
                            "H1_avg_persistence": h1_avg,
                            "filtration": result.get("filtration_type", "sublevel"),
                        }

                    # Capture image_analyzer output
                    if tool_name == "image_analyzer" and result.get("success", False):
                        image_analysis = result

                    tool_messages.append(ToolMessage(
                        content=str(truncate_output_for_prompt(result, max_chars=500)),
                        tool_call_id=tc_id,
                    ))
                except Exception as e:
                    new_memory.append((tool_name, {"success": False, "error": str(e)}))
                    tool_messages.append(ToolMessage(
                        content=f"Error: {str(e)[:300]}",
                        tool_call_id=tc_id,
                    ))
                    self._vprint_tool_output(tool_name, str(e), False)
            else:
                tool_messages.append(ToolMessage(
                    content=f"Tool '{tool_name}' not found.",
                    tool_call_id=tc_id,
                ))

        updates = {
            "messages": tool_messages,
            "short_term_memory": new_memory,
        }
        if ph_stats is not None:
            updates["_ph_stats"] = ph_stats
        if image_analysis is not None:
            updates["_v8_image_analysis"] = image_analysis

        return updates

    def _v8_analyze_bypass(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 2 (C4 ablation): Deterministic ANALYZE bypass — no LLM call.

        Uses rule-based lookup for object_type and color_mode:
        - object_type from DATASET_TO_OBJECT_TYPE[dataset_name]
        - color_mode from image channels (3 → per_channel, else grayscale)
        - PH signals still computed (quantitative, not LLM-generated)
        - No descriptor_intuition, image_characteristics, or ph_interpretation
        """
        from .skills.rules_data import (
            DATASET_TO_OBJECT_TYPE, DATASET_COLOR_MODE,
            compute_ph_signals, build_ph_signals_text,
        )

        self._vprint_header("V8 ANALYZE (BYPASS — C4 ablation)")

        # Determine object_type from dataset name in query/image_path
        search_text = (state.get("query", "") + " " + state.get("image_path", "")).lower()
        object_type = "surface_lesions"  # fallback
        gt_object_type = None
        for ds_name, obj_type in DATASET_TO_OBJECT_TYPE.items():
            if ds_name.lower() in search_text:
                object_type = obj_type
                gt_object_type = obj_type
                break

        # Determine color_mode from image channels
        image_analysis = state.get("_v8_image_analysis") or {}
        metrics = image_analysis.get("image_statistics", {})
        original_n_channels = metrics.get("original_n_channels", 1)
        color_mode = "per_channel" if original_n_channels == 3 else "grayscale"

        # Also try DATASET_COLOR_MODE lookup as backup
        for ds_name, ds_color in DATASET_COLOR_MODE.items():
            if ds_name.lower() in search_text:
                color_mode = ds_color
                break

        # Compute PH signals (quantitative, not LLM-generated)
        ph_stats = state.get("_ph_stats") or {}
        ph_signals = []
        if ph_stats:
            ph_signals = compute_ph_signals(
                h0_count=ph_stats.get("H0_count", 0),
                h1_count=ph_stats.get("H1_count", 0),
                h0_avg_persistence=ph_stats.get("H0_avg_persistence", 0.0),
                h1_avg_persistence=ph_stats.get("H1_avg_persistence", 0.0),
                h0_max_persistence=ph_stats.get("H0_max_persistence", 1.0),
                h1_max_persistence=ph_stats.get("H1_max_persistence", 1.0),
            )

        # Extract perceive decisions
        perceive_decisions = state.get("_perceive_decisions") or {}
        perceive_filtration = perceive_decisions.get("filtration_type", "sublevel")

        analysis = {
            "object_type": object_type,
            "object_type_reasoning": "Determined from dataset name (C4 ablation bypass)",
            "color_mode": color_mode,
            "color_mode_reasoning": f"Determined from image channels={original_n_channels} (C4 ablation bypass)",
            "image_characteristics": "",
            "ph_interpretation": "",
            "descriptor_intuition": "",
            "_gt_object_type": gt_object_type,
            "_object_type_correct": True if gt_object_type else None,
        }

        if self.verbose:
            print(f"  object_type: {object_type} (from dataset lookup)")
            print(f"  color_mode: {color_mode}")
            print(f"  PH signals: {[s.get('name', '') for s in ph_signals]}")

        return {
            "_v8_analysis_context": analysis,
            "_ph_signals": ph_signals,
            "_observe_decisions": {
                "object_type": object_type,
                "color_mode": color_mode,
                "color_mode_reasoning": analysis["color_mode_reasoning"],
                "filtration_type": perceive_filtration,
                "filtration_reasoning": "from perceive_decide",
                "object_type_reasoning": analysis["object_type_reasoning"],
                "_gt_object_type": gt_object_type,
                "_object_type_correct": True if gt_object_type else None,
            },
            "llm_interactions": list(state.get("llm_interactions", [])),
            "reasoning_trace": state["reasoning_trace"] + [
                f"V8 Analyze (bypass): object_type={object_type}, color_mode={color_mode}"
            ],
        }

    def _v8_analyze(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 2: ANALYZE — LLM synthesizes all perception outputs.

        Single LLM call that reads all tool outputs and produces a structured
        analysis with object_type, color_mode, and descriptor intuition.
        """
        import json as json_mod
        from .skills.rules_data import compute_ph_signals, build_ph_signals_text

        self._vprint_header("V8 ANALYZE")
        new_interactions = list(state.get("llm_interactions", []))

        # Build image analysis summary
        image_analysis = state.get("_v8_image_analysis") or {}
        if image_analysis:
            metrics = image_analysis.get("image_statistics", {})
            recs = image_analysis.get("recommendations", {})
            # Use ORIGINAL image shape/channels (before grayscale conversion)
            original_n_channels = metrics.get("original_n_channels", 1)
            original_shape = metrics.get("original_shape", metrics.get("shape", []))
            if original_n_channels == 3:
                channel_desc = "RGB 3-channel — use per_channel mode"
            else:
                channel_desc = "grayscale 1-channel — use grayscale mode"
            img_summary = (
                f"Original image: {original_shape} ({channel_desc})\n"
                f"SNR={metrics.get('snr_estimate', 'N/A')}, "
                f"contrast={metrics.get('contrast', 'N/A')}, "
                f"edge_density={metrics.get('edge_density', 'N/A')}, "
                f"bright_ratio={metrics.get('bright_ratio', 'N/A')}, "
                f"dark_ratio={metrics.get('dark_ratio', 'N/A')}, "
                f"variance={metrics.get('variance', 'N/A')}"
            )
        else:
            img_summary = "image_analyzer was not called."

        # Build PH summary
        ph_summary = self._build_ph_summary(state)

        # Build topological_features summary
        topo_summary = "topological_features was not called."
        for name, output in state["short_term_memory"]:
            if name == "topological_features" and isinstance(output, dict) and output.get("success"):
                features_by_dim = output.get("features_by_dimension", {})
                # Summarize key stats per dimension
                topo_lines = []
                for dim_name in ["H0", "H1"]:
                    dim_feats = features_by_dim.get(dim_name, {})
                    for k, v in sorted(dim_feats.items()):
                        if isinstance(v, (int, float)):
                            topo_lines.append(f"{dim_name}_{k}={v:.4f}" if isinstance(v, float) else f"{dim_name}_{k}={v}")
                topo_summary = ", ".join(topo_lines[:20])
                if len(topo_lines) > 20:
                    topo_summary += f" ...({len(topo_lines)} total features)"
                break

        # Compute PH signals
        ph_stats = state.get("_ph_stats") or {}
        ph_signals = []
        if ph_stats:
            ph_signals = compute_ph_signals(
                h0_count=ph_stats.get("H0_count", 0),
                h1_count=ph_stats.get("H1_count", 0),
                h0_avg_persistence=ph_stats.get("H0_avg_persistence", 0.0),
                h1_avg_persistence=ph_stats.get("H1_avg_persistence", 0.0),
                h0_max_persistence=ph_stats.get("H0_max_persistence", 1.0),
                h1_max_persistence=ph_stats.get("H1_max_persistence", 1.0),
            )
        ph_signals_text = build_ph_signals_text(ph_signals, "persistence_statistics")

        # Tools used summary
        tools_used = [name for name, _ in state["short_term_memory"]]
        tools_summary = f"Used {len(tools_used)} tools: {', '.join(tools_used)}"

        # Extract dataset context from user query
        query = state.get("query", "")
        # Truncate to first 300 chars — enough for dataset name + domain description
        dataset_context = query[:300] if query else "No dataset context available."

        # Extract perceive decisions for prompt
        perceive_decisions = state.get("_perceive_decisions") or {}
        perceive_filtration = perceive_decisions.get("filtration_type", "sublevel")
        perceive_filtration_reasoning = perceive_decisions.get("filtration_reasoning", "default")
        perceive_denoising = "yes" if perceive_decisions.get("apply_denoising") else "no"
        perceive_denoising_reasoning = perceive_decisions.get("denoising_reasoning", "default")

        prompt = V8_ANALYZE_PROMPT.format(
            dataset_context=dataset_context,
            perceive_filtration=perceive_filtration,
            perceive_filtration_reasoning=perceive_filtration_reasoning,
            perceive_denoising=perceive_denoising,
            perceive_denoising_reasoning=perceive_denoising_reasoning,
            image_analysis_summary=img_summary,
            ph_summary=ph_summary,
            topo_features_summary=topo_summary,
            ph_signals_text=ph_signals_text,
            tools_used_summary=tools_summary,
        )

        self._vprint_prompt(prompt, max_chars=3000)

        # Call LLM (no tools)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse JSON
        analysis = self._parse_json_response(response_text, defaults={
            "object_type": "surface_lesions",
            "object_type_reasoning": "",
            "color_mode": "grayscale",
            "color_mode_reasoning": "",
            "image_characteristics": "",
            "ph_interpretation": "",
            "descriptor_intuition": "",
            "tools_used_summary": tools_summary,
        })

        # Soft validation: check LLM's object_type vs ground truth
        from .skills.rules_data import DATASET_TO_OBJECT_TYPE
        gt_object_type = None
        # Check both query and image_path for dataset name
        search_text = (state["query"] + " " + state.get("image_path", "")).lower()
        for ds_name, obj_type in DATASET_TO_OBJECT_TYPE.items():
            if ds_name.lower() in search_text:
                gt_object_type = obj_type
                break
        analysis["_gt_object_type"] = gt_object_type
        analysis["_object_type_correct"] = (
            analysis.get("object_type") == gt_object_type if gt_object_type else None
        )

        # Log interaction
        interaction = LLMInteraction(
            step="v8_analyze",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        return {
            "_v8_analysis_context": analysis,
            "_ph_signals": ph_signals,
            # Also set v7-compat fields
            "_observe_decisions": {
                "object_type": analysis.get("object_type"),
                "color_mode": analysis.get("color_mode"),
                "color_mode_reasoning": analysis.get("color_mode_reasoning", ""),
                "filtration_type": perceive_filtration,
                "filtration_reasoning": perceive_filtration_reasoning,
                "object_type_reasoning": analysis.get("image_characteristics", ""),
                "_gt_object_type": gt_object_type,
                "_object_type_correct": analysis.get("_object_type_correct"),
            },
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V8 Analyze: object_type={analysis.get('object_type', '?')}, "
                f"color_mode={analysis.get('color_mode', '?')}, "
                f"intuition={analysis.get('descriptor_intuition', '?')[:60]}"
            ],
        }

    def _v8_plan(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 3: PLAN — LLM selects descriptor with LTM consultation.

        Presents the LLM with:
        - Analysis context from ANALYZE phase
        - PH signals
        - Long-term memory (past experiences)
        - Benchmark rankings + parameters
        """
        import json as json_mod
        from .skills.rules_data import (
            build_descriptor_knowledge_text,
            build_parameter_table,
            build_ph_signals_text,
            get_top_recommendation,
            DESCRIPTOR_PROPERTIES,
            REASONING_CHAINS,
            SUPPORTED_TOP_PERFORMERS,
        )

        self._vprint_header("V8 PLAN")
        new_interactions = list(state.get("llm_interactions", []))

        analysis = state.get("_v8_analysis_context") or {}
        object_type = analysis.get("object_type", "surface_lesions")
        color_mode = analysis.get("color_mode", "grayscale")

        # C1 ablation: remove all skills/benchmark knowledge from prompt
        if self.ablate_skills:
            descriptor_properties = ""
            reasoning_chains = ""
            benchmark_rankings = "No benchmark data available."
            parameter_table = ""
            top_ranked = "persistence_statistics"
            top_accuracy = 0.0  # float required by prompt format {top_accuracy:.1%}
        else:
            # Build descriptor properties text (all 15)
            descriptor_props_lines = []
            for name, props in DESCRIPTOR_PROPERTIES.items():
                descriptor_props_lines.append(
                    f"**{name}** ({props['input']}): {props['captures']}\n"
                    f"  Best when: {props['best_when']}\n"
                    f"  Weakness: {props['weakness']}"
                )
            descriptor_properties = "\n\n".join(descriptor_props_lines)

            # Reasoning chains
            chain_lines = []
            for chain_id, chain in REASONING_CHAINS.items():
                chain_lines.append(
                    f"**{chain_id}**: {chain['condition']}\n"
                    f"  Recommended: {', '.join(chain.get('recommended', []))}"
                )
            reasoning_chains = "\n\n".join(chain_lines)

            # Benchmark rankings (use full TOP_PERFORMERS including ATOL/codebook)
            performers = TOP_PERFORMERS.get(object_type, [])
            ranking_lines = []
            for i, entry in enumerate(performers, 1):
                ranking_lines.append(f"  {i}. {entry['descriptor']} ({entry['accuracy']:.1%})")
            benchmark_rankings = "\n".join(ranking_lines) if ranking_lines else "No rankings available."

            # Parameter table
            parameter_table = build_parameter_table(object_type)

            # Top recommendation
            top_ranked, top_accuracy = get_top_recommendation(object_type)

        # PH signals text
        ph_signals = state.get("_ph_signals") or []
        ph_signals_text = build_ph_signals_text(ph_signals, top_ranked)

        # Long-term memory consultation (skip for C2 ablation)
        ltm_text = "No past experiences yet — this is your first image in this session."
        if not self.ablate_memory and self.long_term_memory is not None:
            all_entries = self.long_term_memory.get_all()
            n_total = len(all_entries)
            query = f"{object_type} {analysis.get('descriptor_intuition', '')} {' '.join(s.get('name', '') for s in ph_signals)}"
            relevant = self.long_term_memory.search_experiences(query, n=5)
            if relevant:
                ltm_lines = [f"You have {n_total} total experience(s). Most relevant:\n"]
                for i, entry in enumerate(relevant, 1):
                    ctx = entry.context or ""
                    ltm_lines.append(
                        f"  **Experience {i}** [{ctx}]:\n"
                        f"    Descriptor tried: {entry.suggestion}\n"
                        f"    Lesson: {entry.experience[:300]}"
                    )
                ltm_text = "\n".join(ltm_lines)
            elif n_total > 0:
                ltm_text = f"You have {n_total} experience(s) but none matched this query."

        # Learned rules (skip for C1 ablation)
        learned_rules = ""
        if not self.ablate_skills:
            try:
                learned_rules = self.skill_registry.skill_memory.get_learned_context(object_type)
                if learned_rules and learned_rules != "No learned rules or reflections yet.":
                    learned_rules = f"Learned rules (from past experience):\n{learned_rules}"
                else:
                    learned_rules = ""
            except Exception:
                pass

        # Retry context
        retry_context = ""
        retry_feedback = state.get("_retry_feedback")
        if retry_feedback:
            retry_context = f"## Previous Attempt Feedback\n{retry_feedback}\n"

        # C2 ablation: remove STM summaries from prompt
        if self.ablate_memory:
            plan_image_chars = "Not available."
            plan_ph_interp = "Not available."
        else:
            plan_image_chars = analysis.get("image_characteristics", "")
            plan_ph_interp = analysis.get("ph_interpretation", "")

        prompt = V8_PLAN_PROMPT.format(
            object_type=object_type,
            color_mode=color_mode,
            image_characteristics=plan_image_chars,
            ph_interpretation=plan_ph_interp,
            descriptor_intuition=analysis.get("descriptor_intuition", ""),
            ph_signals_text=ph_signals_text,
            long_term_memory=ltm_text,
            descriptor_properties=descriptor_properties,
            reasoning_chains=reasoning_chains,
            benchmark_rankings=benchmark_rankings,
            parameter_table=parameter_table,
            top_ranked=top_ranked,
            top_accuracy=top_accuracy,
            learned_rules=learned_rules,
            retry_context=retry_context,
        )

        self._vprint_prompt(prompt, max_chars=3000)

        # Call LLM (no tools)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse plan JSON
        plan = self._parse_json_response(response_text, defaults={
            "reasoning": "",
            "stance": "FOLLOW",
            "primary_descriptor": top_ranked,
            "primary_params": {},
            "backup_descriptor": "persistence_statistics",
            "request_fusion": False,
        })

        # Validate descriptor name
        descriptor = plan.get("primary_descriptor", top_ranked)
        if descriptor not in SUPPORTED_DESCRIPTORS:
            if self.verbose:
                print(f"  [WARN] Invalid descriptor '{descriptor}', falling back to {top_ranked}")
            descriptor = top_ranked
            plan["primary_descriptor"] = descriptor

        # Inject optimal params from benchmark rules if not provided
        from .skills.rules_data import get_optimal_params
        if not plan.get("primary_params"):
            plan["primary_params"] = get_optimal_params(descriptor, object_type)
        else:
            # Merge: benchmark defaults + LLM overrides
            optimal = get_optimal_params(descriptor, object_type)
            merged = {**optimal, **plan["primary_params"]}
            plan["primary_params"] = merged

        # Track stance
        stance = plan.get("stance", "FOLLOW").upper()
        if "DEVIATE" in (response_text or "").upper():
            stance = "DEVIATE"

        # Log interaction
        interaction = LLMInteraction(
            step="v8_plan",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        if self.verbose:
            print(f"\n  Descriptor: {descriptor} (stance={stance})")
            print(f"  Params: {plan.get('primary_params', {})}")

        return {
            "_v8_plan_context": plan,
            "_benchmark_stance": stance,
            "_color_mode": color_mode,
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V8 Plan: {descriptor} (stance={stance})"
            ],
        }

    def _v8_extract(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 4: EXTRACT — Programmatically execute chosen descriptor.

        Uses the plan's primary_descriptor and primary_params. Handles
        per-channel mode (R,G,B separate + concatenate) and auto-injection.
        """
        import numpy as np
        from PIL import Image as PILImage

        self._vprint_header("V8 EXTRACT")

        plan = state.get("_v8_plan_context") or {}
        analysis = state.get("_v8_analysis_context") or {}
        descriptor_name = plan.get("primary_descriptor", "persistence_statistics")
        descriptor_params = plan.get("primary_params", {})
        color_mode = analysis.get("color_mode", state.get("_color_mode", "grayscale"))

        # Clean params: remove non-tool keys
        clean_params = {k: v for k, v in descriptor_params.items()
                        if k not in ("dim", "total_dim", "classifier", "color_mode")}

        new_memory = list(state["short_term_memory"])
        feature_quality = None

        # Image-based vs PH-based
        IMAGE_BASED = {"minkowski_functionals", "euler_characteristic_curve",
                       "euler_characteristic_transform", "edge_histogram", "lbp_texture"}

        # Per-channel execution
        if color_mode == "per_channel" and descriptor_name in self.tools:
            try:
                rgb_img = np.array(PILImage.open(state["image_path"]).convert("RGB"),
                                   dtype=np.float32) / 255.0
                channels = {"R": rgb_img[:, :, 0], "G": rgb_img[:, :, 1], "B": rgb_img[:, :, 2]}

                is_image_based = descriptor_name in IMAGE_BASED
                channel_vectors = []

                # Get grayscale PH from short-term memory for PH-based descriptors
                grayscale_ph = None
                for name, output in state["short_term_memory"]:
                    if name == "compute_ph" and isinstance(output, dict) and output.get("success"):
                        grayscale_ph = output.get("persistence", {})
                        break

                # Use filtration_type from perceive decisions
                perceive_decisions = state.get("_perceive_decisions") or {}
                extract_filtration = perceive_decisions.get(
                    "filtration_type",
                    state.get("_ph_stats", {}).get("filtration", "sublevel"),
                )

                for ch_name in ["R", "G", "B"]:
                    ch_args = clean_params.copy()
                    if is_image_based:
                        ch_args["image_array"] = channels[ch_name].tolist()
                    else:
                        # Compute PH on this channel
                        if "compute_ph" in self.tools:
                            ch_ph_result = self.tools["compute_ph"].invoke({
                                "image_array": channels[ch_name].tolist(),
                                "filtration_type": extract_filtration,
                                "max_dimension": 1,
                            })
                            if ch_ph_result.get("success"):
                                ch_args["persistence_data"] = ch_ph_result.get("persistence", {})
                            else:
                                ch_args["persistence_data"] = grayscale_ph or {}
                        else:
                            ch_args["persistence_data"] = grayscale_ph or {}

                    ch_result = self.tools[descriptor_name].invoke(ch_args)
                    fv = ch_result.get("combined_vector") or ch_result.get("feature_vector")
                    if fv is not None:
                        channel_vectors.append(np.asarray(fv, dtype=np.float64))

                if channel_vectors:
                    concat_vector = np.concatenate(channel_vectors).tolist()
                    result = {
                        "success": True,
                        "combined_vector": concat_vector,
                        "vector_length": len(concat_vector),
                        "tool_name": descriptor_name,
                        "color_mode": "per_channel",
                        "per_channel_dims": [len(v) for v in channel_vectors],
                    }
                    for k, v in clean_params.items():
                        result[k] = v
                else:
                    result = {"success": False, "error": "No per-channel features extracted"}

                self._vprint_tool_output(descriptor_name, result, result.get("success", False))
                new_memory.append((descriptor_name, result))

                if result.get("success"):
                    arr = np.asarray(result["combined_vector"], dtype=np.float64)
                    feature_quality = self._compute_feature_quality(arr)

            except Exception as e:
                result = {"success": False, "error": str(e)}
                new_memory.append((descriptor_name, result))
                self._vprint_tool_output(descriptor_name, str(e), False)

        else:
            # Grayscale or single-channel extraction
            tool_args = clean_params.copy()
            tool_args = self._auto_inject_args(descriptor_name, tool_args,
                                               {**state, "short_term_memory": new_memory})

            if descriptor_name in self.tools:
                try:
                    result = self.tools[descriptor_name].invoke(tool_args)
                    self._vprint_tool_output(descriptor_name, result, result.get("success", False))
                    new_memory.append((descriptor_name, result))

                    if result.get("success", False):
                        fv = result.get("combined_vector") or result.get("feature_vector")
                        if fv is not None:
                            arr = np.asarray(fv, dtype=np.float64)
                            feature_quality = self._compute_feature_quality(arr)
                except Exception as e:
                    new_memory.append((descriptor_name, {"success": False, "error": str(e)}))
                    self._vprint_tool_output(descriptor_name, str(e), False)
            else:
                new_memory.append((descriptor_name, {"success": False, "error": f"Tool '{descriptor_name}' not found"}))

        updates = {
            "short_term_memory": new_memory,
        }
        if feature_quality is not None:
            updates["_feature_quality"] = feature_quality

        return updates

    def _v8_extract_with_report(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 4 (C3 ablation): EXTRACT + build final report (no REFLECT).

        Runs the normal extraction, then builds the AgentReport and marks
        the task as complete — skipping the REFLECT phase entirely.
        """
        import time

        # Run normal extraction first
        extract_updates = self._v8_extract(state)

        # Merge extract updates into a working state copy
        merged_state = {**state, **extract_updates}

        plan = merged_state.get("_v8_plan_context") or {}
        analysis = merged_state.get("_v8_analysis_context") or {}
        ph_stats = merged_state.get("_ph_stats") or {}
        feature_quality = merged_state.get("_feature_quality") or {}
        descriptor_name = plan.get("primary_descriptor", "unknown")
        descriptor_params = plan.get("primary_params", {})
        object_type = analysis.get("object_type", "unknown")
        stance = merged_state.get("_benchmark_stance", "FOLLOW")
        elapsed = time.time() - getattr(self, "_workflow_start_time", time.time())
        n_tools = len(merged_state.get("short_term_memory", []))
        n_llm_calls = len(merged_state.get("llm_interactions", []))

        agent_report = AgentReport(
            descriptor=descriptor_name,
            object_type=object_type,
            reasoning_chain=plan.get("reasoning", ""),
            image_analysis=analysis.get("image_characteristics", ""),
            descriptor_rationale=plan.get("reasoning", ""),
            alternatives_considered=plan.get("backup_descriptor", ""),
            parameters=descriptor_params,
            feature_dimension=feature_quality.get("dimension", 0),
            color_mode=merged_state.get("_color_mode") or "grayscale",
            actual_dimension=feature_quality.get("dimension", 0),
            sparsity_pct=feature_quality.get("sparsity", 0.0),
            variance=feature_quality.get("variance", 0.0),
            nan_count=feature_quality.get("nan_count", 0),
            quality_ok=True,  # No REFLECT to check — assume OK
            error_analysis="",
            experience="",
            observe_object_type=analysis.get("object_type", ""),
            observe_color_mode=analysis.get("color_mode", ""),
            observe_filtration=merged_state.get("_perceive_decisions", {}).get(
                "filtration_type", ph_stats.get("filtration", "sublevel")),
            observe_reasoning=analysis.get("image_characteristics", ""),
            object_type_correct=analysis.get("_object_type_correct"),
            benchmark_stance=stance,
            ph_interpretation=analysis.get("ph_interpretation", ""),
            descriptor_intuition=analysis.get("descriptor_intuition", ""),
            n_llm_calls=n_llm_calls,
            n_tools_executed=n_tools,
            retry_used=False,
            total_time_seconds=round(elapsed, 2),
        )

        report = agent_report.to_dict()
        extract_updates["final_answer"] = report
        extract_updates["confidence"] = 85.0
        extract_updates["evidence"] = [str(item) for item in merged_state["short_term_memory"]]
        extract_updates["task_complete"] = True
        extract_updates["_reflect_decision"] = "COMPLETE"
        extract_updates["_reflect_history"] = []
        extract_updates["_v8_reflect_experience"] = None

        return extract_updates

    def _v8_reflect(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 5: REFLECT — LLM evaluates quality and generates experience entry.

        Key change from v7: The experience_entry is written to long-term memory
        for cross-session learning. Within-session: memory from image 1's REFLECT
        is available for image 2's PLAN phase.
        """
        import json as json_mod
        import time

        self._vprint_header("V8 REFLECT")
        new_interactions = list(state.get("llm_interactions", []))

        ph_stats = state.get("_ph_stats") or {}
        feature_quality = state.get("_feature_quality") or {}
        analysis = state.get("_v8_analysis_context") or {}
        plan = state.get("_v8_plan_context") or {}

        descriptor_name = plan.get("primary_descriptor", "unknown")
        descriptor_params = plan.get("primary_params", {})
        object_type = analysis.get("object_type", "unknown")
        stance = state.get("_benchmark_stance", "FOLLOW")

        # Get relevant long-term memory (skip for C2 ablation)
        relevant_memory = "No past experiences."
        if not self.ablate_memory and self.long_term_memory is not None:
            query = f"{object_type} {descriptor_name}"
            relevant = self.long_term_memory.search_experiences(query, n=3)
            if relevant:
                ltm_lines = []
                for i, entry in enumerate(relevant, 1):
                    ltm_lines.append(f"  {i}. {entry.experience[:200]}")
                relevant_memory = "\n".join(ltm_lines)

        # Build descriptor-specific quality assessment
        sparsity_val = feature_quality.get("sparsity", 0)
        variance_val = feature_quality.get("variance", 0)
        quality_assessment = build_quality_assessment_text(
            descriptor_name, sparsity_val, variance_val,
        )

        # Get backup descriptor from plan
        backup_descriptor = plan.get("backup_descriptor", "persistence_statistics")

        # Build rich feature statistics string for LLM
        fq = feature_quality
        feature_stats_lines = [
            f"- Dimension: {fq.get('dimension', 0)}",
            f"- Sparsity: {fq.get('sparsity', 0):.1f}% (fraction of near-zero values)",
            f"- Variance: {fq.get('variance', 0):.6f}",
            f"- Mean: {fq.get('mean', 0):.6f}, Std: {fq.get('std', 0):.6f}",
            f"- Min: {fq.get('min', 0):.6f}, Max: {fq.get('max', 0):.6f}",
            f"- Dynamic range: {fq.get('dynamic_range', 0):.6f}",
            f"- IQR (25th-75th): {fq.get('p25', 0):.6f} to {fq.get('p75', 0):.6f} (IQR={fq.get('iqr', 0):.6f})",
            f"- Kurtosis: {fq.get('kurtosis', 'N/A')}, Skewness: {fq.get('skewness', 'N/A')}",
            f"- NaN count: {fq.get('nan_count', 0)}, Inf count: {fq.get('inf_count', 0)}",
            f"- Informative features (>1% variance): {fq.get('n_informative_features', 'N/A')} / {fq.get('dimension', 0)}",
            f"- Constant features: {fq.get('n_constant_features', 0)}",
        ]
        feature_stats_text = "\n".join(feature_stats_lines)

        prompt = V8_REFLECT_PROMPT.format(
            object_type=object_type,
            image_characteristics=analysis.get("image_characteristics", ""),
            ph_interpretation=analysis.get("ph_interpretation", ""),
            descriptor_name=descriptor_name,
            stance=stance,
            params=json_mod.dumps(descriptor_params) if descriptor_params else "defaults",
            backup_descriptor=backup_descriptor,
            feature_stats=feature_stats_text,
            quality_assessment=quality_assessment,
            relevant_memory=relevant_memory,
        )

        self._vprint_prompt(prompt)

        # Call LLM (no tools)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse reflect JSON
        reflect = self._parse_json_response(response_text, defaults={
            "quality_ok": True,
            "quality_reasoning": "",
            "decision": "COMPLETE",
            "retry_suggestion": "",
            "experience_entry": {},
        })

        # NOTE: No code-level overrides — REFLECT is fully LLM-driven.
        # The LLM receives rich quality data (sparsity, variance, distribution
        # stats, expected ranges) and makes the COMPLETE/RETRY decision itself.

        # Log interaction
        interaction = LLMInteraction(
            step="v8_reflect",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        decision = reflect.get("decision", "COMPLETE").upper()

        if self.verbose:
            print(f"\n  Decision: {decision}")
            print(f"  Quality OK: {reflect.get('quality_ok', True)}")

        # Build reflect record
        reflect_record = {
            "round": len(state.get("_reflect_history", [])) + 1,
            "descriptor": descriptor_name,
            "descriptor_params": descriptor_params,
            "feature_quality": feature_quality,
            "quality_ok": reflect.get("quality_ok", True),
            "decision": decision,
            "reasoning": reflect.get("quality_reasoning", ""),
        }

        updates = {
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V8 Reflect: {decision} - {reflect.get('quality_reasoning', '')[:80]}"
            ],
            "_reflect_history": list(state.get("_reflect_history", [])) + [reflect_record],
            "_v8_reflect_experience": reflect.get("experience_entry"),
        }

        # Write experience to long-term memory (skip for C2 ablation)
        # NOTE: Use `is not None` instead of truthiness check because
        # LongTermMemory defines __len__, making empty LTM falsy.
        experience_entry = reflect.get("experience_entry", {})
        if experience_entry and not self.ablate_memory and self.long_term_memory is not None:
            lesson = experience_entry.get("lesson", "")
            if lesson:
                from .memory.long_term import ReflectionEntry as LTMReflectionEntry
                ltm_entry = LTMReflectionEntry(
                    round=reflect_record["round"],
                    error_analysis=reflect.get("quality_reasoning", ""),
                    suggestion=descriptor_name,
                    experience=lesson,
                    context=f"object_type={object_type}, descriptor={descriptor_name}, "
                            f"quality={experience_entry.get('quality', '?')}",
                )
                self.long_term_memory.add(ltm_entry)
                if self.verbose:
                    print(f"  [LTM] Wrote experience: {lesson[:80]}")

        elapsed = time.time() - getattr(self, "_workflow_start_time", time.time())

        if decision == "RETRY_EXTRACT" and state.get("_retry_count", 0) < 3 and not self._is_time_exceeded():
            # Retry: clean descriptor from memory
            cleaned_memory = [
                (name, out) for name, out in state["short_term_memory"]
                if name not in SUPPORTED_DESCRIPTORS
            ]
            updates["short_term_memory"] = cleaned_memory
            updates["_feature_quality"] = None
            retry_count = state.get("_retry_count", 0) + 1
            updates["_retry_count"] = retry_count

            # Accumulate all failed descriptors across retries
            failed_so_far = list(state.get("_failed_descriptors", []))
            failed_so_far.append(descriptor_name)
            updates["_failed_descriptors"] = failed_so_far

            # First retry: use backup descriptor directly (skip PLAN LLM)
            # This ensures retries actually try something different.
            backup = plan.get("backup_descriptor", "persistence_statistics")
            if backup in failed_so_far:
                # Backup already failed too — fall back to persistence_statistics
                backup = "persistence_statistics"
                if backup in failed_so_far:
                    backup = "betti_curves"  # ultimate fallback

            if retry_count == 1 and backup != descriptor_name:
                # Force backup descriptor: update plan context, skip PLAN LLM
                from .skills.rules_data import get_optimal_params
                forced_plan = dict(plan)
                forced_plan["primary_descriptor"] = backup
                forced_plan["primary_params"] = get_optimal_params(backup, object_type)
                forced_plan["reasoning"] = (
                    f"RETRY: '{descriptor_name}' failed quality check "
                    f"({reflect.get('quality_reasoning', '')[:100]}). "
                    f"Switching to backup descriptor '{backup}'."
                )
                forced_plan["stance"] = "DEVIATE"
                updates["_v8_plan_context"] = forced_plan
                updates["_benchmark_stance"] = "DEVIATE"
                updates["_reflect_decision"] = "RETRY_FORCED"
                if self.verbose:
                    print(f"  [RETRY] Forcing backup descriptor: {backup} (was {descriptor_name})")
            else:
                # 2nd+ retry or backup==primary: go through PLAN LLM with blacklist
                failed_list = ", ".join(failed_so_far)
                updates["_retry_feedback"] = (
                    f"RETRY {retry_count}/3. "
                    f"Latest failure: '{descriptor_name}' — {reflect.get('quality_reasoning', '')}. "
                    f"Previously failed descriptors: [{failed_list}]. "
                    f"Do NOT use any of these again. "
                    f"LLM suggestion: {reflect.get('retry_suggestion', '')}"
                )
                updates["_reflect_decision"] = "RETRY"
        else:
            # COMPLETE: build final report
            decisions = state.get("_observe_decisions") or {}
            n_tools = len(state["short_term_memory"])
            n_llm_calls = len(new_interactions)

            agent_report = AgentReport(
                descriptor=descriptor_name,
                object_type=object_type,
                reasoning_chain=plan.get("reasoning", ""),
                image_analysis=analysis.get("image_characteristics", ""),
                descriptor_rationale=plan.get("reasoning", ""),
                alternatives_considered=plan.get("backup_descriptor", ""),
                parameters=descriptor_params,
                feature_dimension=feature_quality.get("dimension", 0),
                color_mode=state.get("_color_mode") or "grayscale",
                actual_dimension=feature_quality.get("dimension", 0),
                sparsity_pct=feature_quality.get("sparsity", 0.0),
                variance=feature_quality.get("variance", 0.0),
                nan_count=feature_quality.get("nan_count", 0),
                quality_ok=reflect.get("quality_ok", True),
                error_analysis=reflect.get("quality_reasoning", ""),
                experience=experience_entry.get("lesson", ""),
                observe_object_type=analysis.get("object_type", ""),
                observe_color_mode=analysis.get("color_mode", ""),
                observe_filtration=state.get("_perceive_decisions", {}).get("filtration_type", ph_stats.get("filtration", "sublevel")),
                observe_reasoning=analysis.get("image_characteristics", ""),
                object_type_correct=analysis.get("_object_type_correct"),
                benchmark_stance=stance,
                ph_interpretation=analysis.get("ph_interpretation", ""),
                descriptor_intuition=analysis.get("descriptor_intuition", ""),
                n_llm_calls=n_llm_calls,
                n_tools_executed=n_tools,
                retry_used=state.get("_retry_count", 0) > 0,
                total_time_seconds=round(elapsed, 2),
            )

            report = agent_report.to_dict()
            quality_ok = reflect.get("quality_ok", True)
            confidence = 85.0 if quality_ok else 50.0

            updates["final_answer"] = report
            updates["confidence"] = confidence
            updates["evidence"] = [str(item) for item in state["short_term_memory"]]
            updates["task_complete"] = True
            updates["_reflect_decision"] = "COMPLETE"

        return updates

    def _v8_route_perceive(self, state: TopoAgentState) -> Literal["tools", "analyze"]:
        """Route after PERCEIVE: if tool_calls → tools, else → ANALYZE.

        Ensures minimum pipeline (image_loader + compute_ph) before advancing.
        Uses forced progression in _v8_perceive_tools to prevent stuck loops.
        """
        last_msg = state["messages"][-1] if state["messages"] else None
        turns = state.get("_v8_perceive_turns", 0)

        # Safety: max 8 perceive turns
        if turns >= 8:
            return "analyze"

        # Time limit check
        if self._is_time_exceeded():
            return "analyze"

        if last_msg is None:
            return "analyze"

        tools_called = {name for name, _ in state.get("short_term_memory", [])}
        has_img = "image_loader" in tools_called
        has_ph = "compute_ph" in tools_called

        tool_calls = getattr(last_msg, "tool_calls", []) or []
        if tool_calls:
            # LLM wants to call more tools — let it (forced progression
            # in _v8_perceive_tools handles duplicates)
            return "tools"

        # No tool calls = LLM thinks it's done perceiving
        # But enforce minimum pipeline: image_loader + compute_ph
        if not has_img or not has_ph:
            if turns < 5:
                # Force another perceive turn — prompt will show what's
                # already been called and nudge the LLM
                return "tools"

        return "analyze"

    def _v8_route_reflect(self, state: TopoAgentState) -> Literal["complete", "retry", "retry_forced"]:
        """Route after REFLECT: complete, retry (via PLAN LLM), or retry_forced (skip PLAN, use backup)."""
        decision = state.get("_reflect_decision", "COMPLETE").upper()

        if self._is_time_exceeded() and decision != "COMPLETE":
            if self.verbose:
                print(f"  [TIME LIMIT] Forcing COMPLETE")
            return "complete"

        if decision == "RETRY_FORCED":
            return "retry_forced"
        if decision == "RETRY":
            return "retry"
        return "complete"

    # =========================================================================
    # v9 Agentic Pipeline: 6-Phase Hypothesis-First Workflow
    # OBSERVE → INTERPRET → ANALYZE → ACT → EXTRACT → REFLECT
    # =========================================================================

    def _build_agentic_v9_workflow(self) -> Any:
        """Build the v9 6-phase hypothesis-first workflow.

        Architecture:
            OBSERVE (deterministic) -> INTERPRET (LLM#1) -> ANALYZE (LLM#2)
            -> ACT (LLM#3) -> EXTRACT (deterministic) -> REFLECT (LLM#4)
            -> END or retry back to ACT

        LLM calls: 4 (no retry), 6 (one retry), 8 max (two retries).
        """
        workflow = StateGraph(TopoAgentState)

        workflow.add_node("v9_observe", self._v9_observe)
        workflow.add_node("v9_interpret", self._v9_interpret)
        workflow.add_node("v9_analyze", self._v9_analyze)
        workflow.add_node("v9_act", self._v9_act)
        workflow.add_node("v9_extract", self._v9_extract)
        workflow.add_node("v9_reflect", self._v9_reflect)

        workflow.set_entry_point("v9_observe")
        workflow.add_edge("v9_observe", "v9_interpret")      # LLM #1
        workflow.add_edge("v9_interpret", "v9_analyze")      # LLM #2
        workflow.add_edge("v9_analyze", "v9_act")            # LLM #3
        workflow.add_edge("v9_act", "v9_extract")
        workflow.add_edge("v9_extract", "v9_reflect")        # LLM #4
        # REFLECT -> END or retry -> ACT
        workflow.add_conditional_edges("v9_reflect", self._v9_route_reflect,
            {"complete": END, "retry": "v9_act"})

        return workflow.compile()

    @staticmethod
    def _v9_scrub_query(query: str) -> str:
        """Strip all identifying info for blind inference in INTERPRET/ANALYZE."""
        return (
            "Analyze the provided medical image, compute its persistent homology, "
            "and determine the most suitable topology descriptor to produce a "
            "fixed-length feature vector."
        )

    def _v9_observe(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 1: OBSERVE -- deterministic tool execution.

        Runs image_loader, image_analyzer, compute_ph (sublevel), topological_features,
        and betti_ratios. No LLM calls. Always uses sublevel filtration.
        """
        import time
        import numpy as np
        from PIL import Image as PILImage

        if not hasattr(self, '_workflow_start_time'):
            self._workflow_start_time = time.time()

        self._vprint_header("V9 OBSERVE (deterministic)")

        new_memory = list(state["short_term_memory"])
        tools_used = []
        image_analysis = None
        ph_stats = None
        reasoning_trace = list(state["reasoning_trace"])

        # Helper to execute a tool with auto-injection
        def _exec_tool(name, extra_args=None):
            if name not in self.tools:
                if self.verbose:
                    print(f"  [V9 OBSERVE] Tool '{name}' not available, skipping")
                return None
            args = (extra_args or {}).copy()
            args = self._auto_inject_args(name, args,
                                          {**state, "short_term_memory": new_memory})
            try:
                result = self.tools[name].invoke(args)
                self._vprint_tool_output(name, result, result.get("success", False))
                new_memory.append((name, result))
                tools_used.append(name)
                return result
            except Exception as e:
                if self.verbose:
                    print(f"  [V9 OBSERVE] {name} failed: {e}")
                new_memory.append((name, {"success": False, "error": str(e)}))
                tools_used.append(name)
                return None

        # Detect original image channel count from file (before grayscale conversion)
        original_n_channels = 1
        try:
            pil_img = PILImage.open(state["image_path"])
            if pil_img.mode == "RGB":
                original_n_channels = 3
            elif pil_img.mode == "RGBA":
                original_n_channels = 3  # treat as RGB
            original_shape = list(np.array(pil_img).shape)
        except Exception:
            original_shape = []

        # 1. Load image (converts to grayscale for PH computation)
        _exec_tool("image_loader", {"image_path": state["image_path"]})

        # 2. Image analyzer (quantitative stats: SNR, contrast, edges, etc.)
        img_analysis_result = _exec_tool("image_analyzer",
                                         {"image_path": state["image_path"]})
        if img_analysis_result and img_analysis_result.get("success", False):
            image_analysis = img_analysis_result
            if "image_statistics" not in image_analysis:
                image_analysis["image_statistics"] = {}
            image_analysis["image_statistics"]["original_n_channels"] = original_n_channels
            image_analysis["image_statistics"]["original_shape"] = original_shape

        # 3. Compute PH with sublevel filtration (always default for v9)
        ph_result = _exec_tool("compute_ph", {
            "filtration_type": "sublevel",
            "max_dimension": 1,
        })

        if ph_result and ph_result.get("success", False):
            stats = ph_result.get("statistics", {})
            persistence_data = ph_result.get("persistence", {})
            h0_pers = self._extract_persistence_values(persistence_data.get("H0", []))
            h1_pers = self._extract_persistence_values(persistence_data.get("H1", []))
            h0_avg = float(np.mean(h0_pers)) if h0_pers else 0.0
            h1_avg = float(np.mean(h1_pers)) if h1_pers else 0.0
            ph_stats = {
                "H0_count": stats.get("H0", {}).get("count", 0),
                "H1_count": stats.get("H1", {}).get("count", 0),
                "H0_max_persistence": max(h0_pers) if h0_pers else 0.0,
                "H1_max_persistence": max(h1_pers) if h1_pers else 0.0,
                "H0_avg_persistence": h0_avg,
                "H1_avg_persistence": h1_avg,
                "filtration": ph_result.get("filtration_type", "sublevel"),
            }

        # 4. Topological features (62 stats)
        _exec_tool("topological_features")

        # 5. Betti ratios
        _exec_tool("betti_ratios")

        reasoning_trace.append(
            f"V9 Observe: executed {len(tools_used)} tools: {tools_used}"
        )

        scrubbed_query = self._v9_scrub_query(state.get("query", ""))

        updates = {
            "short_term_memory": new_memory,
            "_v9_mode": True,
            "_v9_scrubbed_query": scrubbed_query,
            "reasoning_trace": reasoning_trace,
        }
        if image_analysis is not None:
            updates["_v8_image_analysis"] = image_analysis
        if ph_stats is not None:
            updates["_ph_stats"] = ph_stats

        return updates

    def _v9_interpret(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 2: INTERPRET -- LLM call #1 (blind observation).

        The LLM sees image stats and PH data but NO dataset names, no benchmark
        rankings, no descriptor recommendations. It must infer the object type,
        modality, and color diagnostic value purely from quantitative data.
        """
        import json as json_mod

        self._vprint_header("V9 INTERPRET (LLM #1 -- blind)")
        new_interactions = list(state.get("llm_interactions", []))

        # Build image stats summary
        image_analysis = state.get("_v8_image_analysis") or {}
        if image_analysis:
            metrics = image_analysis.get("image_statistics", {})
            original_n_channels = metrics.get("original_n_channels", 1)
            original_shape = metrics.get("original_shape", [])
            img_stats = (
                f"Image shape: {original_shape}, channels: {original_n_channels}\n"
                f"SNR={metrics.get('snr_estimate', 'N/A')}, "
                f"contrast={metrics.get('contrast', 'N/A')}, "
                f"edge_density={metrics.get('edge_density', 'N/A')}, "
                f"bright_ratio={metrics.get('bright_ratio', 'N/A')}, "
                f"dark_ratio={metrics.get('dark_ratio', 'N/A')}, "
                f"variance={metrics.get('variance', 'N/A')}"
            )
        else:
            img_stats = "Image statistics not available."

        # Build PH stats summary
        ph_stats = state.get("_ph_stats") or {}
        if ph_stats:
            ph_stats_text = (
                f"H0_count={ph_stats.get('H0_count', 0)}, "
                f"H1_count={ph_stats.get('H1_count', 0)}, "
                f"H0_max_persistence={self._fmt_pers(ph_stats.get('H0_max_persistence', 0))}, "
                f"H1_max_persistence={self._fmt_pers(ph_stats.get('H1_max_persistence', 0))}, "
                f"H0_avg_persistence={self._fmt_pers(ph_stats.get('H0_avg_persistence', 0))}, "
                f"H1_avg_persistence={self._fmt_pers(ph_stats.get('H1_avg_persistence', 0))}, "
                f"filtration={ph_stats.get('filtration', 'sublevel')}"
            )
        else:
            ph_stats_text = "PH statistics not available."

        # Build topological features summary
        topo_summary = "topological_features was not called."
        for name, output in state["short_term_memory"]:
            if name == "topological_features" and isinstance(output, dict) and output.get("success"):
                features_by_dim = output.get("features_by_dimension", {})
                topo_lines = []
                for dim_name in ["H0", "H1"]:
                    dim_feats = features_by_dim.get(dim_name, {})
                    for k, v in sorted(dim_feats.items()):
                        if isinstance(v, (int, float)):
                            topo_lines.append(f"{dim_name}_{k}={v:.4f}" if isinstance(v, float) else f"{dim_name}_{k}={v}")
                topo_summary = ", ".join(topo_lines[:20])
                if len(topo_lines) > 20:
                    topo_summary += f" ...({len(topo_lines)} total features)"
                break

        # Build betti ratios summary
        betti_summary = "betti_ratios was not called."
        for name, output in state["short_term_memory"]:
            if name == "betti_ratios" and isinstance(output, dict) and output.get("success"):
                ratios = output.get("ratios", {})
                betti_lines = []
                for k, v in sorted(ratios.items()):
                    if isinstance(v, (int, float)):
                        betti_lines.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
                betti_summary = ", ".join(betti_lines) if betti_lines else str(output)[:300]
                break

        # Build object type taxonomy text
        taxonomy_lines = []
        for obj_type, info in OBJECT_TYPE_TAXONOMY.items():
            taxonomy_lines.append(f"**{obj_type}**:")
            taxonomy_lines.append(f"  PH signature: {info['ph_signature']}")
            taxonomy_lines.append(f"  Typical metrics: {info['typical_metrics']}")
            taxonomy_lines.append(f"  Image clues: {info['image_clues']}")
            taxonomy_lines.append("")
        taxonomy_text = "\n".join(taxonomy_lines)

        # Build prompt
        scrubbed_query = state.get("_v9_scrubbed_query") or self._v9_scrub_query("")
        prompt = V9_INTERPRET_PROMPT.format(
            scrubbed_query=scrubbed_query,
            image_stats=img_stats,
            ph_stats=ph_stats_text,
            topo_features_summary=topo_summary,
            betti_ratios=betti_summary,
            object_type_taxonomy=taxonomy_text,
        )

        self._vprint_prompt(prompt, max_chars=3000)

        # Call LLM with image (multimodal -- image+text-in/JSON-out)
        import base64, os
        image_path = state.get("image_path", "")
        message_content = [{"type": "text", "text": prompt}]
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}.get(ext.lstrip("."), "image/png")
            message_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
            if self.verbose:
                print(f"  [Image attached: {image_path} ({mime})]")
        else:
            if self.verbose:
                print(f"  [WARNING: Image not found at '{image_path}', proceeding text-only]")
        response = self.model.invoke([HumanMessage(content=message_content)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse JSON
        interpret_output = self._parse_json_response(response_text, defaults={
            "modality_guess": "unknown",
            "modality_evidence": "",
            "object_type_guess": "surface_lesions",
            "object_type_evidence": "",
            "image_profile": "",
            "ph_profile": "",
            "color_diagnostic": False,
            "color_evidence": "",
        })

        # Validate object_type_guess against ground truth (for logging only)
        from .skills.rules_data import DATASET_TO_OBJECT_TYPE
        gt_object_type = None
        search_text = (state.get("query", "") + " " + state.get("image_path", "")).lower()
        for ds_name, obj_type in DATASET_TO_OBJECT_TYPE.items():
            if ds_name.lower() in search_text:
                gt_object_type = obj_type
                break
        interpret_output["_gt_object_type"] = gt_object_type
        interpret_output["_object_type_correct"] = (
            interpret_output.get("object_type_guess") == gt_object_type if gt_object_type else None
        )

        # Log interaction
        interaction = LLMInteraction(
            step="v9_interpret",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        if self.verbose:
            print(f"\n  Modality: {interpret_output.get('modality_guess')}")
            print(f"  Object type: {interpret_output.get('object_type_guess')} "
                  f"(GT={gt_object_type}, correct={interpret_output.get('_object_type_correct')})")
            print(f"  Color diagnostic: {interpret_output.get('color_diagnostic')}")

        return {
            "_v9_interpret_output": interpret_output,
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V9 Interpret: object_type={interpret_output.get('object_type_guess', '?')}, "
                f"modality={interpret_output.get('modality_guess', '?')}, "
                f"color_diagnostic={interpret_output.get('color_diagnostic', '?')}"
            ],
        }

    def _v9_analyze(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 3: ANALYZE -- LLM call #2 (hypothesis formation).

        The LLM sees its own INTERPRET output plus descriptor properties and
        reasoning principles (no rankings, no accuracy numbers). It forms a
        hypothesis about which descriptor to use and why, with parameter
        intuition derived from PH data.
        """
        import json as json_mod
        from .skills.rules_data import compute_ph_signals

        self._vprint_header("V9 ANALYZE (LLM #2 -- hypothesis)")
        new_interactions = list(state.get("llm_interactions", []))

        interpret_output = state.get("_v9_interpret_output") or {}

        # Compute PH signals for observation text
        ph_stats = state.get("_ph_stats") or {}
        ph_signals = []
        if ph_stats:
            ph_signals = compute_ph_signals(
                h0_count=ph_stats.get("H0_count", 0),
                h1_count=ph_stats.get("H1_count", 0),
                h0_avg_persistence=ph_stats.get("H0_avg_persistence", 0.0),
                h1_avg_persistence=ph_stats.get("H1_avg_persistence", 0.0),
                h0_max_persistence=ph_stats.get("H0_max_persistence", 1.0),
                h1_max_persistence=ph_stats.get("H1_max_persistence", 1.0),
            )

        # Build ph signal observations (WITHOUT descriptor recommendations)
        ph_signal_observations = build_ph_signal_observations(ph_signals)

        # Build descriptor properties (WITHOUT rankings/accuracy)
        descriptor_properties = build_descriptor_properties_only()

        # Build reasoning principles (WITHOUT recommended/avoid lists)
        reasoning_principles = build_reasoning_principles()

        # Build parameter reasoning guide for ALL descriptors that have entries
        param_guide_sections = []
        for desc_name in PARAMETER_REASONING_GUIDE:
            section = build_parameter_reasoning_text(desc_name)
            param_guide_sections.append(section)
        parameter_reasoning_guide = "\n\n".join(param_guide_sections)

        # Build raw PH stats text for the LLM to reason about directly
        raw_ph_stats_text = "PH statistics not available."
        if ph_stats:
            h0 = ph_stats.get("H0_count", 0)
            h1 = ph_stats.get("H1_count", 0)
            total = h0 + h1
            ratio = h1 / max(h0, 1)
            raw_ph_stats_text = (
                f"H0_count (connected components): {h0}\n"
                f"H1_count (loops/holes): {h1}\n"
                f"Total features: {total}\n"
                f"H1/H0 ratio: {ratio:.3f}\n"
                f"H0_avg_persistence: {ph_stats.get('H0_avg_persistence', 0.0):.6f}\n"
                f"H1_avg_persistence: {ph_stats.get('H1_avg_persistence', 0.0):.6f}\n"
                f"H0_max_persistence: {ph_stats.get('H0_max_persistence', 0.0):.6f}\n"
                f"H1_max_persistence: {ph_stats.get('H1_max_persistence', 0.0):.6f}\n"
                f"Filtration: {ph_stats.get('filtration', 'sublevel')}"
            )

        # Extract domain context (strips dataset names)
        domain_context = extract_domain_context(state.get("query", ""))
        if not domain_context:
            domain_context = "No domain context available."

        # Object type guess from INTERPRET
        object_type_guess = interpret_output.get("object_type_guess", "unknown")

        # Strip ground-truth fields so ANALYZE stays blind
        interpret_for_prompt = {
            k: v for k, v in interpret_output.items()
            if not k.startswith("_gt_") and k != "_object_type_correct"
        }

        prompt = V9_ANALYZE_PROMPT.format(
            interpret_output=json_mod.dumps(interpret_for_prompt, indent=2, default=str),
            domain_context=domain_context,
            object_type_guess=object_type_guess,
            ph_signal_observations=ph_signal_observations,
            raw_ph_stats=raw_ph_stats_text,
            descriptor_properties=descriptor_properties,
            reasoning_principles=reasoning_principles,
            parameter_reasoning_guide=parameter_reasoning_guide,
        )

        self._vprint_prompt(prompt, max_chars=3000)

        # Call LLM (no tools)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse JSON
        hypothesis = self._parse_json_response(response_text, defaults={
            "object_type_reconciled": "",
            "color_mode": "grayscale",
            "color_reasoning": "",
            "descriptor_hypothesis": "persistence_statistics",
            "descriptor_reasoning": "",
            "alternatives": [],
            "parameter_intuition": {},
            "confidence": "medium",
        })

        # Log interaction
        interaction = LLMInteraction(
            step="v9_analyze",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        if self.verbose:
            print(f"\n  Hypothesis: {hypothesis.get('descriptor_hypothesis')}")
            print(f"  Color mode: {hypothesis.get('color_mode')}")
            print(f"  Confidence: {hypothesis.get('confidence')}")
            print(f"  Alternatives: {hypothesis.get('alternatives', [])}")

        return {
            "_v9_hypothesis": hypothesis,
            "_ph_signals": ph_signals,
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V9 Analyze: hypothesis={hypothesis.get('descriptor_hypothesis', '?')}, "
                f"color_mode={hypothesis.get('color_mode', '?')}, "
                f"confidence={hypothesis.get('confidence', '?')}"
            ],
        }

    def _v9_act(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 4: ACT -- LLM call #3 (reconcile hypothesis with benchmarks).

        The LLM sees its blind hypothesis from ANALYZE plus the full benchmark
        advisory (rankings, accuracy, optimal parameters). It reconciles the
        two sources and makes the final descriptor + parameter decision.

        KEY CHANGE from v8: Does NOT merge with optimal params. Uses LLM params
        AS-IS. Only falls back to benchmark params if LLM provides no params.
        """
        import json as json_mod
        from .skills.rules_data import (
            build_ph_signals_text, get_top_recommendation,
            get_optimal_params, SUPPORTED_TOP_PERFORMERS,
        )

        self._vprint_header("V9 ACT (LLM #3 -- reconcile)")
        new_interactions = list(state.get("llm_interactions", []))

        hypothesis = state.get("_v9_hypothesis") or {}
        interpret_output = state.get("_v9_interpret_output") or {}
        ph_signals = state.get("_ph_signals") or []
        ph_stats = state.get("_ph_stats") or {}

        # Object type: prefer reconciled type from ANALYZE, fall back to INTERPRET guess
        from .skills.rules_data import OBJECT_TYPES
        reconciled_ot = hypothesis.get("object_type_reconciled", "")
        if reconciled_ot not in OBJECT_TYPES:
            reconciled_ot = ""
        object_type = (reconciled_ot
                       or interpret_output.get("object_type_guess", "surface_lesions"))
        if object_type not in OBJECT_TYPES:
            object_type = "surface_lesions"

        # Build TIERED benchmark advisory (no exact accuracy %)
        tiered_benchmark_advisory = build_tiered_benchmark_advisory(object_type)

        # Long-term memory consultation (v9-style profile-based search)
        ltm_text = "No past experiences."
        memory_stats = {
            "entries_available": 0,
            "entries_retrieved": 0,
            "top_similarity": 0.0,
            "memory_influenced": False,
        }
        if self.long_term_memory is not None:
            # Build ph_metrics for profile-based search
            ph_metrics = {
                "h0_count": ph_stats.get("H0_count", 0),
                "h1_count": ph_stats.get("H1_count", 0),
                "h0_avg_pers": ph_stats.get("H0_avg_persistence", 0.0),
                "h1_avg_pers": ph_stats.get("H1_avg_persistence", 0.0),
            }
            ltm_text = self.long_term_memory.format_for_v9_prompt(
                object_type=object_type,
                ph_metrics=ph_metrics,
                n=5,
            )
            # Collect memory stats
            from .memory.long_term import V9ExperienceEntry
            v9_total = sum(1 for e in self.long_term_memory._memory
                          if isinstance(e, V9ExperienceEntry))
            results = self.long_term_memory.search_by_profile(
                object_type, ph_metrics, n=5
            )
            memory_stats["entries_available"] = v9_total
            memory_stats["entries_retrieved"] = len(results)
            if results:
                memory_stats["top_similarity"] = results[0][0]
                # Memory influenced if there's a relevant entry with high similarity
                memory_stats["memory_influenced"] = results[0][0] > 0.85

        # Build PH signals text (full version WITH recommendations for ACT)
        top_desc, _ = get_top_recommendation(object_type)
        ph_signals_text = build_ph_signals_text(ph_signals, top_desc)

        # Retry context
        retry_context = ""
        retry_feedback = state.get("_retry_feedback")
        blacklisted = state.get("_failed_descriptors", [])
        if retry_feedback:
            retry_context = (
                f"\n## RETRY Context\n{retry_feedback}\n"
                f"Blacklisted descriptors (DO NOT use): {blacklisted}\n"
            )
            # Add retry info to hypothesis so LLM sees it
            hypothesis = dict(hypothesis)
            hypothesis["_retry_note"] = retry_feedback

        prompt = V9_ACT_PROMPT.format(
            hypothesis_json=json_mod.dumps(hypothesis, indent=2, default=str),
            tiered_benchmark_advisory=tiered_benchmark_advisory,
            long_term_memory=ltm_text,
            ph_signals_text=ph_signals_text,
            original_query=state.get("query", ""),
        )

        # Append retry context to prompt if present
        if retry_context:
            prompt += retry_context

        self._vprint_prompt(prompt, max_chars=3000)

        # Call LLM (no tools)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse JSON
        act_decision = self._parse_json_response(response_text, defaults={
            "reasoning": "",
            "final_descriptor": top_desc,
            "final_params": {},
            "color_mode": hypothesis.get("color_mode", "grayscale"),
            "backup_descriptor": "persistence_statistics",
            "backup_reasoning": "",
        })

        # Validate descriptor name
        descriptor = act_decision.get("final_descriptor", top_desc)
        if descriptor not in SUPPORTED_DESCRIPTORS:
            if self.verbose:
                print(f"  [WARN] Invalid descriptor '{descriptor}', falling back to {top_desc}")
            descriptor = top_desc
            act_decision["final_descriptor"] = descriptor

        # Check blacklist
        if descriptor in blacklisted:
            if self.verbose:
                print(f"  [WARN] Descriptor '{descriptor}' is blacklisted, using backup")
            backup = act_decision.get("backup_descriptor", "persistence_statistics")
            if backup in blacklisted:
                # Find first non-blacklisted descriptor from the rankings
                performers = SUPPORTED_TOP_PERFORMERS.get(object_type, [])
                for entry in performers:
                    if entry["descriptor"] not in blacklisted:
                        backup = entry["descriptor"]
                        break
                else:
                    backup = "persistence_statistics"
                    if backup in blacklisted:
                        backup = "betti_curves"
            descriptor = backup
            act_decision["final_descriptor"] = descriptor

        # KEY CHANGE: Use LLM params AS-IS, only fallback if empty
        if not act_decision.get("final_params"):
            act_decision["final_params"] = get_optimal_params(descriptor, object_type)
            if self.verbose:
                print(f"  [PARAMS] No LLM params, using benchmark defaults")
        else:
            # Clean non-numeric/non-string values but DO NOT merge with benchmark
            raw_params = act_decision["final_params"]
            clean = {}
            for k, v in raw_params.items():
                if k in ("dim", "total_dim", "classifier", "color_mode"):
                    continue
                # Try to convert string numbers to proper types
                if isinstance(v, str):
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                clean[k] = v
            act_decision["final_params"] = clean
            if self.verbose:
                print(f"  [PARAMS] Using LLM params as-is: {clean}")

        # Infer stance post-hoc (for logging/analysis — NOT shown to LLM)
        hyp_desc = hypothesis.get("descriptor_hypothesis", "")
        if descriptor == hyp_desc:
            stance = "CONFIRMED_HYPOTHESIS"
        elif descriptor == top_desc:
            stance = "SWITCHED_TO_BENCHMARK"
        else:
            stance = "CHOSE_ALTERNATIVE"

        # Store in act_decision for downstream analysis
        act_decision["_inferred_stance"] = stance
        act_decision["_hypothesis_descriptor"] = hyp_desc
        act_decision["_benchmark_top"] = top_desc

        # Log interaction
        interaction = LLMInteraction(
            step="v9_act",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        if self.verbose:
            print(f"\n  Final descriptor: {descriptor} ({stance})")
            print(f"  Hypothesis was: {hyp_desc}, Benchmark top: {top_desc}")
            print(f"  Final params: {act_decision.get('final_params', {})}")
            print(f"  Color mode: {act_decision.get('color_mode', 'grayscale')}")
            print(f"  Memory stats: {memory_stats}")

        return {
            "_v9_act_decision": act_decision,
            "_v9_memory_stats": memory_stats,
            "_benchmark_stance": stance,
            "_color_mode": act_decision.get("color_mode", "grayscale"),
            # Also set v8-compat fields for shared helpers
            "_observe_decisions": {
                "object_type": object_type,
                "color_mode": act_decision.get("color_mode", "grayscale"),
                "filtration_type": "sublevel",
            },
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V9 Act: {descriptor} (stance={stance}), "
                f"params={act_decision.get('final_params', {})}"
            ],
        }

    def _v9_extract(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 5: EXTRACT -- deterministic descriptor execution.

        Reads descriptor decision from _v9_act_decision and executes it.
        Reuses the per-channel and grayscale logic from v8.
        """
        import numpy as np
        from PIL import Image as PILImage

        self._vprint_header("V9 EXTRACT")

        act_decision = state.get("_v9_act_decision") or {}
        descriptor_name = act_decision.get("final_descriptor", "persistence_statistics")
        descriptor_params = act_decision.get("final_params", {})
        color_mode = act_decision.get("color_mode", state.get("_color_mode", "grayscale"))

        # Clean params: remove non-tool keys
        clean_params = {k: v for k, v in descriptor_params.items()
                        if k not in ("dim", "total_dim", "classifier", "color_mode")}

        new_memory = list(state["short_term_memory"])
        feature_quality = None

        # Image-based vs PH-based
        IMAGE_BASED = {"minkowski_functionals", "euler_characteristic_curve",
                       "euler_characteristic_transform", "edge_histogram", "lbp_texture"}

        # Per-channel execution
        if color_mode == "per_channel" and descriptor_name in self.tools:
            try:
                rgb_img = np.array(PILImage.open(state["image_path"]).convert("RGB"),
                                   dtype=np.float32) / 255.0
                channels = {"R": rgb_img[:, :, 0], "G": rgb_img[:, :, 1], "B": rgb_img[:, :, 2]}

                is_image_based = descriptor_name in IMAGE_BASED
                channel_vectors = []

                # Get grayscale PH from short-term memory for PH-based descriptors
                grayscale_ph = None
                for name, output in state["short_term_memory"]:
                    if name == "compute_ph" and isinstance(output, dict) and output.get("success"):
                        grayscale_ph = output.get("persistence", {})
                        break

                for ch_name in ["R", "G", "B"]:
                    ch_args = clean_params.copy()
                    if is_image_based:
                        ch_args["image_array"] = channels[ch_name].tolist()
                    else:
                        # Compute PH on this channel
                        if "compute_ph" in self.tools:
                            ch_ph_result = self.tools["compute_ph"].invoke({
                                "image_array": channels[ch_name].tolist(),
                                "filtration_type": "sublevel",
                                "max_dimension": 1,
                            })
                            if ch_ph_result.get("success"):
                                ch_args["persistence_data"] = ch_ph_result.get("persistence", {})
                            else:
                                ch_args["persistence_data"] = grayscale_ph or {}
                        else:
                            ch_args["persistence_data"] = grayscale_ph or {}

                    ch_result = self.tools[descriptor_name].invoke(ch_args)
                    fv = ch_result.get("combined_vector") or ch_result.get("feature_vector")
                    if fv is not None:
                        channel_vectors.append(np.asarray(fv, dtype=np.float64))

                if channel_vectors:
                    concat_vector = np.concatenate(channel_vectors).tolist()
                    result = {
                        "success": True,
                        "combined_vector": concat_vector,
                        "vector_length": len(concat_vector),
                        "tool_name": descriptor_name,
                        "color_mode": "per_channel",
                        "per_channel_dims": [len(v) for v in channel_vectors],
                    }
                    for k, v in clean_params.items():
                        result[k] = v
                else:
                    result = {"success": False, "error": "No per-channel features extracted"}

                self._vprint_tool_output(descriptor_name, result, result.get("success", False))
                new_memory.append((descriptor_name, result))

                if result.get("success"):
                    arr = np.asarray(result["combined_vector"], dtype=np.float64)
                    feature_quality = self._compute_feature_quality(arr)

            except Exception as e:
                result = {"success": False, "error": str(e)}
                new_memory.append((descriptor_name, result))
                self._vprint_tool_output(descriptor_name, str(e), False)

        else:
            # Grayscale or single-channel extraction
            tool_args = clean_params.copy()
            tool_args = self._auto_inject_args(descriptor_name, tool_args,
                                               {**state, "short_term_memory": new_memory})

            if descriptor_name in self.tools:
                try:
                    result = self.tools[descriptor_name].invoke(tool_args)
                    self._vprint_tool_output(descriptor_name, result, result.get("success", False))
                    new_memory.append((descriptor_name, result))

                    if result.get("success", False):
                        fv = result.get("combined_vector") or result.get("feature_vector")
                        if fv is not None:
                            arr = np.asarray(fv, dtype=np.float64)
                            feature_quality = self._compute_feature_quality(arr)
                except Exception as e:
                    new_memory.append((descriptor_name, {"success": False, "error": str(e)}))
                    self._vprint_tool_output(descriptor_name, str(e), False)
            else:
                new_memory.append((descriptor_name, {"success": False, "error": f"Tool '{descriptor_name}' not found"}))

        updates = {
            "short_term_memory": new_memory,
        }
        if feature_quality is not None:
            updates["_feature_quality"] = feature_quality

        return updates

    def _v9_reflect(self, state: TopoAgentState) -> Dict[str, Any]:
        """Phase 6: REFLECT -- LLM call #4 (evaluate quality and learn).

        The LLM evaluates feature quality from raw statistics, decides COMPLETE
        or RETRY, and generates a V9ExperienceEntry for long-term memory.
        """
        import json as json_mod
        import time

        self._vprint_header("V9 REFLECT (LLM #4)")
        new_interactions = list(state.get("llm_interactions", []))

        ph_stats = state.get("_ph_stats") or {}
        feature_quality = state.get("_feature_quality") or {}
        interpret_output = state.get("_v9_interpret_output") or {}
        hypothesis = state.get("_v9_hypothesis") or {}
        act_decision = state.get("_v9_act_decision") or {}
        memory_stats = state.get("_v9_memory_stats") or {}

        descriptor_name = act_decision.get("final_descriptor", "unknown")
        descriptor_params = act_decision.get("final_params", {})
        # Use reconciled object type (same logic as ACT) — NOT raw INTERPRET guess
        from .skills.rules_data import OBJECT_TYPES
        reconciled_ot = hypothesis.get("object_type_reconciled", "")
        if reconciled_ot not in OBJECT_TYPES:
            reconciled_ot = ""
        object_type = (reconciled_ot
                       or interpret_output.get("object_type_guess", "unknown"))
        if object_type not in OBJECT_TYPES:
            object_type = "surface_lesions"
        color_mode = act_decision.get("color_mode", "grayscale")
        stance = state.get("_benchmark_stance", "CHOSE_ALTERNATIVE")
        act_reasoning = act_decision.get("reasoning", "")

        # Build decision summary for REFLECT prompt
        decision_summary = (
            f"Object type (guessed): {object_type}\n"
            f"Modality (guessed): {interpret_output.get('modality_guess', '?')}\n"
            f"Color mode: {color_mode}\n"
            f"Descriptor hypothesis (blind): {hypothesis.get('descriptor_hypothesis', '?')}\n"
            f"Hypothesis confidence: {hypothesis.get('confidence', '?')}\n"
            f"Reconciliation stance: {stance}\n"
            f"  Benchmark top: {act_decision.get('_benchmark_top', '?')}\n"
            f"  ACT reasoning: {act_reasoning[:200]}\n"
            f"Final descriptor: {descriptor_name}\n"
            f"Final params: {json_mod.dumps(descriptor_params, default=str)}\n"
            f"Memory influence: {memory_stats.get('memory_influenced', False)} "
            f"(top_sim={memory_stats.get('top_similarity', 0):.3f}, "
            f"retrieved={memory_stats.get('entries_retrieved', 0)}/"
            f"{memory_stats.get('entries_available', 0)})"
        )

        # Build feature statistics string
        fq = feature_quality
        feature_stats_lines = [
            f"- Dimension: {fq.get('dimension', 0)}",
            f"- Sparsity: {fq.get('sparsity', 0):.1f}% (fraction of near-zero values)",
            f"- Variance: {fq.get('variance', 0):.6f}",
            f"- Mean: {fq.get('mean', 0):.6f}, Std: {fq.get('std', 0):.6f}",
            f"- Min: {fq.get('min', 0):.6f}, Max: {fq.get('max', 0):.6f}",
            f"- Dynamic range: {fq.get('dynamic_range', 0):.6f}",
            f"- IQR (25th-75th): {fq.get('p25', 0):.6f} to {fq.get('p75', 0):.6f} (IQR={fq.get('iqr', 0):.6f})",
            f"- Kurtosis: {fq.get('kurtosis', 'N/A')}, Skewness: {fq.get('skewness', 'N/A')}",
            f"- NaN count: {fq.get('nan_count', 0)}, Inf count: {fq.get('inf_count', 0)}",
            f"- Informative features (>1% variance): {fq.get('n_informative_features', 'N/A')} / {fq.get('dimension', 0)}",
            f"- Constant features: {fq.get('n_constant_features', 0)}",
        ]
        feature_stats_text = "\n".join(feature_stats_lines)

        # Reference quality ranges (neutral, no verdict)
        reference_quality_ranges = build_reference_quality_ranges(descriptor_name)

        # Get relevant long-term memory for context
        relevant_memory = "No past experiences."
        if self.long_term_memory is not None:
            ph_metrics = {
                "h0_count": ph_stats.get("H0_count", 0),
                "h1_count": ph_stats.get("H1_count", 0),
                "h0_avg_pers": ph_stats.get("H0_avg_persistence", 0.0),
                "h1_avg_pers": ph_stats.get("H1_avg_persistence", 0.0),
            }
            relevant_memory = self.long_term_memory.format_for_v9_prompt(
                object_type=object_type,
                ph_metrics=ph_metrics,
                n=3,
            )

        # Retry context
        retry_context = ""
        retry_feedback = state.get("_retry_feedback")
        if retry_feedback:
            retry_context = f"## Previous Retry Feedback\n{retry_feedback}"

        prompt = V9_REFLECT_PROMPT.format(
            decision_summary=decision_summary,
            feature_stats=feature_stats_text,
            reference_quality_ranges=reference_quality_ranges,
            relevant_memory=relevant_memory,
            retry_context=retry_context,
        )

        self._vprint_prompt(prompt, max_chars=3000)

        # Call LLM (no tools)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content if hasattr(response, "content") else str(response)
        self._vprint_response(response_text)

        # Parse reflect JSON
        reflect = self._parse_json_response(response_text, defaults={
            "quality_assessment": {
                "checks_performed": [],
                "overall_quality": "acceptable",
                "reasoning": "",
            },
            "decision": "COMPLETE",
            "retry_feedback": "",
            "experience_entry": {
                "object_type": object_type,
                "descriptor": descriptor_name,
                "image_metrics": {},
                "ph_metrics": {},
                "feature_quality": {},
                "quality_verdict": "acceptable",
                "lesson": "",
                "would_choose_again": True,
            },
        })

        # Log interaction
        interaction = LLMInteraction(
            step="v9_reflect",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions.append(interaction)

        decision = reflect.get("decision", "COMPLETE").upper()

        if self.verbose:
            qa = reflect.get("quality_assessment", {})
            print(f"\n  Decision: {decision}")
            print(f"  Quality: {qa.get('overall_quality', '?')}")
            print(f"  Reasoning: {qa.get('reasoning', '')[:120]}")

        # Build reflect record for history
        reflect_record = {
            "round": len(state.get("_reflect_history", [])) + 1,
            "descriptor": descriptor_name,
            "descriptor_params": descriptor_params,
            "feature_quality": feature_quality,
            "quality_ok": decision == "COMPLETE",
            "decision": decision,
            "reasoning": reflect.get("quality_assessment", {}).get("reasoning", ""),
        }

        updates = {
            "_v9_reflect_output": reflect,
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"V9 Reflect: {decision} - {reflect.get('quality_assessment', {}).get('overall_quality', '?')}"
            ],
            "_reflect_history": list(state.get("_reflect_history", [])) + [reflect_record],
        }

        elapsed = time.time() - getattr(self, "_workflow_start_time", time.time())

        if decision == "RETRY" and state.get("_retry_count", 0) < 2 and not self._is_time_exceeded():
            # RETRY: store feedback, blacklist descriptor, route back to ACT
            retry_count = state.get("_retry_count", 0) + 1
            failed_so_far = list(state.get("_failed_descriptors", []))
            failed_so_far.append(descriptor_name)

            # Clean descriptor from memory for re-extraction
            cleaned_memory = [
                (name, out) for name, out in state["short_term_memory"]
                if name not in SUPPORTED_DESCRIPTORS
            ]

            feedback = reflect.get("retry_feedback", "")
            if not feedback:
                feedback = reflect.get("quality_assessment", {}).get("reasoning", "Quality check failed")
            failed_list = ", ".join(failed_so_far)

            updates["_retry_count"] = retry_count
            updates["_failed_descriptors"] = failed_so_far
            updates["short_term_memory"] = cleaned_memory
            updates["_feature_quality"] = None
            updates["_retry_feedback"] = (
                f"RETRY {retry_count}/2. "
                f"Latest failure: '{descriptor_name}' -- {feedback}. "
                f"Previously failed descriptors: [{failed_list}]. "
                f"Do NOT use any of these again."
            )
            updates["_reflect_decision"] = "RETRY"

            if self.verbose:
                print(f"  [RETRY] Blacklisted: {failed_so_far}, routing back to ACT")
        else:
            # COMPLETE: build final AgentReport + write V9ExperienceEntry to LTM
            n_tools = len(state["short_term_memory"])
            n_llm_calls = len(new_interactions)

            agent_report = AgentReport(
                descriptor=descriptor_name,
                object_type=object_type,
                reasoning_chain=hypothesis.get("descriptor_reasoning", ""),
                image_analysis=interpret_output.get("image_profile", ""),
                descriptor_rationale=hypothesis.get("descriptor_reasoning", ""),
                alternatives_considered=", ".join(hypothesis.get("alternatives", [])),
                parameters=descriptor_params,
                feature_dimension=feature_quality.get("dimension", 0),
                color_mode=color_mode,
                actual_dimension=feature_quality.get("dimension", 0),
                sparsity_pct=feature_quality.get("sparsity", 0.0),
                variance=feature_quality.get("variance", 0.0),
                nan_count=feature_quality.get("nan_count", 0),
                quality_ok=decision == "COMPLETE",
                error_analysis=reflect.get("quality_assessment", {}).get("reasoning", ""),
                experience=reflect.get("experience_entry", {}).get("lesson", ""),
                observe_object_type=object_type,
                observe_color_mode=color_mode,
                observe_filtration="sublevel",
                observe_reasoning=interpret_output.get("object_type_evidence", ""),
                object_type_correct=interpret_output.get("_object_type_correct"),
                benchmark_stance=stance,
                ph_interpretation=interpret_output.get("ph_profile", ""),
                descriptor_intuition=hypothesis.get("descriptor_hypothesis", ""),
                n_llm_calls=n_llm_calls,
                n_tools_executed=n_tools,
                retry_used=state.get("_retry_count", 0) > 0,
                total_time_seconds=round(elapsed, 2),
            )

            report = agent_report.to_dict()
            quality_ok = decision == "COMPLETE"
            confidence = 85.0 if quality_ok else 50.0

            updates["final_answer"] = report
            updates["confidence"] = confidence
            updates["evidence"] = [str(item) for item in state["short_term_memory"]]
            updates["task_complete"] = True
            updates["_reflect_decision"] = "COMPLETE"

            # Write V9ExperienceEntry to long-term memory
            experience_entry = reflect.get("experience_entry", {})
            if experience_entry and self.long_term_memory is not None:
                lesson = experience_entry.get("lesson", "")
                if lesson:
                    from .memory.long_term import V9ExperienceEntry

                    # Always use actual state values for structured metrics
                    # (LLM often returns template zeros instead of real values)
                    img_analysis = state.get("_v8_image_analysis") or {}
                    img_metrics = img_analysis.get("image_statistics", {})
                    actual_image_metrics = {
                        "snr": img_metrics.get("snr_estimate", 0.0),
                        "contrast": img_metrics.get("contrast", 0.0),
                        "edge_density": img_metrics.get("edge_density", 0.0),
                    }
                    actual_ph_metrics = {
                        "h0_count": ph_stats.get("H0_count", 0),
                        "h1_count": ph_stats.get("H1_count", 0),
                        "h0_avg_pers": ph_stats.get("H0_avg_persistence", 0.0),
                        "h1_avg_pers": ph_stats.get("H1_avg_persistence", 0.0),
                    }
                    actual_feature_quality = {
                        "sparsity": feature_quality.get("sparsity", 0.0),
                        "variance": feature_quality.get("variance", 0.0),
                        "dimension": feature_quality.get("dimension", 0),
                    }

                    v9_entry = V9ExperienceEntry(
                        object_type=experience_entry.get("object_type", object_type),
                        descriptor=experience_entry.get("descriptor", descriptor_name),
                        image_metrics=actual_image_metrics,
                        ph_metrics=actual_ph_metrics,
                        feature_quality=actual_feature_quality,
                        quality_verdict=experience_entry.get("quality_verdict", "acceptable"),
                        lesson=lesson,
                        would_choose_again=experience_entry.get("would_choose_again", True),
                        stance=stance,
                        descriptor_params=descriptor_params,
                    )
                    self.long_term_memory.add_v9(v9_entry)
                    if self.verbose:
                        print(f"  [LTM] Wrote V9 experience: {lesson[:80]}")

        return updates

    def _v9_route_reflect(self, state: TopoAgentState) -> Literal["complete", "retry"]:
        """Route after v9 REFLECT: complete or retry back to ACT."""
        decision = state.get("_reflect_decision", "COMPLETE").upper()
        if self._is_time_exceeded() and decision != "COMPLETE":
            if self.verbose:
                print("  [TIME LIMIT] Forcing COMPLETE")
            return "complete"
        if decision == "RETRY" and state.get("_retry_count", 0) < 2:
            return "retry"
        return "complete"

    def _parse_json_response(self, text: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a JSON response from LLM, with fallback to defaults."""
        import json as json_mod

        # Try direct parse
        try:
            parsed = json_mod.loads(text.strip())
            for k, v in defaults.items():
                if k not in parsed:
                    parsed[k] = v
            return parsed
        except json_mod.JSONDecodeError:
            pass

        # Try markdown code block
        for marker in ["```json", "```"]:
            if marker in text:
                try:
                    start = text.index(marker) + len(marker)
                    end = text.index("```", start)
                    parsed = json_mod.loads(text[start:end].strip())
                    for k, v in defaults.items():
                        if k not in parsed:
                            parsed[k] = v
                    return parsed
                except (json_mod.JSONDecodeError, ValueError):
                    pass

        # Try finding JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                parsed = json_mod.loads(text[brace_start:brace_end + 1])
                for k, v in defaults.items():
                    if k not in parsed:
                        parsed[k] = v
                return parsed
            except json_mod.JSONDecodeError:
                pass

        if self.verbose:
            print(f"  [WARN] Could not parse JSON response, using defaults")
        return dict(defaults)

    # =========================================================================
    # Verbose Logging Helpers
    # =========================================================================

    def _vprint_header(self, title: str, round_num: int = None):
        """Print a formatted section header."""
        if not self.verbose:
            return
        width = 70
        if round_num is not None:
            label = f" ROUND {round_num}/{self.max_rounds} — {title} "
        else:
            label = f" {title} "
        print(f"\n{'═' * width}")
        print(f"║{label:^{width - 2}}║")
        print(f"{'═' * width}")

    def _vprint_prompt(self, prompt: str, max_chars: int = 2000):
        """Print a truncated prompt."""
        if not self.verbose:
            return
        text = prompt[:max_chars]
        if len(prompt) > max_chars:
            text += f"\n  ...[truncated, {len(prompt)} chars total]"
        print(f"\n📋 PROMPT ({len(prompt)} chars):")
        for line in text.split("\n"):
            print(f"  {line}")

    def _vprint_response(self, response_text: str, max_chars: int = 1500):
        """Print an LLM response."""
        if not self.verbose:
            return
        text = response_text[:max_chars]
        if len(response_text) > max_chars:
            text += f"\n  ...[truncated, {len(response_text)} chars total]"
        print(f"\n🤖 LLM RESPONSE ({len(response_text)} chars):")
        for line in text.split("\n"):
            print(f"  {line}")

    def _vprint_tool_calls(self, tool_calls: list):
        """Print tool calls."""
        if not self.verbose or not tool_calls:
            return
        print(f"\n🔧 TOOL CALLS:")
        for i, tc in enumerate(tool_calls, 1):
            name = tc.get("name", "?")
            args = tc.get("args", {})
            # Summarize large args (arrays, persistence data)
            summarized = {}
            for k, v in args.items():
                if isinstance(v, (list, dict)) and len(str(v)) > 200:
                    summarized[k] = f"<{type(v).__name__}, len={len(v)}>"
                elif hasattr(v, 'shape'):
                    summarized[k] = f"<ndarray, shape={v.shape}>"
                else:
                    summarized[k] = v
            print(f"  {i}. {name}({', '.join(f'{k}={v}' for k, v in summarized.items())})")

    def _vprint_tool_output(self, tool_name: str, output: Any, success: bool):
        """Print a summarized tool output."""
        if not self.verbose:
            return
        import numpy as np
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n── TOOL: {tool_name} — {status} ──")

        if isinstance(output, dict):
            for k, v in output.items():
                if k in ("persistence", "persistence_data") and isinstance(v, dict):
                    # Summarize PH data
                    for hk, hv in v.items():
                        count = len(hv) if isinstance(hv, (list, np.ndarray)) else "?"
                        print(f"  {hk}: {count} features")
                elif k in ("image_array",) and hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                elif k in ("combined_vector", "feature_vector") and isinstance(v, (list, np.ndarray)):
                    arr = np.asarray(v) if isinstance(v, list) else v
                    print(f"  {k}: dim={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}")
                elif k == "statistics" and isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            count = sv.get("count", "?")
                            avg_pers = sv.get("mean_persistence", sv.get("avg_persistence", "?"))
                            print(f"  {sk}: count={count}, avg_persistence={avg_pers}")
                        else:
                            print(f"  {sk}: {sv}")
                elif k == "success":
                    continue
                else:
                    val_str = str(v)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    print(f"  {k}: {val_str}")
        else:
            val_str = str(output)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            print(f"  {val_str}")

    def _vprint_reflection(self, reflection):
        """Print reflection details."""
        if not self.verbose or reflection is None:
            return
        print(f"\n💭 REFLECTION (round {reflection.round}):")
        print(f"  Error Analysis: {reflection.error_analysis}")
        print(f"  Suggestion: {reflection.suggestion}")
        print(f"  Experience: {reflection.experience}")

    def _vprint_separator(self):
        """Print a section separator."""
        if not self.verbose:
            return
        print(f"{'─' * 70}")

    # =========================================================================
    # Optimized Skills Pipeline (v4.1): 4 nodes, 2 LLM calls
    # =========================================================================

    def _build_skills_workflow(self) -> Any:
        """Build the optimized 4-node skills workflow.

        Architecture (2 LLM calls total):
            pre_execute (0 LLM) → plan_descriptor (1 LLM) →
            execute_descriptor (0 LLM) → verify_and_reflect (1 LLM) → END

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(TopoAgentState)

        workflow.add_node("pre_execute", self._skills_pre_execute)
        workflow.add_node("plan_descriptor", self._skills_plan_descriptor)
        workflow.add_node("execute_descriptor", self._skills_execute_descriptor)
        workflow.add_node("verify_and_reflect", self._skills_verify_and_reflect)

        workflow.add_node("retry_descriptor", self._skills_retry_descriptor)

        workflow.set_entry_point("pre_execute")
        workflow.add_edge("pre_execute", "plan_descriptor")
        workflow.add_edge("plan_descriptor", "execute_descriptor")

        # Conditional: retry with alternative if execution fails
        workflow.add_conditional_edges(
            "execute_descriptor",
            self._skills_should_retry,
            {
                "retry": "execute_descriptor",
                "continue": "verify_and_reflect",
            },
        )

        # Conditional: retry after verification if quality is poor (max 1 retry)
        workflow.add_conditional_edges(
            "verify_and_reflect",
            self._skills_should_retry_after_verify,
            {
                "retry": "retry_descriptor",
                "done": END,
            },
        )
        workflow.add_edge("retry_descriptor", "execute_descriptor")

        return workflow.compile()

    def _skills_pre_execute(self, state: TopoAgentState) -> Dict[str, Any]:
        """Run image_loader + compute_ph deterministically (0 LLM calls).

        Extracts structured PH statistics and determines color mode.
        """
        import numpy as np
        import time

        self._workflow_start_time = time.time()
        self._vprint_header("PRE-EXECUTE (deterministic)")

        new_memory = list(state["short_term_memory"])
        trace = list(state["reasoning_trace"])
        ph_stats = {}

        # Step 1: Load image
        if "image_loader" in self.tools:
            try:
                img_result = self.tools["image_loader"].invoke({
                    "image_path": state["image_path"],
                    "normalize": True,
                    "grayscale": True,
                })
                new_memory.append(("image_loader", img_result))
                self._vprint_tool_output("image_loader", img_result, img_result.get("success", False))

                if img_result.get("success", False):
                    trace.append(f"Loaded image: shape={img_result.get('shape')}")
                else:
                    trace.append(f"image_loader failed: {img_result.get('error')}")
            except Exception as e:
                trace.append(f"image_loader exception: {e}")
                new_memory.append(("image_loader", {"success": False, "error": str(e)}))

        # Step 2: Compute PH
        image_array = get_image_array({"short_term_memory": new_memory})
        if image_array is not None and "compute_ph" in self.tools:
            try:
                ph_result = self.tools["compute_ph"].invoke({
                    "image_array": image_array,
                    "filtration_type": "sublevel",
                    "max_dimension": 1,
                })
                new_memory.append(("compute_ph", ph_result))
                self._vprint_tool_output("compute_ph", ph_result, ph_result.get("success", False))

                if ph_result.get("success", False):
                    stats = ph_result.get("statistics", {})
                    persistence_data = ph_result.get("persistence", {})

                    # Extract structured PH statistics
                    h0_pairs = persistence_data.get("H0", [])
                    h1_pairs = persistence_data.get("H1", [])

                    h0_count = stats.get("H0", {}).get("count", 0)
                    h1_count = stats.get("H1", {}).get("count", 0)

                    # Extract persistence values — handles both dict and array formats
                    def _extract_persistences(pairs):
                        """Extract persistence values from PH pairs (dict or array format)."""
                        persistences = []
                        if pairs is None or not hasattr(pairs, '__len__') or len(pairs) == 0:
                            return persistences
                        # Check first element to determine format
                        first = pairs[0] if len(pairs) > 0 else None
                        if isinstance(first, dict):
                            # Dict format: {'birth': ..., 'death': ..., 'persistence': ...}
                            for p in pairs:
                                pers = p.get("persistence")
                                if pers is not None and np.isfinite(pers):
                                    persistences.append(float(pers))
                        else:
                            # Numpy array format: [[birth, death], ...]
                            arr = np.asarray(pairs)
                            if arr.ndim == 2 and arr.shape[1] >= 2:
                                finite_mask = np.isfinite(arr[:, 1])
                                finite_pairs = arr[finite_mask]
                                if len(finite_pairs) > 0:
                                    persistences = (finite_pairs[:, 1] - finite_pairs[:, 0]).tolist()
                        return persistences

                    h0_persistences = _extract_persistences(h0_pairs)
                    h1_persistences = _extract_persistences(h1_pairs)

                    ph_stats = {
                        "H0_count": h0_count,
                        "H1_count": h1_count,
                        "H0_max_persistence": max(h0_persistences) if h0_persistences else 0.0,
                        "H1_max_persistence": max(h1_persistences) if h1_persistences else 0.0,
                        "H0_mean_persistence": float(np.mean(h0_persistences)) if h0_persistences else 0.0,
                        "H1_mean_persistence": float(np.mean(h1_persistences)) if h1_persistences else 0.0,
                        "total_persistence": sum(h0_persistences) + sum(h1_persistences),
                        "filtration": "sublevel",
                    }
                    trace.append(f"Computed PH: H0={h0_count}, H1={h1_count}")
                else:
                    trace.append(f"compute_ph failed: {ph_result.get('error')}")
            except Exception as e:
                trace.append(f"compute_ph exception: {e}")
                new_memory.append(("compute_ph", {"success": False, "error": str(e)}))

        # Step 3: Determine color mode
        color_mode = self.skill_registry.select_color_mode(
            image_path=state["image_path"],
        )

        if self.verbose and ph_stats:
            print(f"\n  PH Stats: H0={ph_stats.get('H0_count')}, H1={ph_stats.get('H1_count')}, "
                  f"H0_max_pers={ph_stats.get('H0_max_persistence', 0):.4f}, "
                  f"H1_max_pers={ph_stats.get('H1_max_persistence', 0):.4f}")
            print(f"  Color mode: {color_mode}")

        return {
            "short_term_memory": new_memory,
            "reasoning_trace": trace,
            "_ph_stats": ph_stats,
            "skill_color_mode": color_mode,
            "current_round": 1,
        }

    def _skills_plan_descriptor(self, state: TopoAgentState) -> Dict[str, Any]:
        """LLM selects descriptor with structured JSON reasoning (1 LLM call)."""
        import json as json_mod

        self._vprint_header("PLAN DESCRIPTOR (LLM #1)")

        ph_stats = state.get("_ph_stats") or {}
        color_mode = state.get("skill_color_mode") or "grayscale"

        # Infer object type hint for benchmark rankings
        object_type_hint = self.skill_registry.infer_object_type_hint(
            query=state["query"],
            image_path=state["image_path"],
        )

        # Build full knowledge context
        skill_knowledge = self.skill_registry.build_skill_context(
            object_type_hint=object_type_hint,
            color_mode=color_mode,
        )

        # Format prompt
        prompt = SKILLS_PLAN_PROMPT.format(
            query=state["query"],
            image_path=state["image_path"],
            ph_stats=json_mod.dumps(ph_stats, indent=2) if ph_stats else "PH computation not available",
            color_mode=color_mode,
            skill_knowledge=skill_knowledge,
        )

        self._vprint_prompt(prompt)

        # Call LLM (plain text, no tool binding)
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        self._vprint_response(response_text)

        # Parse structured JSON response
        plan = self._parse_plan_json(response_text)

        # Validate plan
        plan = self._validate_skills_plan(plan, object_type_hint, color_mode)

        if self.verbose:
            print(f"\n  Plan: descriptor={plan.get('descriptor_choice')}, "
                  f"object_type={plan.get('object_type')}, "
                  f"reasoning={plan.get('reasoning_chain')}")

        # Configure params from benchmark rules
        object_type = plan.get("object_type", object_type_hint or "surface_lesions")
        descriptor = plan["descriptor_choice"]
        params = self.skill_registry.configure_after_selection(
            descriptor=descriptor,
            object_type=object_type,
            color_mode=plan.get("color_mode", color_mode),
        )

        # Log LLM interaction
        interaction = LLMInteraction(
            step="plan_descriptor",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions = list(state.get("llm_interactions", []))
        new_interactions.append(interaction)

        return {
            "_plan": plan,
            "skill_descriptor": descriptor,
            "skill_params": params,
            "skill_color_mode": plan.get("color_mode", color_mode),
            "llm_interactions": new_interactions,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Plan: selected {descriptor} for {object_type} ({plan.get('reasoning_chain', 'unknown')})"
            ],
        }

    def _parse_plan_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        import json as json_mod

        # Try direct parse
        try:
            return json_mod.loads(text.strip())
        except json_mod.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end_idx = text.index("```", start)
                try:
                    return json_mod.loads(text[start:end_idx].strip())
                except (json_mod.JSONDecodeError, ValueError):
                    pass

        # Try finding JSON object in text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json_mod.loads(text[brace_start:brace_end + 1])
            except json_mod.JSONDecodeError:
                pass

        # Fallback
        return {
            "object_type": "surface_lesions",
            "reasoning_chain": "h0_dominant",
            "image_analysis": "Could not parse LLM response. Using default.",
            "descriptor_choice": "template_functions",
            "descriptor_rationale": "Default fallback.",
            "needs_ph": True,
            "alternative_descriptor": "betti_curves",
            "alternative_rationale": "Default alternative.",
            "expected_feature_dim": 50,
            "color_mode": "grayscale",
        }

    def _validate_skills_plan(
        self,
        plan: Dict[str, Any],
        object_type_hint: Optional[str],
        color_mode: str,
    ) -> Dict[str, Any]:
        """Validate and fix plan fields."""
        # Ensure descriptor_choice is valid
        if plan.get("descriptor_choice") not in SUPPORTED_DESCRIPTORS:
            plan["descriptor_choice"] = "template_functions"

        # Ensure alternative is valid and different from primary
        alt = plan.get("alternative_descriptor")
        if alt not in SUPPORTED_DESCRIPTORS or alt == plan["descriptor_choice"]:
            ot = plan.get("object_type", object_type_hint or "surface_lesions")
            performers = TOP_PERFORMERS.get(ot, [])
            for p in performers:
                if p["descriptor"] != plan["descriptor_choice"]:
                    plan["alternative_descriptor"] = p["descriptor"]
                    break
            else:
                plan["alternative_descriptor"] = "betti_curves"

        # Fix needs_ph based on descriptor type
        plan["needs_ph"] = plan["descriptor_choice"] not in IMAGE_BASED_DESCRIPTORS

        # Fix color_mode if not valid
        if plan.get("color_mode") not in ("grayscale", "per_channel"):
            plan["color_mode"] = color_mode

        # Override expected_feature_dim with actual value from benchmark rules
        # (LLM often hallucinates the dimension, causing false mismatch in verification)
        # Always use "grayscale" because the agent pipeline computes PH on a single
        # grayscale image — per_channel is a benchmark-only mode not supported by the tools.
        ot = plan.get("object_type", object_type_hint or "surface_lesions")
        actual_dim = get_descriptor_dim(plan["descriptor_choice"], ot, "grayscale")
        if actual_dim > 0:
            plan["expected_feature_dim"] = actual_dim

        return plan

    def _skills_execute_descriptor(self, state: TopoAgentState) -> Dict[str, Any]:
        """Run chosen descriptor with auto-injected params (0 LLM calls).

        On failure, sets _needs_retry and swaps to alternative_descriptor.
        """
        import numpy as np

        plan = state.get("_plan") or {}
        descriptor = state.get("skill_descriptor")
        skill_params = state.get("skill_params") or {}
        retry_count = state.get("_retry_count", 0)

        # Normalize descriptor name case if needed
        if descriptor and descriptor not in self.tools:
            # Try case-insensitive match
            for tname in self.tools:
                if tname.lower() == descriptor.lower():
                    descriptor = tname
                    break

        self._vprint_header(f"EXECUTE DESCRIPTOR: {descriptor}")

        if not descriptor or descriptor not in self.tools:
            return {
                "_needs_retry": False,
                "_feature_quality": {"error": f"Descriptor '{descriptor}' not available"},
                "reasoning_trace": state["reasoning_trace"] + [
                    f"Descriptor '{descriptor}' not found in tools"
                ],
            }

        # Build tool args
        tool_args = {}
        if descriptor in IMAGE_BASED_DESCRIPTORS:
            image_array = get_image_array(state)
            if image_array is not None:
                tool_args["image_array"] = image_array
        else:
            persistence_data = get_persistence_data(state)
            if persistence_data is not None:
                tool_args["persistence_data"] = persistence_data
            else:
                # PH data not available — try alternative
                if retry_count < 1:
                    alt = plan.get("alternative_descriptor", "betti_curves")
                    if self.verbose:
                        print(f"  No persistence data, swapping to alternative: {alt}")
                    alt_params = self.skill_registry.configure_after_selection(
                        descriptor=alt,
                        object_type=plan.get("object_type", "surface_lesions"),
                        color_mode=state.get("skill_color_mode") or "grayscale",
                    )
                    return {
                        "skill_descriptor": alt,
                        "skill_params": alt_params,
                        "_needs_retry": True,
                        "_retry_count": retry_count + 1,
                        "reasoning_trace": state["reasoning_trace"] + [
                            f"No persistence data for {descriptor}, retrying with {alt}"
                        ],
                    }
                return {
                    "_needs_retry": False,
                    "_feature_quality": {"error": "No persistence data available"},
                    "reasoning_trace": state["reasoning_trace"] + [
                        "No persistence data and no retries left"
                    ],
                }

        # Inject benchmark params
        for key, value in skill_params.items():
            if key in ("total_dim", "classifier", "color_mode", "dim"):
                continue
            if key not in tool_args:
                tool_args[key] = value

        # Execute
        try:
            result = self.tools[descriptor].invoke(tool_args)
            self._vprint_tool_output(descriptor, result, result.get("success", False))
        except Exception as e:
            result = {"success": False, "error": str(e)}
            self._vprint_tool_output(descriptor, str(e), False)

        new_memory = list(state["short_term_memory"])
        new_memory.append((descriptor, result))

        if result.get("success", False):
            # Extract feature vector
            fv = result.get("combined_vector") or result.get("feature_vector")
            if fv is not None:
                arr = np.asarray(fv, dtype=np.float64)
                # Compute feature quality metrics
                feature_quality = {
                    "dimension": len(arr),
                    "sparsity": float((np.abs(arr) < 1e-10).mean()) * 100,
                    "variance": float(np.var(arr)),
                    "nan_count": int(np.isnan(arr).sum()),
                }
                if self.verbose:
                    print(f"\n  Feature quality: dim={feature_quality['dimension']}, "
                          f"sparsity={feature_quality['sparsity']:.1f}%, "
                          f"variance={feature_quality['variance']:.6f}, "
                          f"NaN={feature_quality['nan_count']}")
                return {
                    "short_term_memory": new_memory,
                    "_feature_quality": feature_quality,
                    "_needs_retry": False,
                    "reasoning_trace": state["reasoning_trace"] + [
                        f"Executed {descriptor}: dim={feature_quality['dimension']}"
                    ],
                }
            else:
                # Success but no feature vector — shouldn't happen
                return {
                    "short_term_memory": new_memory,
                    "_feature_quality": {"dimension": 0, "error": "No feature vector in output"},
                    "_needs_retry": False,
                    "reasoning_trace": state["reasoning_trace"] + [
                        f"{descriptor} succeeded but returned no feature vector"
                    ],
                }
        else:
            # Execution failed — retry with alternative if available
            if retry_count < 1:
                alt = plan.get("alternative_descriptor", "betti_curves")
                if self.verbose:
                    print(f"  Execution failed, swapping to alternative: {alt}")
                alt_params = self.skill_registry.configure_after_selection(
                    descriptor=alt,
                    object_type=plan.get("object_type", "surface_lesions"),
                    color_mode=state.get("skill_color_mode") or "grayscale",
                )
                return {
                    "short_term_memory": new_memory,
                    "skill_descriptor": alt,
                    "skill_params": alt_params,
                    "_needs_retry": True,
                    "_retry_count": retry_count + 1,
                    "reasoning_trace": state["reasoning_trace"] + [
                        f"{descriptor} failed: {result.get('error')}, retrying with {alt}"
                    ],
                }
            return {
                "short_term_memory": new_memory,
                "_feature_quality": {"error": result.get("error", "Unknown error")},
                "_needs_retry": False,
                "reasoning_trace": state["reasoning_trace"] + [
                    f"{descriptor} failed and no retries left"
                ],
            }

    def _skills_should_retry(self, state: TopoAgentState) -> Literal["retry", "continue"]:
        """Check if descriptor execution needs retry with alternative."""
        if state.get("_needs_retry", False) and state.get("_retry_count", 0) <= 1:
            return "retry"
        return "continue"

    def _skills_should_retry_after_verify(self, state: TopoAgentState) -> Literal["retry", "done"]:
        """Check if verification flagged quality issues requiring a retry.

        Only allows 1 verification-triggered retry (so max 3 LLM calls total:
        plan + verify + verify-after-retry).
        """
        # Already retried once — don't retry again
        if state.get("_retry_count", 0) >= 1:
            return "done"

        # Check the verification result
        final_answer = state.get("final_answer")
        if not isinstance(final_answer, dict):
            return "done"

        verification = final_answer.get("verification", {})
        quality_ok = verification.get("quality_ok", True)
        suggestion = verification.get("suggestion", "COMPLETE")

        # Trust the LLM's judgment — if it says RETRY, retry
        if not quality_ok and isinstance(suggestion, str) and suggestion.upper().startswith("RETRY"):
            if self.verbose:
                feature_quality = state.get("_feature_quality") or {}
                print(f"\n  RETRY TRIGGERED: quality_ok=False, suggestion={suggestion}, "
                      f"sparsity={feature_quality.get('sparsity', 0):.1f}%")
            return "retry"

        return "done"

    def _skills_retry_descriptor(self, state: TopoAgentState) -> Dict[str, Any]:
        """Swap to alternative descriptor and prepare for re-execution.

        Called when verify_and_reflect flags poor quality. Swaps to the
        alternative_descriptor from the original plan and increments retry count.
        """
        plan = state.get("_plan") or {}
        alt = plan.get("alternative_descriptor", "betti_curves")
        object_type = plan.get("object_type", "surface_lesions")
        color_mode = state.get("skill_color_mode") or "grayscale"

        if self.verbose:
            print(f"\n  🔄 RETRY: Swapping from {state.get('skill_descriptor')} to {alt}")

        # Configure params for the alternative descriptor
        alt_params = self.skill_registry.configure_after_selection(
            descriptor=alt,
            object_type=object_type,
            color_mode=color_mode,
        )

        return {
            "skill_descriptor": alt,
            "skill_params": alt_params,
            "_needs_retry": True,
            "_retry_count": state.get("_retry_count", 0) + 1,
            # Reset feature quality so verify_and_reflect re-evaluates
            "_feature_quality": None,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Verification retry: swapping to {alt} due to quality issues"
            ],
        }

    # =========================================================================
    # Reference Stats Helpers (for Fix 2B: Improved Reflection Quality)
    # =========================================================================

    # PH count ranges for 224×224 grayscale images (sublevel filtration).
    # Counts vary widely with image content — these are broad reference ranges.
    PH_REFERENCE_PATTERNS = {
        "discrete_cells": {
            "H0_range": (200, 2500), "H1_range": (100, 2000),
            "description": "Many small discrete components. H0 typically > H1.",
        },
        "glands_lumens": {
            "H0_range": (200, 3000), "H1_range": (200, 4000),
            "description": "Glandular tissue with lumens. Rich PH with many H1 loops. H1 often > H0.",
        },
        "vessel_trees": {
            "H0_range": (100, 2000), "H1_range": (50, 2000),
            "description": "Branching vascular structures. Variable H0/H1 depending on vessel density.",
        },
        "surface_lesions": {
            "H0_range": (200, 3000), "H1_range": (200, 4000),
            "description": "Skin/surface lesions. Rich texture yields high PH counts.",
        },
        "organ_shape": {
            "H0_range": (100, 1500), "H1_range": (50, 1200),
            "description": "Organ silhouettes. Shape features dominate. Moderate PH complexity.",
        },
    }

    # Expected sparsity range per descriptor. Dense descriptors naturally have ~0% sparsity.
    DESCRIPTOR_SPARSITY = {
        # Dense by design (statistical summaries, histograms, direct image features)
        "persistence_statistics": (0, 5, "Dense descriptor (62 statistical summaries). 0% sparsity is normal."),
        "lbp_texture": (0, 5, "Dense histogram descriptor. 0% sparsity is normal."),
        "edge_histogram": (0, 10, "Dense histogram descriptor. Low sparsity is normal."),
        "minkowski_functionals": (0, 10, "Dense geometric descriptor. Low sparsity is normal."),
        "euler_characteristic_curve": (0, 15, "Dense curve descriptor. Low sparsity is normal."),
        "euler_characteristic_transform": (0, 20, "Dense transform descriptor. Low sparsity is normal."),
        # Moderate sparsity (vectorized PH diagrams)
        "persistence_image": (5, 50, "Gaussian-smoothed grid. Moderate sparsity from empty diagram regions."),
        "betti_curves": (10, 60, "Histogram bins. Sparsity depends on filtration range coverage."),
        "persistence_silhouette": (10, 60, "Weighted landscape summary. Moderate sparsity typical."),
        "persistence_entropy": (15, 70, "Entropy curve. Higher sparsity from zero-entropy regions."),
        "tropical_coordinates": (20, 70, "Algebraic coordinates. Higher sparsity common."),
        "persistence_codebook": (10, 50, "Codebook histogram. Moderate sparsity."),
        "ATOL": (5, 40, "Center-based summary. Low-moderate sparsity."),
        "template_functions": (15, 60, "Template evaluations. Moderate sparsity from zero-persistence regions."),
        # High sparsity (high-dimensional representations)
        "persistence_landscapes": (20, 80, "Multi-layer landscapes. Higher sparsity from inactive layers."),
    }

    @staticmethod
    def _extract_persistence_values(pairs) -> list:
        """Extract persistence values from PH pairs (dict or array format)."""
        import numpy as np
        if pairs is None or not hasattr(pairs, '__len__') or len(pairs) == 0:
            return []
        first = pairs[0] if len(pairs) > 0 else None
        if isinstance(first, dict):
            return [float(p["persistence"]) for p in pairs
                    if p.get("persistence") is not None and np.isfinite(p["persistence"])]
        arr = np.asarray(pairs)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            finite = arr[np.isfinite(arr[:, 1])]
            if len(finite) > 0:
                return (finite[:, 1] - finite[:, 0]).tolist()
        return []

    @staticmethod
    def _fmt_pers(val: float) -> str:
        """Format persistence: scientific notation for very small values."""
        if val == 0:
            return "0"
        elif abs(val) < 0.001:
            return f"{val:.4e}"
        else:
            return f"{val:.6f}"

    def _build_ph_summary(self, state) -> str:
        """Build a formatted PH summary from short_term_memory.

        Uses scientific notation for very small persistence values to avoid
        showing '0.000000' for values like 1.2e-5.
        """
        for tool_name, output in reversed(state["short_term_memory"]):
            if tool_name == "compute_ph" and isinstance(output, dict):
                stats = output.get("statistics", {})
                h0 = stats.get("H0", {})
                h1 = stats.get("H1", {})
                filt = output.get("filtration_type", "sublevel")
                return (
                    f"H0: {h0.get('count', 0)} features, "
                    f"max_persistence={self._fmt_pers(h0.get('max_persistence', 0))}, "
                    f"avg_persistence={self._fmt_pers(h0.get('mean_persistence', 0))}\n"
                    f"H1: {h1.get('count', 0)} features, "
                    f"max_persistence={self._fmt_pers(h1.get('max_persistence', 0))}, "
                    f"avg_persistence={self._fmt_pers(h1.get('mean_persistence', 0))}\n"
                    f"Filtration: {filt}\n"
                    f"Note: Many features with small persistence is NORMAL for "
                    f"medical images (sublevel filtration on [0,1] normalized pixels)."
                )
        return "No PH data available."

    def _build_reference_stats(self, object_type: str, descriptor: str) -> str:
        """Build reference PH patterns and typical sparsity for the verify prompt."""
        ref = self.PH_REFERENCE_PATTERNS.get(object_type, {})
        if not ref:
            return "No reference patterns available for this object type."

        h0_lo, h0_hi = ref.get("H0_range", (0, 0))
        h1_lo, h1_hi = ref.get("H1_range", (0, 0))
        desc = ref.get("description", "")

        lines = [
            f"Object type: {object_type}",
            f"Typical PH: {desc}",
            f"H0 count range: {h0_lo}-{h0_hi} (approximate, varies with image content)",
            f"H1 count range: {h1_lo}-{h1_hi} (approximate, varies with image content)",
        ]

        # Per-descriptor sparsity from lookup table
        sparsity_info = self.DESCRIPTOR_SPARSITY.get(descriptor)
        if sparsity_info:
            lo, hi, note = sparsity_info
            lines.append(f"Expected sparsity for {descriptor}: {lo}-{hi}%. {note}")
        else:
            lines.append(f"No specific sparsity reference for {descriptor}.")

        return "\n".join(lines)

    def _get_descriptor_ranking(self, object_type: str, primary: str, alternative: str) -> str:
        """Look up benchmark rankings for the primary and alternative descriptors."""
        performers = TOP_PERFORMERS.get(object_type, [])
        if not performers:
            return "No ranking data available."

        lines = []
        for label, desc_name in [("Primary", primary), ("Alternative", alternative)]:
            for i, p in enumerate(performers):
                if p["descriptor"] == desc_name:
                    lines.append(f"{label}: {desc_name} ranked #{i+1}/{len(performers)} "
                                 f"for {object_type}")
                    break
            else:
                lines.append(f"{label}: {desc_name} not in top-15 rankings for {object_type}")

        # Add top-3 for context (rank only, no accuracy)
        top3 = performers[:3]
        lines.append(f"Top-3 for {object_type}: " + ", ".join(
            p['descriptor'] for p in top3
        ))

        return "\n".join(lines)

    def _skills_verify_and_reflect(self, state: TopoAgentState) -> Dict[str, Any]:
        """LLM verifies choice + reflects + generates report (1 LLM call).

        Produces structured JSON with verification, reflection, and final report.
        """
        import json as json_mod

        self._vprint_header("VERIFY & REFLECT (LLM #2)")

        plan = state.get("_plan") or {}
        ph_stats = state.get("_ph_stats") or {}
        feature_quality = state.get("_feature_quality") or {}
        descriptor = state.get("skill_descriptor", "unknown")
        skill_params = state.get("skill_params") or {}

        # Build prompt
        # Filter out non-serializable params
        safe_params = {
            k: v for k, v in skill_params.items()
            if k not in ("total_dim", "classifier", "color_mode", "dim")
        }

        object_type = plan.get("object_type", "surface_lesions")
        alternative = plan.get("alternative_descriptor", "betti_curves")

        prompt = SKILLS_VERIFY_PROMPT.format(
            plan_json=json_mod.dumps(plan, indent=2, default=str),
            descriptor_name=descriptor,
            actual_dim=feature_quality.get("dimension", 0),
            expected_dim=plan.get("expected_feature_dim", 0),
            variance=f"{feature_quality.get('variance', 0):.6f}",
            sparsity=f"{feature_quality.get('sparsity', 0):.1f}",
            nan_count=feature_quality.get("nan_count", 0),
            ph_stats=json_mod.dumps(ph_stats, indent=2, default=str) if ph_stats else "N/A",
            params_used=json_mod.dumps(safe_params, indent=2, default=str),
            reference_stats=self._build_reference_stats(object_type, descriptor),
            descriptor_ranking=self._get_descriptor_ranking(object_type, descriptor, alternative),
            object_type=object_type,
        )

        self._vprint_prompt(prompt)

        # Call LLM
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        self._vprint_response(response_text)

        # Parse verification JSON
        verify_json = self._parse_plan_json(response_text)

        # Extract reflection into ReflectionEntry
        reflection_data = verify_json.get("reflection", {})
        verification_data = verify_json.get("verification", {})
        # Use the LLM's actual suggestion instead of hardcoded "COMPLETE"
        llm_suggestion = verification_data.get("suggestion", "COMPLETE")
        reflection = ReflectionEntry(
            round=state.get("_retry_count", 0) + 1,
            error_analysis=reflection_data.get("error_analysis", "No error analysis provided"),
            suggestion=llm_suggestion,
            experience=reflection_data.get("experience", "No experience recorded"),
        )

        self._vprint_reflection(reflection)

        new_ltm = list(state["long_term_memory"])
        new_ltm.append(reflection)

        # Log LLM interaction
        interaction = LLMInteraction(
            step="verify_and_reflect",
            round=1,
            prompt=prompt,
            response=response_text,
        )
        new_interactions = list(state.get("llm_interactions", []))
        new_interactions.append(interaction)

        # Build final answer as structured JSON
        llm_report = verify_json.get("report", {})

        # Compute timing
        import time
        elapsed = time.time() - getattr(self, "_workflow_start_time", time.time())

        # Count tools executed
        n_tools = len([name for name, _ in state["short_term_memory"]])

        # Count LLM calls (current + previous)
        n_llm_calls = len(new_interactions)

        # Extract confidence from verification
        quality_ok = verification_data.get("quality_ok", True)
        confidence = 85.0 if quality_ok else 50.0

        # Build canonical AgentReport
        agent_report = AgentReport(
            # Core selection
            descriptor=descriptor,
            object_type=object_type,
            reasoning_chain=plan.get("reasoning_chain", ""),
            # Reasoning
            image_analysis=plan.get("image_analysis", ""),
            descriptor_rationale=plan.get("descriptor_rationale", ""),
            alternatives_considered=llm_report.get("alternatives_considered",
                                                   plan.get("alternative_rationale", "")),
            # Parameters
            parameters=safe_params,
            feature_dimension=plan.get("expected_feature_dim", 0),
            color_mode=state.get("skill_color_mode", "grayscale"),
            # Quality
            actual_dimension=feature_quality.get("dimension", 0),
            sparsity_pct=feature_quality.get("sparsity", 0.0),
            variance=feature_quality.get("variance", 0.0),
            nan_count=feature_quality.get("nan_count", 0),
            # Verification
            ph_confirms_object_type=verification_data.get("ph_confirms_object_type", True),
            dimension_correct=verification_data.get("dimension_correct", True),
            quality_ok=quality_ok,
            issues=verification_data.get("issues", []),
            # Reflection
            error_analysis=reflection.error_analysis,
            experience=reflection.experience,
            # Execution metadata
            n_llm_calls=n_llm_calls,
            n_tools_executed=n_tools,
            retry_used=state.get("_retry_count", 0) > 0,
            total_time_seconds=round(elapsed, 2),
        )

        # Build backward-compatible report dict with verification embedded
        report = agent_report.to_dict()
        # Also include the LLM's report fields for backward compat
        if llm_report:
            report["reasoning"] = llm_report.get("reasoning",
                                                  plan.get("descriptor_rationale", ""))
        report["verification"] = verification_data

        return {
            "final_answer": report,
            "long_term_memory": new_ltm,
            "llm_interactions": new_interactions,
            "confidence": confidence,
            "evidence": [str(item) for item in state["short_term_memory"]],
            "task_complete": True,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Verified: quality_ok={quality_ok}, time={elapsed:.1f}s, "
                f"experience='{reflection.experience[:60]}...'"
            ],
        }

    def _analyze_query(self, state: TopoAgentState) -> Dict[str, Any]:
        """Analyze the initial query and set up context.

        In skills_mode, also determines color mode from query/path context.

        Args:
            state: Current state

        Returns:
            Updated state dict
        """
        # Select system prompt based on mode
        if self.skills_mode:
            sys_prompt = SKILLS_SYSTEM_PROMPT
        else:
            sys_prompt = SYSTEM_PROMPT

        # Create initial system message
        system_message = HumanMessage(content=f"{sys_prompt}\n\nTask: {state['query']}\nImage: {state['image_path']}")

        result = {
            "messages": [system_message],
            "current_round": 1,
            "reasoning_trace": [f"Analyzing query: {state['query']}"]
        }

        # Skills mode: determine color mode early (metadata fact, not a reasoning decision)
        if self.skills_mode:
            color_mode = self.skill_registry.select_color_mode(
                image_path=state["image_path"],
            )
            result["skill_color_mode"] = color_mode

        return result

    def _select_tool(self, state: TopoAgentState) -> Dict[str, Any]:
        """Select the next tool to execute using ReAct pattern.

        Args:
            state: Current state

        Returns:
            Updated state dict
        """
        # Choose prompt based on mode (v4 skills > v3 adaptive > v2 default)
        if self.skills_mode:
            prompt_template = SKILLS_TOOL_SELECTION_PROMPT
        elif self.adaptive_mode:
            prompt_template = ADAPTIVE_TOOL_SELECTION_PROMPT
        else:
            prompt_template = TOOL_SELECTION_PROMPT

        # Build format kwargs
        format_kwargs = dict(
            query=state["query"],
            image_path=state["image_path"],
            short_term_memory=format_short_term_memory(state),
            long_term_memory=format_long_term_memory(state),
            current_round=state["current_round"],
            max_rounds=state["max_rounds"],
            tool_descriptions=format_tool_descriptions(self.tools),
        )

        # Add skill knowledge and context for skills mode
        if self.skills_mode:
            # Infer object type hint for benchmark rankings
            object_type_hint = self.skill_registry.infer_object_type_hint(
                query=state["query"],
                image_path=state["image_path"],
            )
            # Determine color mode
            color_mode = state.get("skill_color_mode") or "grayscale"

            # Build full knowledge context (descriptor properties + reasoning chains + rankings)
            format_kwargs["skill_knowledge"] = self.skill_registry.build_skill_context(
                object_type_hint=object_type_hint,
                color_mode=color_mode,
            )
            # Current selection state (descriptor + params if already chosen)
            format_kwargs["skill_context"] = format_skill_context(state)

        # Format the tool selection prompt
        prompt = prompt_template.format(**format_kwargs)

        # Verbose: print header and prompt
        self._vprint_header("TOOL SELECTION", round_num=state["current_round"])
        self._vprint_prompt(prompt)

        # Call model with tools — build proper conversation history.
        # Add the current prompt as a HumanMessage so the history alternates
        # correctly: [Human, AI+tool_calls, Tool, Human, AI+tool_calls, Tool, ...]
        current_prompt_msg = HumanMessage(content=prompt)
        messages_to_send = state["messages"] + [current_prompt_msg]

        response = self.model_with_tools.invoke(messages_to_send)

        # Verbose: print response
        self._vprint_response(
            response.content if hasattr(response, "content") else str(response)
        )

        # Log LLM interaction — check both langchain-native and raw tool_calls
        tool_calls_info = None
        lc_tool_calls = getattr(response, "tool_calls", []) or []
        raw_tool_calls = getattr(response, "additional_kwargs", {}).get("tool_calls", []) or []
        effective_tool_calls = lc_tool_calls or raw_tool_calls

        if effective_tool_calls:
            tool_calls_info = []
            for tc in effective_tool_calls:
                if isinstance(tc, dict):
                    name = tc.get("name") or tc.get("function", {}).get("name", "?")
                    args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        import json as _json
                        try:
                            args = _json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    tool_calls_info.append({"name": name, "args": args})

        # Verbose: print tool calls
        self._vprint_tool_calls(tool_calls_info or [])

        interaction = LLMInteraction(
            step="tool_selection",
            round=state["current_round"],
            prompt=prompt,
            response=response.content if hasattr(response, "content") else str(response),
            tool_calls=tool_calls_info
        )

        new_interactions = state.get("llm_interactions", []).copy()
        new_interactions.append(interaction)

        # If no tool calls detected, replace the AIMessage with a clean one
        # to avoid serialization issues where raw tool_calls leak through.
        if not effective_tool_calls:
            from langchain_core.messages import AIMessage as _AIMessage
            clean_response = _AIMessage(
                content=response.content or "(No tool selected this round)"
            )
            return {
                "messages": [current_prompt_msg, clean_response],
                "reasoning_trace": state["reasoning_trace"] + [f"Round {state['current_round']}: No tool selected"],
                "llm_interactions": new_interactions
            }

        return {
            "messages": [current_prompt_msg, response],
            "reasoning_trace": state["reasoning_trace"] + [f"Round {state['current_round']}: Selecting tool..."],
            "llm_interactions": new_interactions
        }

    def _execute_tool(self, state: TopoAgentState) -> Dict[str, Any]:
        """Execute the selected tool.

        Args:
            state: Current state

        Returns:
            Updated state dict with tool output
        """
        # Get the last message (should contain tool calls)
        last_message = state["messages"][-1]

        tool_outputs = []

        # Check both langchain-native and raw tool_calls
        lc_calls = getattr(last_message, "tool_calls", []) or []
        if not lc_calls:
            # Fallback: parse from additional_kwargs (raw OpenAI format)
            raw_calls = getattr(last_message, "additional_kwargs", {}).get("tool_calls", []) or []
            for rc in raw_calls:
                func = rc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                import json as _json
                try:
                    args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args = {}
                lc_calls.append({"name": name, "args": args, "id": rc.get("id", str(len(lc_calls)))})

        # === FORCED PIPELINE PROGRESSION (last-resort safety net) ===
        # With tool_choice="required", the LLM should always return a tool call.
        # This fallback only triggers if the LLM somehow returns no tool calls
        # (e.g., API error, malformed response). Log a WARNING when it triggers.
        forced = False
        if not lc_calls and self.skills_mode:
            import logging
            logger = logging.getLogger(__name__)
            called_tools = {name for name, _ in state.get("short_term_memory", [])}

            if "image_loader" not in called_tools and "image_loader" in self.tools:
                lc_calls = [{"name": "image_loader", "args": {"image_path": state["image_path"]}, "id": "forced_0"}]
                forced = True
                logger.warning("FORCED FALLBACK: image_loader (LLM returned no tool call)")
                if self.verbose:
                    print(f"\n  ⚠ WARNING FORCED: image_loader (LLM returned no tool call despite tool_choice=required)")
            elif "compute_ph" not in called_tools and "compute_ph" in self.tools:
                lc_calls = [{"name": "compute_ph", "args": {"filtration_type": "sublevel", "max_dimension": 1}, "id": "forced_1"}]
                forced = True
                logger.warning("FORCED FALLBACK: compute_ph (LLM returned no tool call)")
                if self.verbose:
                    print(f"\n  ⚠ WARNING FORCED: compute_ph (LLM returned no tool call despite tool_choice=required)")
            elif not any(name in SUPPORTED_DESCRIPTORS for name in called_tools):
                desc_name = self._get_forced_descriptor(state)
                if desc_name and desc_name in self.tools:
                    lc_calls = [{"name": desc_name, "args": {}, "id": "forced_2"}]
                    forced = True
                    logger.warning(f"FORCED FALLBACK: {desc_name} (LLM returned no tool call)")
                    if self.verbose:
                        print(f"\n  ⚠ WARNING FORCED: {desc_name} (LLM returned no tool call despite tool_choice=required)")

        if lc_calls:
            for tool_call in lc_calls:
                tool_name = tool_call["name"]
                tool_args = (tool_call.get("args") or {}).copy()

                # === v2 AUTO-INJECTION: Fix data passing between tools ===
                # Auto-inject missing data from previous tool outputs
                tool_args = self._auto_inject_args(tool_name, tool_args, state)

                if tool_name in self.tools:
                    try:
                        # Execute the tool
                        result = self.tools[tool_name].invoke(tool_args)
                        tool_outputs.append({
                            "tool_name": tool_name,
                            "output": result,
                            "success": True
                        })
                        # Verbose: print tool output
                        self._vprint_tool_output(tool_name, result, success=True)
                    except Exception as e:
                        tool_outputs.append({
                            "tool_name": tool_name,
                            "output": str(e),
                            "success": False
                        })
                        # Verbose: print failure
                        self._vprint_tool_output(tool_name, str(e), success=False)
                else:
                    tool_outputs.append({
                        "tool_name": tool_name,
                        "output": f"Tool '{tool_name}' not found",
                        "success": False
                    })
                    self._vprint_tool_output(tool_name, "Tool not found", success=False)

        # Create tool messages for LangGraph message history.
        # For forced calls, DON'T add ToolMessages — there's no matching
        # AIMessage with tool_calls, so OpenAI would error. The tool results
        # are still stored in short_term_memory via _tool_outputs.
        tool_messages = []
        if not forced:
            tool_messages = [
                ToolMessage(
                    content=str(truncate_output_for_prompt(out["output"], max_chars=1000)),
                    tool_call_id=lc_calls[i].get("id", str(i)) if i < len(lc_calls) else str(i)
                )
                for i, out in enumerate(tool_outputs)
            ]

            # DEFENSIVE: If the LLM returned tool_calls but fewer ToolMessages
            # were generated than expected, OpenAI will error on the next call
            # because every tool_call_id needs a matching ToolMessage response.
            responded_ids = {tm.tool_call_id for tm in tool_messages}
            for tc in lc_calls:
                tc_id = tc.get("id", "")
                if tc_id and tc_id not in responded_ids:
                    tool_messages.append(ToolMessage(
                        content="Tool execution skipped.",
                        tool_call_id=tc_id
                    ))

        result = {
            "messages": tool_messages,
            "_tool_outputs": tool_outputs  # Store for memory update
        }

        # === Skills: Detect descriptor choice and apply benchmark params ===
        # Record descriptor selection even on failure (we care about the choice)
        if self.skills_mode and not state.get("skill_descriptor"):
            for out in tool_outputs:
                tool_name = out["tool_name"]
                if tool_name in SUPPORTED_DESCRIPTORS:
                    # LLM chose this descriptor — record it and look up params
                    object_type_hint = self.skill_registry.infer_object_type_hint(
                        query=state["query"],
                        image_path=state["image_path"],
                    )
                    color_mode = state.get("skill_color_mode") or "grayscale"
                    if object_type_hint:
                        params = self.skill_registry.configure_after_selection(
                            descriptor=tool_name,
                            object_type=object_type_hint,
                            color_mode=color_mode,
                        )
                    else:
                        # No object type hint — use default params
                        from .skills.rules_data import get_optimal_params, get_descriptor_dim, get_classifier
                        params = get_optimal_params(tool_name, "surface_lesions")
                        params["total_dim"] = get_descriptor_dim(tool_name, "surface_lesions", color_mode)
                        params["classifier"] = get_classifier(tool_name, "surface_lesions", color_mode)
                        params["color_mode"] = color_mode
                    result["skill_descriptor"] = tool_name
                    result["skill_params"] = params
                    break

        return result

    @staticmethod
    def _compute_feature_quality(arr) -> Dict[str, Any]:
        """Compute rich feature quality metrics for REFLECT.

        Provides the LLM with enough information to make an informed
        COMPLETE/RETRY decision without code-level overrides.
        """
        import numpy as np
        from scipy import stats as scipy_stats

        finite = arr[np.isfinite(arr)]
        n = len(arr)
        n_finite = len(finite)

        quality = {
            "dimension": n,
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
            "sparsity": float((np.abs(arr) < 1e-10).mean()) * 100,
            "variance": float(np.var(arr)) if n_finite > 0 else 0.0,
        }

        if n_finite > 0:
            quality["mean"] = float(np.mean(finite))
            quality["std"] = float(np.std(finite))
            quality["min"] = float(np.min(finite))
            quality["max"] = float(np.max(finite))
            quality["dynamic_range"] = float(np.max(finite) - np.min(finite))

            # Distribution shape
            if n_finite > 3:
                quality["kurtosis"] = float(scipy_stats.kurtosis(finite, fisher=True))
                quality["skewness"] = float(scipy_stats.skew(finite))

            # Percentiles for spread analysis
            quality["p25"] = float(np.percentile(finite, 25))
            quality["p75"] = float(np.percentile(finite, 75))
            quality["iqr"] = quality["p75"] - quality["p25"]

            # Effective dimensionality: how many features carry >1% of total variance
            if quality["variance"] > 0:
                per_feature_var = (finite - np.mean(finite)) ** 2
                total_var = per_feature_var.sum()
                threshold = total_var * 0.01  # 1% of total
                quality["n_informative_features"] = int((per_feature_var > threshold / n_finite).sum())
            else:
                quality["n_informative_features"] = 0

            # Constant features (zero variance per feature — useful for detecting degenerate dims)
            quality["n_constant_features"] = int((np.abs(finite - np.mean(finite)) < 1e-12).sum())
        else:
            quality["mean"] = 0.0
            quality["std"] = 0.0
            quality["min"] = 0.0
            quality["max"] = 0.0
            quality["dynamic_range"] = 0.0
            quality["n_informative_features"] = 0
            quality["n_constant_features"] = n

        return quality

    def _auto_inject_args(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        state: TopoAgentState
    ) -> Dict[str, Any]:
        """Auto-inject missing arguments from previous tool outputs.

        This fixes the critical issue where LLM fails to pass data between tools.
        For example, Round 3 calling topological_features with empty args.

        v3 Enhancement: Also injects adaptive recommendations from image_analyzer.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments provided by the LLM
            state: Current state with short-term memory

        Returns:
            Updated tool_args with injected data
        """
        # Tools that need persistence_data from compute_ph
        persistence_tools = [
            "persistence_image",
            "topological_features",
            "betti_curves",
            "persistence_landscapes",
            "persistence_silhouette",
            "persistence_entropy",
            "persistence_statistics",
            "tropical_coordinates",
            "template_functions",
            "persistence_codebook",
            "ATOL",
            "wasserstein_distance",
            "bottleneck_distance",
            "betti_ratios",
            "persistence_diagram",
            "euler_characteristic",
            "total_persistence_stats",
        ]

        # Auto-inject persistence_data — ALWAYS override to avoid JSON corruption
        if tool_name in persistence_tools:
            persistence_data = get_persistence_data(state)
            if persistence_data is not None:
                tool_args["persistence_data"] = persistence_data

        # Auto-inject feature_vector for classifiers — ALWAYS override
        if tool_name in ["pytorch_classifier", "mlp_classifier", "knn_classifier"]:
            feature_vector = get_feature_vector(state)
            if feature_vector is not None:
                tool_args["feature_vector"] = feature_vector

        # Auto-inject image_array for compute_ph, image_analyzer, and image-based descriptors
        # ALWAYS override: LLM often copies arrays through JSON, corrupting structure
        image_needs_array = {
            "compute_ph", "image_analyzer", "noise_filter",
            "minkowski_functionals", "euler_characteristic_curve",
            "euler_characteristic_transform", "edge_histogram", "lbp_texture",
        }
        if tool_name in image_needs_array:
            image_array = get_image_array(state)
            if image_array is not None:
                tool_args["image_array"] = image_array

        # === v3 ADAPTIVE INJECTION ===
        # Inject recommended filtration_type from image_analyzer for compute_ph
        if tool_name == "compute_ph":
            # Only inject if not explicitly provided by the LLM
            if "filtration_type" not in tool_args or tool_args.get("filtration_type") is None:
                recommended_filtration = get_recommended_filtration(state)
                if recommended_filtration is not None:
                    tool_args["filtration_type"] = recommended_filtration

        # Inject recommended sigma from image_analyzer for persistence_image
        if tool_name == "persistence_image":
            # Only inject if not explicitly provided by the LLM
            if "sigma" not in tool_args or tool_args.get("sigma") is None:
                recommended_sigma = get_recommended_sigma(state)
                if recommended_sigma is not None:
                    tool_args["sigma"] = recommended_sigma

        # === v4 SKILLS INJECTION ===
        # Inject skill-configured parameters when skills_mode is active.
        # In agentic mode, the LLM sees parameters in the ACT prompt and
        # decides what to use — skip silent injection to preserve agency.
        if self.skills_mode and not self.agentic_mode:
            skill_params = state.get("skill_params") or {}
            skill_descriptor = state.get("skill_descriptor")

            if skill_params and tool_name == skill_descriptor:
                # Inject optimal parameters for the selected descriptor
                for key, value in skill_params.items():
                    if key in ("total_dim", "classifier", "color_mode", "dim"):
                        continue  # Skip meta-params
                    if key not in tool_args or tool_args.get(key) is None:
                        tool_args[key] = value

        return tool_args

    def _get_forced_descriptor(self, state: TopoAgentState) -> Optional[str]:
        """Get the recommended descriptor for forced pipeline progression.

        Uses skill knowledge (TOP_PERFORMERS by object type) to pick
        the best descriptor when the LLM fails to call a tool.
        """
        if not self.skills_mode:
            return "template_functions"

        object_type = self.skill_registry.infer_object_type_hint(
            query=state["query"],
            image_path=state["image_path"],
        )
        if object_type and object_type in TOP_PERFORMERS:
            return TOP_PERFORMERS[object_type][0]["descriptor"]
        return "template_functions"

    def _get_forced_descriptor_for_retry(self, state: TopoAgentState) -> str:
        """Get a descriptor for forced pipeline, avoiding the previously failed one.

        Used by the agentic pipeline when the LLM doesn't call a descriptor tool
        (common GPT-4o issue). Uses SUPPORTED_TOP_PERFORMERS to avoid training-based
        descriptors, and skips any descriptor mentioned in retry feedback.
        """
        from .skills.rules_data import SUPPORTED_TOP_PERFORMERS

        # Find which descriptor failed (mentioned in retry feedback)
        failed_desc = None
        retry_fb = state.get("_retry_feedback", "") or ""
        for desc in SUPPORTED_DESCRIPTORS:
            if desc in retry_fb:
                failed_desc = desc
                break

        # Get object type — prefer LLM's decision in agentic mode
        decisions = state.get("_observe_decisions") or {}
        object_type = decisions.get("object_type")
        if not object_type and hasattr(self, 'skill_registry'):
            object_type = self.skill_registry.infer_object_type_hint(
                query=state.get("query", ""),
                image_path=state.get("image_path"),
            )

        # Pick the top-ranked supported descriptor that isn't the failed one
        if object_type and object_type in SUPPORTED_TOP_PERFORMERS:
            for entry in SUPPORTED_TOP_PERFORMERS[object_type]:
                if entry["descriptor"] != failed_desc:
                    return entry["descriptor"]

        # Fallback: persistence_statistics (reliable, works everywhere)
        if failed_desc != "persistence_statistics":
            return "persistence_statistics"
        return "template_functions"

    def _update_short_memory(self, state: TopoAgentState) -> Dict[str, Any]:
        """Update short-term memory with tool outputs.

        Ms = Ms ∪ {(tool_t, output_t)}

        Respects enable_short_memory ablation flag.

        Args:
            state: Current state

        Returns:
            Updated state dict
        """
        tool_outputs = state.get("_tool_outputs", [])

        if self.enable_short_memory:
            # Normal behavior: accumulate memory
            new_memory = state["short_term_memory"].copy()
            for out in tool_outputs:
                new_memory.append((out["tool_name"], out["output"]))
        else:
            # Ablation: only keep current round's outputs (no accumulation)
            new_memory = [(out["tool_name"], out["output"]) for out in tool_outputs]

        return {
            "short_term_memory": new_memory,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Executed {len(tool_outputs)} tool(s), updated short-term memory"
            ]
        }

    def _reflect(self, state: TopoAgentState) -> Dict[str, Any]:
        """Generate reflection on current round.

        This is EndoAgent's key innovation adding +26.5% visual accuracy.

        Args:
            state: Current state

        Returns:
            Updated state dict with reflection
        """
        # Get recent actions (summarized to avoid token explosion from large arrays)
        recent_actions = state["short_term_memory"][-3:] if state["short_term_memory"] else []
        recent_output = recent_actions[-1][1] if recent_actions else "No output"

        # Summarize actions for prompt — raw outputs contain full pixel arrays
        summarized_actions = [
            (name, truncate_output_for_prompt(out, max_chars=500))
            for name, out in recent_actions
        ]
        summarized_output = truncate_output_for_prompt(recent_output, max_chars=500)

        # Choose prompt based on mode
        if self.skills_mode:
            prompt_template = SKILLS_REFLECTION_PROMPT
        elif self.adaptive_mode:
            prompt_template = ADAPTIVE_REFLECTION_PROMPT
        else:
            prompt_template = REFLECTION_PROMPT

        # Build format kwargs
        format_kwargs = dict(
            query=state["query"],
            current_round=state["current_round"],
            max_rounds=state["max_rounds"],
            recent_actions=str(summarized_actions),
            tool_output=str(summarized_output),
        )

        # Add skill context for skills mode
        if self.skills_mode:
            format_kwargs["skill_context"] = format_skill_context(state)

        # Format reflection prompt
        prompt = prompt_template.format(**format_kwargs)

        # Verbose: print reflection header and prompt
        self._vprint_header("REFLECTION", round_num=state["current_round"])
        self._vprint_prompt(prompt, max_chars=1200)

        # Call model for reflection
        response = self.model.invoke([HumanMessage(content=prompt)])

        # Verbose: print reflection response
        self._vprint_response(response.content, max_chars=1000)

        # Parse reflection (simplified - in practice would use structured output)
        reflection_text = response.content
        reflection = ReflectionEntry(
            round=state["current_round"],
            error_analysis=self._extract_section(reflection_text, "Error Analysis"),
            suggestion=self._extract_section(reflection_text, "Suggestion"),
            experience=self._extract_section(reflection_text, "Experience")
        )

        # Verbose: print parsed reflection
        self._vprint_reflection(reflection)

        # Log LLM interaction
        interaction = LLMInteraction(
            step="reflection",
            round=state["current_round"],
            prompt=prompt,
            response=reflection_text
        )

        new_interactions = state.get("llm_interactions", []).copy()
        new_interactions.append(interaction)

        return {
            "_current_reflection": reflection,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Reflection: {reflection.suggestion}"
            ],
            "llm_interactions": new_interactions
        }

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from reflection text.

        Args:
            text: Full reflection text
            section_name: Name of section to extract

        Returns:
            Extracted section content
        """
        # Simple extraction - look for section header
        lines = text.split("\n")
        in_section = False
        content = []

        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                if any(s in line.lower() for s in ["error analysis", "suggestion", "experience"]):
                    break
                content.append(line)

        return " ".join(content).strip() or f"No {section_name.lower()} provided"

    def _update_long_memory(self, state: TopoAgentState) -> Dict[str, Any]:
        """Update long-term memory with reflection.

        Ml = Ml ∪ {reflection_t}

        Respects enable_long_memory ablation flag.

        Args:
            state: Current state

        Returns:
            Updated state dict
        """
        if not self.enable_long_memory:
            # Ablation: skip long-term memory updates
            return {}

        reflection = state.get("_current_reflection")
        if reflection:
            new_memory = state["long_term_memory"].copy()
            new_memory.append(reflection)
            return {"long_term_memory": new_memory}
        return {}

    def _should_recover(self, state: TopoAgentState) -> Literal["recover", "continue"]:
        """Determine if error recovery is needed based on reflection.

        Checks if:
        1. The last tool execution failed
        2. Reflection contains actionable recovery suggestion
        3. We haven't exceeded recovery attempts

        Args:
            state: Current state

        Returns:
            "recover" if recovery action needed, "continue" otherwise
        """
        # Check if last tool execution failed
        tool_outputs = state.get("_tool_outputs", [])
        has_failure = any(not out.get("success", True) for out in tool_outputs)

        # Check if reflection suggests recovery
        reflection = state.get("_current_reflection")
        if reflection is None:
            return "continue"

        suggestion = reflection.suggestion.upper() if reflection.suggestion else ""

        # Keywords that indicate actionable recovery
        recovery_keywords = ["RETRY", "TRY", "SWITCH", "USE", "CHANGE", "ALTERNATIVE"]
        has_recovery_suggestion = any(kw in suggestion for kw in recovery_keywords)

        # Check recovery attempt count
        recovery_count = state.get("_recovery_count", 0)
        max_recoveries = 2  # Limit recovery attempts per session

        if has_failure and has_recovery_suggestion and recovery_count < max_recoveries:
            return "recover"
        return "continue"

    def _handle_recovery(self, state: TopoAgentState) -> Dict[str, Any]:
        """Execute recovery action based on reflection suggestion.

        Parses the reflection suggestion and executes appropriate recovery:
        - RETRY with different filtration (superlevel vs sublevel)
        - RETRY with different parameters
        - AUGMENT with additional features

        Args:
            state: Current state

        Returns:
            Updated state dict with recovery results
        """
        reflection = state.get("_current_reflection")
        if not reflection:
            return {}

        suggestion = reflection.suggestion.lower() if reflection.suggestion else ""
        recovery_action = None
        recovery_result = None

        # Increment recovery counter
        recovery_count = state.get("_recovery_count", 0) + 1

        # Parse suggestion and execute recovery
        try:
            if "superlevel" in suggestion and "compute_ph" in self.tools:
                # Retry PH computation with superlevel filtration
                recovery_action = "Switching to superlevel filtration"

                # Get image array from previous tool output
                image_array = get_image_array(state)
                if image_array is not None:
                    recovery_result = self.tools["compute_ph"].invoke({
                        "image_array": image_array,
                        "filtration_type": "superlevel"
                    })

            elif "sublevel" in suggestion and "compute_ph" in self.tools:
                # Retry with sublevel filtration
                recovery_action = "Switching to sublevel filtration"

                image_array = get_image_array(state)
                if image_array is not None:
                    recovery_result = self.tools["compute_ph"].invoke({
                        "image_array": image_array,
                        "filtration_type": "sublevel"
                    })

            elif "landscape" in suggestion and "persistence_landscapes" in self.tools:
                # Augment with persistence landscapes
                recovery_action = "Adding persistence landscapes features"

                persistence_data = get_persistence_data(state)
                if persistence_data is not None:
                    recovery_result = self.tools["persistence_landscapes"].invoke({
                        "persistence_data": persistence_data
                    })

            elif "topological_features" in suggestion and "topological_features" in self.tools:
                # Augment with statistical features
                recovery_action = "Adding topological statistical features"

                persistence_data = get_persistence_data(state)
                if persistence_data is not None:
                    recovery_result = self.tools["topological_features"].invoke({
                        "persistence_data": persistence_data
                    })

            elif "binariz" in suggestion and "binarization" in self.tools:
                # Preprocess with binarization
                recovery_action = "Applying binarization preprocessing"

                image_array = get_image_array(state)
                if image_array is not None:
                    recovery_result = self.tools["binarization"].invoke({
                        "image_array": image_array
                    })

            elif "noise" in suggestion and "noise_filter" in self.tools:
                # Preprocess with noise filtering
                recovery_action = "Applying noise filtering"

                image_array = get_image_array(state)
                if image_array is not None:
                    recovery_result = self.tools["noise_filter"].invoke({
                        "image_array": image_array
                    })

        except Exception as e:
            recovery_action = f"Recovery failed: {str(e)}"
            recovery_result = {"success": False, "error": str(e)}

        # Update state with recovery results
        updates = {
            "_recovery_count": recovery_count,
            "reasoning_trace": state["reasoning_trace"] + [
                f"Recovery action: {recovery_action or 'No action taken'}"
            ]
        }

        # Add recovery result to short-term memory if successful
        if recovery_result is not None and recovery_result.get("success", False):
            new_memory = state["short_term_memory"].copy()
            new_memory.append(("recovery", recovery_result))
            updates["short_term_memory"] = new_memory

        return updates

    def _check_completion(self, state: TopoAgentState) -> Dict[str, Any]:
        """Check if the task is complete.

        Args:
            state: Current state

        Returns:
            Updated state dict with completion status
        """
        # Format completion check prompt
        prompt = COMPLETION_CHECK_PROMPT.format(
            query=state["query"],
            short_term_memory=format_short_term_memory(state),
            current_round=state["current_round"],
            max_rounds=state["max_rounds"]
        )

        # Call model to check completion
        response = self.model.invoke([HumanMessage(content=prompt)])

        # Verbose: print completion check
        if self.verbose:
            self._vprint_header("COMPLETION CHECK", round_num=state["current_round"])
            self._vprint_response(response.content, max_chars=500)

        # Parse response (simplified)
        is_complete = "true" in response.content.lower() and "is_complete" in response.content.lower()

        # Skills mode: don't declare complete until a descriptor has been called
        if self.skills_mode and is_complete:
            called_tools = {name for name, _ in state.get("short_term_memory", [])}
            has_descriptor = any(name in SUPPORTED_DESCRIPTORS for name in called_tools)
            if not has_descriptor:
                is_complete = False
                if self.verbose:
                    print(f"  ⚠ Overriding completion: no descriptor tool called yet")

        # Also complete if max rounds reached
        if state["current_round"] >= state["max_rounds"]:
            is_complete = True

        if self.verbose:
            status = "COMPLETE" if is_complete else "CONTINUING"
            print(f"\n  ⏱  Status: {status} (round {state['current_round']}/{state['max_rounds']})")

        # Log LLM interaction
        interaction = LLMInteraction(
            step="completion_check",
            round=state["current_round"],
            prompt=prompt,
            response=response.content
        )

        new_interactions = state.get("llm_interactions", []).copy()
        new_interactions.append(interaction)

        return {
            "task_complete": is_complete,
            "current_round": state["current_round"] + 1,
            "llm_interactions": new_interactions
        }

    def _should_continue(self, state: TopoAgentState) -> Literal["continue", "finish"]:
        """Determine whether to continue or finish.

        Args:
            state: Current state

        Returns:
            "continue" or "finish"
        """
        if state["task_complete"] or state["current_round"] > state["max_rounds"]:
            return "finish"
        return "continue"

    def _generate_answer(self, state: TopoAgentState) -> Dict[str, Any]:
        """Generate the final answer.

        Args:
            state: Current state

        Returns:
            Updated state dict with final answer
        """
        # Format final answer prompt
        prompt = FINAL_ANSWER_PROMPT.format(
            query=state["query"],
            image_path=state["image_path"],
            short_term_memory=format_short_term_memory(state),
            long_term_memory=format_long_term_memory(state)
        )

        # Call model for final answer
        response = self.model.invoke([HumanMessage(content=prompt)])

        # Verbose: print final answer
        self._vprint_header("FINAL ANSWER")
        self._vprint_response(response.content, max_chars=2000)

        # Parse response to extract components
        answer_text = response.content

        # Extract confidence (simplified)
        confidence = 0.0
        if "confidence" in answer_text.lower():
            import re
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", answer_text)
            if match:
                confidence = float(match.group(1))

        # Log LLM interaction
        interaction = LLMInteraction(
            step="final_answer",
            round=state["current_round"],
            prompt=prompt,
            response=answer_text
        )

        new_interactions = state.get("llm_interactions", []).copy()
        new_interactions.append(interaction)

        return {
            "final_answer": answer_text,
            "confidence": confidence,
            "evidence": [str(item) for item in state["short_term_memory"]],
            "reasoning_trace": state["reasoning_trace"] + ["Generated final answer"],
            "llm_interactions": new_interactions
        }

    def invoke(self, query: str, image_path: str) -> TopoAgentState:
        """Run the workflow on an input.

        Args:
            query: User's query/task
            image_path: Path to input image

        Returns:
            Final state with answer
        """
        initial_state = create_initial_state(
            query=query,
            image_path=image_path,
            max_rounds=self.max_rounds
        )
        # Agentic mode needs more recursion budget (observe/act loops + retry)
        recursion_limit = 50 if self.agentic_mode else self.max_rounds * 10 + 10
        return self.workflow.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
        )

    async def ainvoke(self, query: str, image_path: str) -> TopoAgentState:
        """Async version of invoke.

        Args:
            query: User's query/task
            image_path: Path to input image

        Returns:
            Final state with answer
        """
        initial_state = create_initial_state(
            query=query,
            image_path=image_path,
            max_rounds=self.max_rounds
        )
        return await self.workflow.ainvoke(initial_state)


def create_topoagent_workflow(
    model,
    tools: Dict[str, Any],
    max_rounds: int = 4,
    enable_reflection: bool = True,
    enable_short_memory: bool = True,
    enable_long_memory: bool = True,
    adaptive_mode: bool = False,  # v3
    skills_mode: bool = False,    # v4
    agentic_mode: bool = False,   # v6
    verbose: bool = False,
) -> TopoAgentWorkflow:
    """Factory function to create TopoAgent workflow.

    Args:
        model: LangChain chat model
        tools: Dictionary of tools
        max_rounds: Maximum reasoning rounds (default 4 for v2, 5 for v3)
        enable_reflection: Enable reflection mechanism (ablation)
        enable_short_memory: Enable short-term memory (ablation)
        enable_long_memory: Enable long-term memory (ablation)
        adaptive_mode: Enable v3 adaptive pipeline with image_analyzer
        skills_mode: Enable v4 skills-based pipeline with benchmark rules
        agentic_mode: Enable v6 agentic 3-phase ReAct
        verbose: Print all LLM interactions to stdout

    Returns:
        Configured TopoAgentWorkflow
    """
    return TopoAgentWorkflow(
        model=model,
        tools=tools,
        max_rounds=max_rounds,
        enable_reflection=enable_reflection,
        enable_short_memory=enable_short_memory,
        enable_long_memory=enable_long_memory,
        adaptive_mode=adaptive_mode,
        skills_mode=skills_mode,
        agentic_mode=agentic_mode,
        verbose=verbose,
    )


# =============================================================================
# v5 Workflow: Reasoning-First Topology Feature Engineering Agent
# =============================================================================

# Image-based descriptors that don't need PH computation
IMAGE_BASED_DESCRIPTORS = {
    "minkowski_functionals",
    "euler_characteristic_curve",
    "euler_characteristic_transform",
    "edge_histogram",
    "lbp_texture",
}


class TopoAgentWorkflowV5:
    """v5 Workflow: 2-3 LLM calls, deterministic tool execution, dual verification.

    Architecture:
        1. ANALYZE & PLAN (1 LLM call) → structured JSON plan
        2. EXTRACT FEATURES (0 LLM calls) → deterministic tool chain
        3. VERIFY (0-1 LLM calls) → consistency check or TabPFN CV
        4. RETRY (0 LLM calls) → use alternative if verification failed
        5. OUTPUT REPORT (1 LLM call) → final feature report
        6. LEARN (0 LLM calls) → update skill memory
    """

    def __init__(
        self,
        model,
        tools: Dict[str, Any],
        verify_mode: str = "quick",
        skill_registry=None,
    ):
        self.model = model
        self.tools = tools
        self.verify_mode = verify_mode

        # Skill registry for knowledge context
        if skill_registry is None:
            from .skills import SkillRegistry
            self.skill_registry = SkillRegistry()
        else:
            self.skill_registry = skill_registry

        # Skill memory for learning
        from .memory.skill_memory import SkillMemory
        self.skill_memory = SkillMemory()

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the v5 LangGraph workflow."""
        workflow = StateGraph(TopoAgentStateV5)

        workflow.add_node("plan_and_reason", self._plan_and_reason)
        workflow.add_node("execute_plan", self._execute_plan)
        workflow.add_node("verify_results", self._verify_results)
        workflow.add_node("retry_with_alternative", self._retry_with_alternative)
        workflow.add_node("generate_report", self._generate_report)

        workflow.set_entry_point("plan_and_reason")
        workflow.add_edge("plan_and_reason", "execute_plan")
        workflow.add_edge("execute_plan", "verify_results")

        workflow.add_conditional_edges(
            "verify_results",
            self._should_retry,
            {
                "retry": "retry_with_alternative",
                "pass": "generate_report",
            },
        )
        workflow.add_edge("retry_with_alternative", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    # -------------------------------------------------------------------------
    # Node 1: ANALYZE & PLAN
    # -------------------------------------------------------------------------
    def _plan_and_reason(self, state: TopoAgentStateV5) -> Dict[str, Any]:
        """LLM analyzes context and produces a structured descriptor selection plan."""
        import json as json_mod

        # Infer object type hint
        object_type_hint = self.skill_registry.infer_object_type_hint(
            query=state["query"],
            image_path=state["image_path"],
        )
        color_mode = self.skill_registry.select_color_mode(
            image_path=state["image_path"],
        )

        # Build knowledge context
        skill_knowledge = self.skill_registry.build_skill_context(
            object_type_hint=object_type_hint,
            color_mode=color_mode,
        )

        # Get learned context
        learned_context = self.skill_memory.get_learned_context(object_type_hint)

        # Format prompt
        prompt = ANALYZE_AND_PLAN_PROMPT.format(
            query=state["query"],
            image_path=state["image_path"],
            skill_knowledge=skill_knowledge,
            learned_context=learned_context,
        )

        # Call LLM
        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Log interaction
        interaction = LLMInteraction(
            step="plan_and_reason",
            round=1,
            prompt=prompt,
            response=response_text,
        )

        # Parse JSON from response
        plan = self._parse_plan_json(response_text)

        # Validate and fixup plan
        plan = self._validate_plan(plan, object_type_hint, color_mode)

        # Configure params from benchmark rules
        object_type = plan.get("object_type", object_type_hint or "surface_lesions")
        descriptor = plan["descriptor_choice"]
        params = self.skill_registry.configure_after_selection(
            descriptor=descriptor,
            object_type=object_type,
            color_mode=plan.get("color_mode", color_mode),
        )

        return {
            "plan": plan,
            "skill_descriptor": descriptor,
            "skill_params": params,
            "skill_color_mode": plan.get("color_mode", color_mode),
            "llm_interactions": [interaction],
        }

    def _parse_plan_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        import json as json_mod

        # Try direct parse
        try:
            return json_mod.loads(text.strip())
        except json_mod.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        for marker in ["```json", "```"]:
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start)
                try:
                    return json_mod.loads(text[start:end].strip())
                except (json_mod.JSONDecodeError, ValueError):
                    pass

        # Try finding JSON object in text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                return json_mod.loads(text[brace_start : brace_end + 1])
            except json_mod.JSONDecodeError:
                pass

        # Fallback: return a default plan
        return {
            "object_type": "surface_lesions",
            "reasoning_chain": "h0_dominant",
            "image_analysis": "Could not parse LLM response. Using default.",
            "descriptor_choice": "template_functions",
            "descriptor_rationale": "Default fallback.",
            "needs_ph": True,
            "alternative_descriptor": "betti_curves",
            "alternative_rationale": "Default alternative.",
            "expected_feature_dim": 50,
            "color_mode": "grayscale",
        }

    def _validate_plan(
        self,
        plan: Dict[str, Any],
        object_type_hint: Optional[str],
        color_mode: str,
    ) -> Dict[str, Any]:
        """Validate and fix plan fields."""
        # Ensure descriptor_choice is valid
        if plan.get("descriptor_choice") not in SUPPORTED_DESCRIPTORS:
            plan["descriptor_choice"] = "template_functions"

        # Ensure alternative is valid
        alt = plan.get("alternative_descriptor")
        if alt not in SUPPORTED_DESCRIPTORS or alt == plan["descriptor_choice"]:
            # Pick next-best from TOP_PERFORMERS
            from .skills.rules_data import TOP_PERFORMERS
            ot = plan.get("object_type", object_type_hint or "surface_lesions")
            performers = TOP_PERFORMERS.get(ot, [])
            for p in performers:
                if p["descriptor"] != plan["descriptor_choice"]:
                    plan["alternative_descriptor"] = p["descriptor"]
                    break
            else:
                plan["alternative_descriptor"] = "betti_curves"

        # Fix needs_ph based on descriptor type
        plan["needs_ph"] = plan["descriptor_choice"] not in IMAGE_BASED_DESCRIPTORS

        # Fix color_mode if not valid
        if plan.get("color_mode") not in ("grayscale", "per_channel"):
            plan["color_mode"] = color_mode

        return plan

    # -------------------------------------------------------------------------
    # Node 2: EXECUTE PLAN (deterministic)
    # -------------------------------------------------------------------------
    def _execute_plan(self, state: TopoAgentStateV5) -> Dict[str, Any]:
        """Execute the deterministic tool chain based on the plan."""
        import numpy as np

        plan = state["plan"]
        descriptor = plan["descriptor_choice"]
        needs_ph = plan.get("needs_ph", True)
        trace = []
        image_array = None
        persistence_data = None
        ph_stats = None
        feature_vector = None

        # Step 1: Load image
        try:
            img_result = self.tools["image_loader"].invoke({
                "image_path": state["image_path"],
                "normalize": True,
                "grayscale": True,
            })
            if not img_result.get("success", False):
                trace.append({"step": "image_loader", "success": False, "error": img_result.get("error")})
                return {"execution_trace": trace}
            image_array = img_result["image_array"]
            trace.append({
                "step": "image_loader",
                "success": True,
                "shape": img_result.get("shape"),
            })
        except Exception as e:
            trace.append({"step": "image_loader", "success": False, "error": str(e)})
            return {"execution_trace": trace}

        # Step 2: Compute PH (if needed)
        if needs_ph and "compute_ph" in self.tools:
            try:
                ph_result = self.tools["compute_ph"].invoke({
                    "image_array": image_array,
                    "filtration_type": "sublevel",
                    "max_dimension": 1,
                })
                if ph_result.get("success", False):
                    persistence_data = ph_result.get("persistence", {})
                    ph_stats = ph_result.get("statistics", {})
                    h0_count = ph_stats.get("H0", {}).get("count", 0)
                    h1_count = ph_stats.get("H1", {}).get("count", 0)
                    trace.append({
                        "step": "compute_ph",
                        "success": True,
                        "H0": h0_count,
                        "H1": h1_count,
                        "filtration": "sublevel",
                    })
                else:
                    trace.append({"step": "compute_ph", "success": False, "error": ph_result.get("error")})
            except Exception as e:
                trace.append({"step": "compute_ph", "success": False, "error": str(e)})

        # Step 3: Extract features with chosen descriptor
        feature_result = self._run_descriptor(
            descriptor, image_array, persistence_data, plan, state
        )
        if feature_result.get("success", False):
            feature_vector = feature_result.get("combined_vector") or feature_result.get("feature_vector")
            if feature_vector is not None and isinstance(feature_vector, list):
                feature_vector = np.array(feature_vector, dtype=np.float64)
            trace.append({
                "step": descriptor,
                "success": True,
                "dim": len(feature_vector) if feature_vector is not None else 0,
            })
        else:
            trace.append({
                "step": descriptor,
                "success": False,
                "error": feature_result.get("error"),
            })

        return {
            "execution_trace": trace,
            "_image_array": image_array,
            "_persistence_data": persistence_data,
            "_ph_stats": ph_stats,
            "feature_vector": feature_vector,
        }

    def _run_descriptor(
        self,
        descriptor: str,
        image_array,
        persistence_data,
        plan: Dict,
        state: TopoAgentStateV5,
    ) -> Dict[str, Any]:
        """Run a descriptor tool with auto-injected params."""
        if descriptor not in self.tools:
            return {"success": False, "error": f"Tool '{descriptor}' not found"}

        tool = self.tools[descriptor]

        # Build args
        args = {}
        if descriptor in IMAGE_BASED_DESCRIPTORS:
            args["image_array"] = image_array
        else:
            if persistence_data is None:
                return {"success": False, "error": "No persistence data available"}
            args["persistence_data"] = persistence_data

        # Inject benchmark params from skill_params
        skill_params = state.get("skill_params") or {}
        for key, value in skill_params.items():
            if key in ("total_dim", "classifier", "color_mode", "dim"):
                continue
            args[key] = value

        try:
            return tool.invoke(args)
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Node 3: VERIFY RESULTS
    # -------------------------------------------------------------------------
    def _verify_results(self, state: TopoAgentStateV5) -> Dict[str, Any]:
        """Verify descriptor selection via reasoning or empirical evaluation."""
        import numpy as np

        plan = state["plan"]
        feature_vector = state.get("feature_vector")
        trace = state.get("execution_trace", [])

        # Check for extraction failure
        if feature_vector is None:
            return {
                "verification": {
                    "mode": "none",
                    "recommendation": "retry_alternative",
                    "reasoning": "Feature extraction failed.",
                }
            }

        verify_mode = state.get("verify_mode", self.verify_mode)

        if verify_mode == "thorough" and state.get("dataset_name"):
            return self._verify_thorough(state)
        else:
            return self._verify_quick(state)

    def _verify_quick(self, state: TopoAgentStateV5) -> Dict[str, Any]:
        """Mode A: LLM reasoning consistency check."""
        import numpy as np

        plan = state["plan"]
        feature_vector = state["feature_vector"]
        ph_stats = state.get("_ph_stats", {})

        fv = np.asarray(feature_vector) if feature_vector is not None else np.array([])

        # Compute feature quality stats
        if len(fv) > 0:
            variance = float(np.var(fv))
            nan_count = int(np.isnan(fv).sum())
            sparsity = float((np.abs(fv) < 1e-10).mean())
        else:
            variance = 0.0
            nan_count = 0
            sparsity = 1.0

        expected_dim = plan.get("expected_feature_dim", 0)
        actual_dim = len(fv)

        prompt = VERIFY_REASONING_PROMPT.format(
            plan_json=str(truncate_output_for_prompt(plan)),
            ph_stats=str(ph_stats),
            feature_dim=actual_dim,
            variance=f"{variance:.6f}",
            sparsity=f"{sparsity:.2%}",
            nan_count=nan_count,
            expected_dim=expected_dim,
            actual_dim=actual_dim,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        interaction = LLMInteraction(
            step="verify_reasoning",
            round=1,
            prompt=prompt,
            response=response_text,
        )

        # Parse verification result
        verification = self._parse_plan_json(response_text)
        verification["mode"] = "quick"

        # Determine recommendation from parsed result
        recommendation = verification.get("recommendation", "pass")
        # Auto-pass if basic quality is OK even if LLM is uncertain
        if nan_count == 0 and actual_dim > 0 and sparsity < 0.99:
            if recommendation not in ("pass", "retry_alternative"):
                recommendation = "pass"
        verification["recommendation"] = recommendation

        new_interactions = list(state.get("llm_interactions", []))
        new_interactions.append(interaction)

        return {
            "verification": verification,
            "llm_interactions": new_interactions,
        }

    def _verify_thorough(self, state: TopoAgentStateV5) -> Dict[str, Any]:
        """Mode B: TabPFN empirical verification on reference batch."""
        plan = state["plan"]
        dataset_name = state.get("dataset_name")
        object_type = plan.get("object_type", "surface_lesions")
        color_mode = plan.get("color_mode", "grayscale")

        # Get top 3 descriptors to compare
        from .skills.rules_data import get_top_descriptors
        top3 = get_top_descriptors(object_type, n=3)

        # Make sure primary choice is in the list
        primary = plan["descriptor_choice"]
        descriptors_to_test = [primary]
        for entry in top3:
            if entry["descriptor"] != primary and len(descriptors_to_test) < 3:
                descriptors_to_test.append(entry["descriptor"])

        results = {}
        try:
            results = self._quick_experiment(
                dataset_name=dataset_name,
                descriptors=descriptors_to_test,
                object_type=object_type,
                color_mode=color_mode,
                n_samples=100,
                cv_folds=3,
            )
        except Exception as e:
            # Fallback to pass if experiment fails
            return {
                "verification": {
                    "mode": "thorough",
                    "recommendation": "pass",
                    "reasoning": f"Thorough verification failed: {e}. Accepting plan.",
                    "results": {},
                }
            }

        # Find best descriptor
        if results:
            best_desc = max(results, key=lambda d: results[d].get("accuracy", 0))
            best_acc = results[best_desc]["accuracy"]
            primary_acc = results.get(primary, {}).get("accuracy", 0)

            if best_desc == primary or primary_acc >= best_acc - 0.02:
                recommendation = "pass"
                reasoning = (
                    f"{primary} ({primary_acc:.1%}) confirmed as best "
                    f"(or within 2% of {best_desc} at {best_acc:.1%})."
                )
            else:
                recommendation = "retry_alternative"
                reasoning = (
                    f"{best_desc} ({best_acc:.1%}) outperforms "
                    f"{primary} ({primary_acc:.1%}) by >{2:.0%}."
                )
        else:
            recommendation = "pass"
            reasoning = "No comparison results available."

        return {
            "verification": {
                "mode": "thorough",
                "recommendation": recommendation,
                "reasoning": reasoning,
                "results": results,
            }
        }

    def _quick_experiment(
        self,
        dataset_name: str,
        descriptors: list,
        object_type: str,
        color_mode: str,
        n_samples: int = 100,
        cv_folds: int = 3,
    ) -> Dict[str, Dict]:
        """Run quick TabPFN CV experiment using benchmark4 infrastructure."""
        import sys
        from pathlib import Path

        # Add benchmark4 to path
        project_root = Path(__file__).parent.parent
        b4_dir = project_root / "RuleBenchmark" / "benchmark4"
        if str(b4_dir) not in sys.path:
            sys.path.insert(0, str(b4_dir))

        from data_loader import load_dataset
        from precompute_ph import compute_ph_for_images
        from descriptor_runner import extract_features
        from optimal_rules import OptimalRules
        from classifier_wrapper import get_classifier

        # Load reference batch
        images, labels, class_names = load_dataset(dataset_name, n_samples=n_samples, seed=42)

        # Compute PH (needed for PH-based descriptors)
        ph_needed = any(d not in IMAGE_BASED_DESCRIPTORS for d in descriptors)
        diagrams = None
        if ph_needed:
            diagrams = compute_ph_for_images(images, color_mode=color_mode)

        # Evaluate each descriptor
        rules = OptimalRules()
        results = {}
        for desc in descriptors:
            try:
                params = rules.get_params(desc, object_type)
                features = extract_features(
                    desc, images, diagrams, params,
                    color_mode=color_mode,
                )
                if features is None or len(features) == 0:
                    continue

                # Quick CV with TabPFN
                from sklearn.model_selection import StratifiedKFold
                from sklearn.preprocessing import StandardScaler
                import numpy as np

                clf = get_classifier("TabPFN")
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = []
                for train_idx, test_idx in skf.split(features, labels):
                    X_train, X_test = features[train_idx], features[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # Clip extreme values
                    X_train = np.clip(X_train, -1e6, 1e6)
                    X_test = np.clip(X_test, -1e6, 1e6)

                    clf.fit(X_train, y_train)
                    scores.append(clf.score(X_test, y_test))

                results[desc] = {
                    "accuracy": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "dim": features.shape[1],
                }
            except Exception as e:
                results[desc] = {"accuracy": 0.0, "error": str(e)}

        return results

    def _should_retry(self, state: TopoAgentStateV5) -> Literal["retry", "pass"]:
        """Decide whether to retry with alternative descriptor."""
        verification = state.get("verification", {})
        if not verification:
            return "pass"

        recommendation = verification.get("recommendation", "pass")
        if recommendation == "retry_alternative" and not state.get("retry_used", False):
            return "retry"
        return "pass"

    # -------------------------------------------------------------------------
    # Node 4: RETRY WITH ALTERNATIVE (deterministic)
    # -------------------------------------------------------------------------
    def _retry_with_alternative(self, state: TopoAgentStateV5) -> Dict[str, Any]:
        """Re-extract features with the alternative descriptor from the plan."""
        import numpy as np

        plan = state["plan"]
        alt_descriptor = plan.get("alternative_descriptor", "betti_curves")
        image_array = state.get("_image_array")
        persistence_data = state.get("_persistence_data")

        # May need PH if alternative is PH-based
        if alt_descriptor not in IMAGE_BASED_DESCRIPTORS and persistence_data is None:
            if image_array is not None and "compute_ph" in self.tools:
                try:
                    ph_result = self.tools["compute_ph"].invoke({
                        "image_array": image_array,
                        "filtration_type": "sublevel",
                        "max_dimension": 1,
                    })
                    if ph_result.get("success", False):
                        persistence_data = ph_result.get("persistence", {})
                except Exception:
                    pass

        # Configure params for alternative
        object_type = plan.get("object_type", "surface_lesions")
        color_mode = plan.get("color_mode", "grayscale")
        alt_params = self.skill_registry.configure_after_selection(
            descriptor=alt_descriptor,
            object_type=object_type,
            color_mode=color_mode,
        )

        # Run alternative descriptor
        alt_state = dict(state)
        alt_state["skill_params"] = alt_params
        feature_result = self._run_descriptor(
            alt_descriptor, image_array, persistence_data, plan, alt_state
        )

        trace = list(state.get("execution_trace", []))
        feature_vector = state.get("feature_vector")

        if feature_result.get("success", False):
            fv = feature_result.get("combined_vector") or feature_result.get("feature_vector")
            if fv is not None and isinstance(fv, list):
                fv = np.array(fv, dtype=np.float64)
            feature_vector = fv
            trace.append({
                "step": f"{alt_descriptor} (alternative)",
                "success": True,
                "dim": len(fv) if fv is not None else 0,
            })
        else:
            trace.append({
                "step": f"{alt_descriptor} (alternative)",
                "success": False,
                "error": feature_result.get("error"),
            })

        return {
            "execution_trace": trace,
            "feature_vector": feature_vector,
            "skill_descriptor": alt_descriptor,
            "skill_params": alt_params,
            "retry_used": True,
        }

    # -------------------------------------------------------------------------
    # Node 5: GENERATE REPORT (1 LLM call)
    # -------------------------------------------------------------------------
    def _generate_report(self, state: TopoAgentStateV5) -> Dict[str, Any]:
        """Generate the final topology feature report."""
        plan = state.get("plan", {})
        trace = state.get("execution_trace", [])
        verification = state.get("verification", {})

        # Build execution trace summary
        trace_lines = []
        for entry in trace:
            step = entry.get("step", "unknown")
            success = entry.get("success", False)
            if success:
                details = {k: v for k, v in entry.items() if k not in ("step", "success")}
                trace_lines.append(f"[OK] {step}: {details}")
            else:
                trace_lines.append(f"[FAIL] {step}: {entry.get('error', 'unknown')}")

        prompt = OUTPUT_REPORT_PROMPT.format(
            query=state["query"],
            image_path=state["image_path"],
            plan_summary=str(truncate_output_for_prompt(plan)),
            execution_trace="\n".join(trace_lines),
            verification_result=str(truncate_output_for_prompt(verification)),
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        report = response.content

        interaction = LLMInteraction(
            step="generate_report",
            round=1,
            prompt=prompt,
            response=report,
        )

        new_interactions = list(state.get("llm_interactions", []))
        new_interactions.append(interaction)

        return {
            "report": report,
            "llm_interactions": new_interactions,
        }

    # -------------------------------------------------------------------------
    # invoke / ainvoke
    # -------------------------------------------------------------------------
    def invoke(
        self,
        query: str,
        image_path: str,
        verify_mode: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> TopoAgentStateV5:
        """Run the v5 workflow.

        Args:
            query: User's query/task
            image_path: Path to input image
            verify_mode: "quick" or "thorough" (overrides instance default)
            dataset_name: Dataset name for Mode B verification

        Returns:
            Final TopoAgentStateV5 with plan, features, verification, report
        """
        initial_state = create_initial_state_v5(
            query=query,
            image_path=image_path,
            verify_mode=verify_mode or self.verify_mode,
            dataset_name=dataset_name,
        )
        return self.workflow.invoke(
            initial_state,
            config={"recursion_limit": 20},
        )
