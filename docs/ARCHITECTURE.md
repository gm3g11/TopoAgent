# TopoAgent Architecture

> ⚠️ **Legacy document.** This describes the earlier ReAct + Reflection (v2/v4) workflow. The current paper pipeline is the **v9 Perception–Reasoning–Action–Reflection (PRAR)** loop — see [README → How the PRAR phases map to the code](../README.md#how-the-prar-phases-map-to-the-code) for the authoritative architecture.

## Overview

TopoAgent (legacy v2/v4) implements a **ReAct + Reflection** workflow for medical image classification using Topological Data Analysis (TDA).

## Core Components

### 1. TopoAgentState (`topoagent/state.py`)

The state is a TypedDict that flows through the LangGraph workflow:

```python
class TopoAgentState(TypedDict):
    # Input
    query: str                                    # User's task/question
    image_path: str                               # Path to medical image

    # Short-term memory Ms (EndoAgent)
    short_term_memory: List[Tuple[str, Any]]     # [(tool_name, output), ...]

    # Long-term memory Ml (EndoAgent)
    long_term_memory: List[ReflectionEntry]      # Past reflections

    # Control
    current_round: int                            # Current round (1-3)
    max_rounds: int                               # Max rounds (default 3)
    reasoning_trace: List[str]                    # Step-by-step reasoning

    # LangGraph
    messages: Annotated[List[Any], operator.add]  # Message accumulation

    # Output
    final_answer: Optional[str]
    confidence: float
    evidence: List[str]
    task_complete: bool
```

### 2. LangGraph Workflow (`topoagent/workflow.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Workflow Graph                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [START] → analyze_query → select_tool → execute_tool           │
│                                              ↓                   │
│                                    update_short_memory           │
│                                              ↓                   │
│                                          reflect                 │
│                                              ↓                   │
│                                    update_long_memory            │
│                                              ↓                   │
│                                      check_completion            │
│                                        ↓         ↓               │
│                                  [continue]   [finish]           │
│                                      ↓           ↓               │
│                              select_tool    generate_answer      │
│                                                  ↓               │
│                                               [END]              │
└─────────────────────────────────────────────────────────────────┘
```

**Nodes:**
- `analyze_query`: Parse input and initialize context
- `select_tool`: ReAct-style tool selection using LLM
- `execute_tool`: Run selected tool(s)
- `update_short_memory`: Add tool output to Ms
- `reflect`: Generate reflection on execution
- `update_long_memory`: Add reflection to Ml
- `check_completion`: Decide continue or finish
- `generate_answer`: Produce final classification

**Conditional Edge:**
- `check_completion` → `continue` (if round < 3 and not complete)
- `check_completion` → `finish` (if complete or round >= 3)

### 3. Reflection Engine (`topoagent/reflection.py`)

Implements EndoAgent's reflection mechanism:

```python
class ReflectionResult:
    entry: ReflectionEntry      # Structured reflection
    is_task_complete: bool      # Completion signal
    confidence: float           # Confidence estimate
    next_action_suggestion: str # What to do next

class ReflectionEntry:
    round: int                  # Which round
    error_analysis: str         # What went wrong
    suggestion: str             # What to do next
    experience: str             # Reusable lesson
```

**Reflection Prompt Structure:**
1. Error Analysis: What could be improved?
2. Suggestion: What should be done next?
3. Experience: What general lesson can be learned?
4. Task Completion: Is the task complete?

### 4. Dual-Memory System (`topoagent/memory/`)

**Short-term Memory (Ms):**
- Stores recent tool executions
- Session-scoped (cleared between sessions)
- Used for: avoiding redundant calls, tracking progress

**Long-term Memory (Ml):**
- Stores reflection experiences
- Persistent across sessions
- Used for: learning from past mistakes, improving future decisions

## Tool Architecture

### Tool Categories

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│ PREPROCESSING  │    │   FILTRATION   │    │    HOMOLOGY    │
├────────────────┤    ├────────────────┤    ├────────────────┤
│ ImageLoader    │───▶│ Sublevel       │───▶│ ComputePH      │
│ Binarization   │    │ Superlevel     │    │ PersistDiagram │
│ NoiseFilter    │    │ Cubical        │    │ PersistImage   │
└────────────────┘    └────────────────┘    └────────────────┘
                                                    │
┌────────────────┐    ┌────────────────┐           │
│CLASSIFICATION  │◀───│   FEATURES     │◀──────────┘
├────────────────┤    ├────────────────┤
│ KNN            │    │ TopoFeatures   │
│ MLP            │    │ Wasserstein    │
│ Ensemble       │    │ Bottleneck     │
└────────────────┘    └────────────────┘
```

### Tool Interface

All tools inherit from LangChain's `BaseTool`:

```python
class SomeTool(BaseTool):
    name: str = "tool_name"
    description: str = "When to use this tool..."
    args_schema: Type[BaseModel] = ToolInput

    def _run(self, **kwargs) -> Dict[str, Any]:
        # Tool implementation
        return {"success": True, "output": result}
```

### Tool Selection

The LLM selects tools based on:
1. Task requirements (from query)
2. Short-term memory (what was already done)
3. Long-term memory (past experiences)
4. Tool descriptions (capabilities)

## Data Flow

```
Input Image
    │
    ▼
┌─────────────────┐
│  ImageLoader    │ → numpy array, metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ → cleaned/binarized image
│  (optional)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Filtration    │ → filtration values
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ComputePH     │ → persistence pairs {H0: [...], H1: [...]}
└────────┬────────┘
         │
         ├─────────────────┐
         ▼                 ▼
┌─────────────────┐ ┌─────────────────┐
│ PersistDiagram  │ │ PersistImage    │
│ (analysis)      │ │ (vectorization) │
└────────┬────────┘ └────────┬────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
         ┌─────────────────┐
         │ TopoFeatures    │ → feature vector
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Classifier     │ → prediction, confidence
         └─────────────────┘
```

## Key Algorithms

### Algorithm 1: Main Loop (from EndoAgent)

```
Input: query, image_path
Output: classification, confidence, evidence

Initialize:
    Ms = []  # short-term memory
    Ml = load_long_term_memory()
    round = 0

for round = 1 to MAX_ROUNDS (3):
    # Select tool using ReAct
    tool = SelectTool(query, Ms, Ml, available_tools)

    # Execute tool
    output = tool.invoke(image_path, ...)

    # Update short-term memory
    Ms.append((tool.name, output))

    # Reflect on execution
    reflection = Reflect(query, Ms, Ml, output)

    # Update long-term memory
    Ml.append(reflection)

    # Check completion
    if IsTaskComplete(reflection):
        break

# Generate final answer
answer = GenerateAnswer(query, Ms, Ml)
return answer
```

### Algorithm 2: Reflection

```
Input: query, short_term_memory, long_term_memory, tool_output
Output: ReflectionEntry

Analyze:
    1. What was the goal of this step?
    2. Did the tool output help achieve it?
    3. What errors or issues occurred?

Generate:
    error_analysis = "What went wrong or could improve"
    suggestion = "What to do next"
    experience = "Reusable lesson for future"

Determine:
    is_complete = HasSufficientEvidence(short_term_memory)
    confidence = EstimateConfidence(tool_output)

return ReflectionEntry(error_analysis, suggestion, experience)
```

## Performance Insights (from EndoAgent)

| Configuration | Visual Acc | Language Acc |
|--------------|------------|--------------|
| Baseline (no reflection) | - | - |
| + Reflection | +26.5% | - |
| + Dual-memory | +1.5% | +3.06% |
| Max rounds = 3 | Optimal | Optimal |

## Extension Points

1. **New Tools**: Add to `topoagent/tools/<category>/`
2. **New Prompts**: Modify `topoagent/prompts.py`
3. **Custom Memory**: Extend `ShortTermMemory` or `LongTermMemory`
4. **New Benchmarks**: Add to `benchmark/`
