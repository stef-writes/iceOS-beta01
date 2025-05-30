# ScriptChain 3.0: Architecture Overview

## 1. High-Level Architecture Diagram (Text/ASCII)

```
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|   API Endpoint    |  --->   |  ScriptChain      |  --->   |  GraphContextMgr  |
|  /chains/execute  |         |  (Orchestrator)   |         |  (Context Store)  |
|                   |         |                   |         |                   |
+-------------------+         +-------------------+         +-------------------+
                                      |
                                      v
                          +-------------------------+
                          |  Dependency Graph (DAG) |
                          +-------------------------+
                                      |
                                      v
+-------------------+    +-------------------+    +-------------------+
|                   |    |                   |    |                   |
|   Node (Level 0)  |    |   Node (Level 0)  |    |   Node (Level 0)  |
|   (OpenAI)        |    |   (Gemini)        |    |   (Anthropic)     |
+-------------------+    +-------------------+    +-------------------+
         |                        |                        |
         +-----------+------------+------------+-----------+
                     |                         |
                     v                         v
              +-------------------+    +-------------------+
              |   Node (Level 1)  |    |   Node (Level 1)  |
              |   (OpenAI)        |    |   (Gemini)        |
              +-------------------+    +-------------------+
                     |                         |
                     +-----------+-------------+
                                 |
                                 v
                        +-------------------+
                        |   Node (Level 2)  |
                        |   (Anthropic)     |
                        +-------------------+
```

---

## 2. Component Breakdown

### A. API Layer
- **Receives**: Workflow definition (nodes, dependencies, models, schemas).
- **Calls**: `ScriptChain` orchestrator.

### B. ScriptChain (Orchestrator)
- **Builds**: Dependency graph (DAG) from node configs.
- **Assigns**: Levels to nodes based on dependencies.
- **Executes**: Each level sequentially; nodes within a level in parallel (asyncio).
- **Dispatches**: Each node to the correct LLM handler (OpenAI, Gemini, Anthropic, etc.).
- **Handles**: Errors, metrics, and callbacks.

### C. Node Execution
- **Each Node**:
  - Has its own model/provider, prompt, input/output schema.
  - Receives context (outputs from dependencies) via the context manager.
  - Executes via the appropriate LLM handler.
  - Validates output against schema.

### D. GraphContextManager
- **Stores**: All intermediate and final outputs.
- **Provides**: Context to nodes as needed (based on input mappings or default rules).
- **Ensures**: Data integrity and type safety (via schema validation).

### E. LLM Handlers
- **Abstracted**: Each provider (OpenAI, Gemini, Anthropic, etc.) has its own handler.
- **Plug-and-play**: Nodes can use any supported LLM, even within the same workflow.

---

## 3. Data Flow Example

1. **API** receives a workflow with 5 nodes, each specifying its model and dependencies.
2. **ScriptChain** builds the DAG, assigns levels:
   - Level 0: Nodes A, B, C (no dependencies)
   - Level 1: Nodes D, E (depend on outputs from A, B, C)
   - Level 2: Node F (depends on D, E)
3. **Execution**:
   - Level 0 nodes run in parallel (each with its own LLM handler).
   - Outputs are stored in the context manager.
   - Level 1 nodes start once their dependencies are done, using outputs from Level 0 as context.
   - Level 2 node runs last, using outputs from Level 1.
4. **Context** is passed and validated at each step.
5. **Results** are aggregated and returned via the API.

---

## 4. Key Technical Features

- **Parallelism**: Asyncio + semaphore for concurrency control.
- **Multi-LLM**: Per-node model/provider selection.
- **Context Management**: Centralized, schema-validated, supports complex mappings.
- **Error Handling**: Skips or fails dependents if a node fails, but continues where possible.
- **Extensibility**: New LLMs or node types can be added with minimal changes.

---

## 5. Mermaid Diagram (for Markdown Viewers)

Paste this into a Markdown viewer that supports [Mermaid](https://mermaid-js.github.io/mermaid/):

```mermaid
flowchart TD
    API[/API: /chains/execute/]
    ScriptChain[ScriptChain Orchestrator]
    ContextMgr[GraphContextManager]
    subgraph Level 0
        A[Node A (OpenAI)]
        B[Node B (Gemini)]
        C[Node C (Anthropic)]
    end
    subgraph Level 1
        D[Node D (OpenAI)]
        E[Node E (Gemini)]
    end
    F[Node F (Anthropic)]
    API --> ScriptChain
    ScriptChain --> ContextMgr
    ScriptChain --> A
    ScriptChain --> B
    ScriptChain --> C
    A --> D
    B --> D
    C --> E
    D --> F
    E --> F
    A --> ContextMgr
    B --> ContextMgr
    C --> ContextMgr
    D --> ContextMgr
    E --> ContextMgr
    F --> ContextMgr
```

---

## 6. Summary

Your system flexibly orchestrates complex, multi-level workflows using multiple LLMs, with robust context management, parallel execution, and error handling—enabling advanced, modular AI pipelines.

---

**For further details, see the code in `app/chains/script_chain.py`, `app/models/node_models.py`, and related modules.** 