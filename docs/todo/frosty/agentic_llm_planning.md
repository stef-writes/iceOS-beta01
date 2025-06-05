# Agentic Workflow Support: LLM-Driven Planning for ScriptChain

---

## Overview

This document outlines the requirements, design, and implementation steps for enabling agentic workflow support in iceOS-Beta01, where an LLM outputs executable plans (ScriptChains) based on user goals and available tools/nodes. This bridges the vision in ENGINEERING_TODO.md and frosty.md, and provides a concrete path to LLM-powered, agentic orchestration.

---

## Goals
- Allow users (or agents) to specify high-level goals in natural language.
- Use an LLM to generate a ScriptChain plan (nodes, dependencies, parameters) dynamically.
- Parse, validate, and execute the generated plan using the existing ScriptChain system.
- Support iterative refinement: results can be fed back to the LLM for replanning.
- Integrate with Frosty Copilot for prompt-to-architecture and visual editing.

---

## Requirements
- **LLM Planning Module**: Service/class to prompt the LLM for a plan, given a goal, context, and available tools/nodes.
- **Plan Schema Compliance**: LLM output must conform to ScriptChain/node schemas (JSON or YAML).
- **Plan Parsing & Validation**: Parse LLM output, validate for correctness (no cycles, valid node types, required fields).
- **Dynamic ScriptChain Instantiation**: Instantiate ScriptChain from the generated plan.
- **Execution & Feedback Loop**: Run the chain, optionally replan based on results/errors.
- **UI/API Integration**: Entry point for users/agents to specify goals and review/modify plans.
- **Human-in-the-Loop**: Support user review, editing, and iterative improvement.

---

## Design

### 1. LLM Planning Module
- Input: User goal, available tools/nodes, (optional) context.
- Output: ScriptChain plan (JSON/YAML).
- Prompt engineering: Provide schema, examples, and constraints to the LLM.

### 2. Plan Parsing & Validation
- Parse LLM output into NodeConfig/ScriptChain objects.
- Validate using schema and dependency logic.
- Handle errors gracefully (re-prompt LLM or user for corrections).

### 3. ScriptChain Instantiation & Execution
- Instantiate ScriptChain with generated nodes/levels.
- Inject initial context as needed.
- Execute and collect results.

### 4. Iterative Feedback Loop
- After execution, feed results/errors back to LLM for refinement.
- Support user edits and re-prompts.

### 5. UI/API Integration
- New endpoint or UI action: accepts user goal, triggers LLM planning, displays/editable plan, runs ScriptChain.
- Integrate with Frosty Copilot for prompt-to-architecture and canvas view.

---

## Implementation Steps

1. **Design LLM Planning Prompts**
   - Define prompt templates and schema examples for the LLM.
   - Test with various goals and toolsets.

2. **Build LLM Planning Module**
   - Service/class to call LLM, return plan.
   - Error handling for invalid outputs.

3. **Plan Parsing & Validation**
   - Parse LLM output to ScriptChain/node objects.
   - Validate structure, dependencies, and types.

4. **ScriptChain Instantiation**
   - Instantiate and execute ScriptChain from plan.
   - Collect and return results.

5. **Feedback Loop**
   - Implement mechanism to re-prompt LLM with results/errors for iterative planning.

6. **UI/API Integration**
   - Add endpoint or UI for agentic workflow creation.
   - Integrate with Frosty Copilot and canvas/code views.

7. **Testing & Examples**
   - Create example user flows and test cases.
   - Document with sample prompts, plans, and results.

---

## References
- ENGINEERING_TODO.md (Configuration, Modularity, Copilot, System Success)
- frosty.md (Vision, Prompt-to-Architecture, Human-in-the-Loop, Canvas View)
- ScriptChain schemas and implementation

---

**This document is a living spec. Update as implementation progresses.** 