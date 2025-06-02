# Opportunity Fit Checklist

For each new opportunity, answer the following:

## A. Workflow & Orchestration
- [ ] Does the problem require multi-step, branching, or parallel workflows?
- [ ] Are there dependencies between steps (i.e., outputs from one step feed into another)?
- [ ] Is there a need for persistent or shareable context across steps?

## B. LLM + Tool/Function Integration
- [ ] Does the solution require LLMs to call external tools, APIs, or functions?
- [ ] Is function-calling, plugin use, or tool augmentation a core requirement?
- [ ] Will the LLM need to reason about when/how to use tools?

## C. Agentic/Autonomous Behavior
- [ ] Should the system be able to make decisions, loop, or adapt based on intermediate results?
- [ ] Is there a need for agent-like behavior (e.g., "thought loops," retries, or conditional logic)?

## D. Observability & Metrics
- [ ] Is tracking of token usage, costs, or execution metrics important?
- [ ] Is detailed logging, debugging, or auditability required?

## E. Scale & Performance
- [ ] Will the system need to handle high concurrency or large-scale workflows?
- [ ] Are there strict real-time or low-latency requirements?

## F. Security & Compliance
- [ ] Are there regulatory, compliance, or security requirements?
- [ ] Will tools or LLMs access sensitive data or perform critical actions?

## G. Ecosystem Fit
- [ ] Is the target environment Python-friendly?
- [ ] Are there integration needs with non-Python systems? 