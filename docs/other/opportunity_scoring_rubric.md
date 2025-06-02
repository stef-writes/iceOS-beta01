# Opportunity Scoring Rubric

Score each opportunity on a scale of 0–2 for each dimension:

| Score | Meaning                |
|-------|------------------------|
| 0     | Not required/No fit    |
| 1     | Somewhat/Partial fit   |
| 2     | Strong/Essential fit   |

## Rubric Table

| Dimension                        | Score (0–2) | Notes/Justification                |
|-----------------------------------|-------------|------------------------------------|
| Multi-step/branching workflows    |             |                                    |
| Tool/function/API integration     |             |                                    |
| Agentic/autonomous behavior       |             |                                    |
| Persistent/shareable context      |             |                                    |
| Observability/metrics/audit       |             |                                    |
| High concurrency/scale            |             |                                    |
| Real-time/low-latency required    |             |                                    |
| Security/compliance needs         |             |                                    |
| Python ecosystem fit              |             |                                    |

**How to Use:**
- Add up the scores. (Max = 18)
- Interpretation:
  - 15–18: Excellent fit. Prioritize!
  - 10–14: Good fit. Worth considering, may need minor tweaks.
  - 5–9: Marginal fit. Proceed with caution or consider extensions.
  - 0–4: Poor fit. Likely not worth pursuing with current architecture.

## Example Evaluation

| Dimension                        | Score | Notes                                 |
|-----------------------------------|-------|---------------------------------------|
| Multi-step/branching workflows    | 2     | Research involves multiple steps      |
| Tool/function/API integration     | 2     | Needs web search, summarization, etc. |
| Agentic/autonomous behavior       | 2     | Needs to decide what to do next       |
| Persistent/shareable context      | 2     | Must track findings across steps      |
| Observability/metrics/audit       | 1     | Some logging needed                   |
| High concurrency/scale            | 1     | Not at first, but may grow            |
| Real-time/low-latency required    | 0     | Not critical                          |
| Security/compliance needs         | 1     | Some data privacy needed              |
| Python ecosystem fit              | 2     | Yes                                   |

**Total: 13/18** → Good fit, worth pursuing.

## Template (Copy/Paste for Each Opportunity)

```
### Opportunity: [Name/Description]

| Dimension                        | Score (0–2) | Notes/Justification                |
|-----------------------------------|-------------|------------------------------------|
| Multi-step/branching workflows    |             |                                    |
| Tool/function/API integration     |             |                                    |
| Agentic/autonomous behavior       |             |                                    |
| Persistent/shareable context      |             |                                    |
| Observability/metrics/audit       |             |                                    |
| High concurrency/scale            |             |                                    |
| Real-time/low-latency required    |             |                                    |
| Security/compliance needs         |             |                                    |
| Python ecosystem fit              |             |                                    |

**Total Score:** X/18

**Interpretation:** [Excellent/Good/Marginal/Poor fit] 