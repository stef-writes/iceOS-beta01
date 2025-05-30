# Tools Directory

This directory contains all custom tools/functions that can be invoked by AiNode via LLM function calling.

## Conventions
- Each tool should be defined as a class inheriting from `BaseTool` (see `base.py`).
- Each tool must define:
  - `name`: Unique string identifier for the tool.
  - `description`: Short description of what the tool does.
  - `parameters_schema`: JSON schema dict describing the tool's parameters (for LLM function calling).
  - `run(**kwargs)`: Method to execute the tool logic. Must be implemented by each tool.

## Example
```python
from app.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful."
    parameters_schema = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input string"}
        },
        "required": ["input"]
    }

    def run(self, input: str):
        # Tool logic here
        return {"result": input.upper()}
```

## Registration
- Tools can be imported and registered in a central registry for discovery and invocation by AiNode.
