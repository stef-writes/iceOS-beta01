from pydantic import BaseModel
from typing import Type, Optional

class BaseTool:
    """Base class for all tools. Tools should inherit from this and implement the required interface."""
    name: str = "base_tool"
    description: str = "Base tool class. Override in subclasses."
    parameters_schema: Type[BaseModel] = None  # Must be set in subclass
    output_schema: Optional[Type[BaseModel]] = None  # Optional
    usage_example: str = ""  # Example of how to call this tool (in JSON)

    def __init__(self):
        if self.parameters_schema is None:
            raise NotImplementedError("Tool must define a Pydantic parameters_schema class.")

    def run(self, **kwargs):
        """Execute the tool with the given parameters. Must be implemented by subclasses."""
        raise NotImplementedError("Tool must implement the run() method.")

    @classmethod
    def get_parameters_json_schema(cls) -> dict:
        if cls.parameters_schema is None:
            raise NotImplementedError("Tool must define a parameters_schema.")
        return cls.parameters_schema.schema()

    @classmethod
    def get_output_json_schema(cls) -> Optional[dict]:
        if cls.output_schema is not None:
            return cls.output_schema.schema()
        return None
