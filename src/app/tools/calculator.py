from pydantic import BaseModel
from app.tools.base import BaseTool

class CalculatorParams(BaseModel):
    a: float
    b: float

class CalculatorOutput(BaseModel):
    result: float

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Adds two numbers."
    parameters_schema = CalculatorParams
    output_schema = CalculatorOutput
    usage_example = '{"function_call": {"name": "calculator", "arguments": {"a": 2, "b": 3}}}'

    def run(self, a: float, b: float) -> dict:
        result = a + b
        # If both are ints and result is an int, return int
        if isinstance(a, int) and isinstance(b, int) and result == int(result):
            result = int(result)
        return {"result": result} 