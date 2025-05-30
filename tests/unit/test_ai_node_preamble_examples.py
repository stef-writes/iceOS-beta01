import pytest
from app.nodes.ai_node import AiNode, TOOL_INSTRUCTION
from app.models.node_models import NodeConfig
from app.models.config import LLMConfig
from app.utils.context import GraphContextManager
from app.services.tool_service import ToolService
from app.tools.calculator import CalculatorTool

class DummyConfig(NodeConfig):
    model: str = "dummy-model"
    prompt: str = "Calculate the sum of {a} and {b}."
    input_selection: None = None
    context_rules: dict = {}
    format_specifications: dict = {}
    tools: None = None
    input_schema: None = None
    output_schema: None = None
    templates: None = None
    metadata: None = None
    id: str = "dummy"
    name: str = "dummy"
    type: str = "ai"

def test_build_tool_preamble_with_usage_example():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Calculate the sum of {a} and {b}.")
    node = AiNode(config, GraphContextManager(), config.llm_config, tool_service=tool_service)
    preamble = node.build_tool_preamble(tool_service.list_tools_with_schemas())
    # Check instruction is present
    assert TOOL_INSTRUCTION in preamble
    # Check tool name and description
    assert 'calculator' in preamble
    assert 'Adds two numbers' in preamble
    # Check usage example is present (minified JSON)
    assert '{"function_call": {"name": "calculator", "arguments": {"a": 2, "b": 3}}}' in preamble 