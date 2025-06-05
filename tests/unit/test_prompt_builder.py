import pytest
from app.nodes.prompt_builder import build_tool_preamble, prepare_prompt
from app.models.node_models import NodeConfig
from app.models.config import LLMConfig
from app.utils.context.manager import GraphContextManager
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

def test_build_tool_preamble_direct():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    tools = tool_service.list_tools_with_schemas()
    preamble = build_tool_preamble(tools)
    assert 'calculator' in preamble
    assert 'Adds two numbers' in preamble
    assert 'function_call' in preamble
    assert '"name": "calculator"' in preamble or '"name": "calculator"' in preamble.replace(' ', '')
    assert '"arguments": {"a": 2, "b": 3}' in preamble or '"arguments":{"a":2,"b":3}' in preamble.replace(' ', '')

def test_prepare_prompt_direct():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Calculate the sum of {a} and {b}.")
    prompt = prepare_prompt(config, GraphContextManager(), config.llm_config, tool_service, {'a': 2, 'b': 3})
    assert 'Calculate the sum of 2 and 3' in prompt
    assert 'calculator' in prompt
    assert 'function_call' in prompt 