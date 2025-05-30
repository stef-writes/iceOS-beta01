import pytest
from app.nodes.ai_node import AiNode
from app.models.node_models import NodeConfig, NodeExecutionResult
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

class MockLLMService:
    def __init__(self, response):
        self.response = response
    async def generate(self, llm_config, prompt, context=None, tools=None):
        return (self.response, None, None)

@pytest.mark.asyncio
async def test_valid_tool_call_detection():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Calculate the sum of {a} and {b}.")
    llm_service = MockLLMService('{"function_call": {"name": "calculator", "arguments": {"a": 2, "b": 3}}}')
    node = AiNode(config, GraphContextManager(), config.llm_config, llm_service=llm_service, tool_service=tool_service)
    result: NodeExecutionResult = await node.execute({'a': 2, 'b': 3})
    assert result.success
    assert result.output['result'] == 5

@pytest.mark.asyncio
async def test_invalid_json_tool_call_detection():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Calculate the sum of {a} and {b}.")
    llm_service = MockLLMService('not a json')
    node = AiNode(config, GraphContextManager(), config.llm_config, llm_service=llm_service, tool_service=tool_service)
    result: NodeExecutionResult = await node.execute({'a': 2, 'b': 3})
    # Should fall back to normal output handling, not a tool call
    assert not (result.output and 'result' in result.output)

@pytest.mark.asyncio
async def test_no_tool_call_in_response():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Calculate the sum of {a} and {b}.")
    llm_service = MockLLMService('{"some_other_key": 123}')
    node = AiNode(config, GraphContextManager(), config.llm_config, llm_service=llm_service, tool_service=tool_service)
    result: NodeExecutionResult = await node.execute({'a': 2, 'b': 3})
    # Should fall back to normal output handling, not a tool call
    assert not (result.output and 'result' in result.output) 