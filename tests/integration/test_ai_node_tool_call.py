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
    async def generate(self, llm_config, prompt, context=None, tools=None):
        # Simulate a function_call response from the LLM
        return (
            '{"function_call": {"name": "calculator", "arguments": {"a": 2, "b": 3}}}',
            None,
            None
        )

@pytest.mark.asyncio
async def test_ai_node_tool_call():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Calculate the sum of {a} and {b}.")
    node = AiNode(config, GraphContextManager(), config.llm_config, llm_service=MockLLMService(), tool_service=tool_service)
    result: NodeExecutionResult = await node.execute({'a': 2, 'b': 3})
    assert result.success
    assert result.output['result'] == 5
    assert result.error is None 