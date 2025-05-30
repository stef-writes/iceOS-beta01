import pytest
from app.nodes.ai_node import AiNode
from app.models.node_models import NodeConfig, NodeExecutionResult
from app.models.config import LLMConfig
from app.utils.context import GraphContextManager
from app.services.tool_service import ToolService
from app.tools.calculator import CalculatorTool

class DummyConfig(NodeConfig):
    model: str = "dummy-model"
    prompt: str = "Return a number."
    input_selection: None = None
    context_rules: dict = {}
    format_specifications: dict = {}
    tools: None = None
    input_schema: None = None
    output_schema: dict = {'result': 'int'}
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
async def test_output_schema_validation_success():
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Return a number.")
    node = AiNode(config, GraphContextManager(), config.llm_config, llm_service=MockLLMService('42'))
    result: NodeExecutionResult = await node.execute({})
    assert result.success
    assert result.output['result'] == 42

@pytest.mark.asyncio
async def test_output_schema_validation_failure():
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="Return a number.")
    node = AiNode(config, GraphContextManager(), config.llm_config, llm_service=MockLLMService('not a number'))
    result: NodeExecutionResult = await node.execute({})
    assert not result.success
    assert 'validation' in (result.error or '').lower() or 'failed' in (result.error or '').lower() 