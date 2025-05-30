import pytest
from app.nodes.ai_node import AiNode
from app.models.node_models import NodeConfig, NodeExecutionResult
from app.models.config import LLMConfig
from app.utils.context import GraphContextManager
from app.services.tool_service import ToolService
from app.tools.calculator import CalculatorTool

class DummyConfig(NodeConfig):
    model: str = "dummy-model"
    prompt: str = "First, add {a} and {b}. Then, add the result to {c}."
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

class MultiStepMockLLMService:
    def __init__(self):
        self.step = 0
        self.last_result = None
    async def generate(self, llm_config, prompt, context=None, tools=None):
        self.step += 1
        if self.step == 1:
            # First, ask to add a + b
            return (
                '{"function_call": {"name": "calculator", "arguments": {"a": 2, "b": 3}}}',
                None,
                None
            )
        elif self.step == 2:
            # Then, add the previous result to c
            # We'll parse the prompt to extract the last tool output
            import re, json
            match = re.search(r"Tool 'calculator' output: (\{.*?\})", prompt)
            prev_result = 0
            if match:
                prev_result = json.loads(match.group(1))["result"]
            return (
                f'{{"function_call": {{"name": "calculator", "arguments": {{"a": {prev_result}, "b": 4}}}}}}',
                None,
                None
            )
        else:
            # Finally, return a normal answer
            return ("The final result is 9.", None, None)

@pytest.mark.asyncio
async def test_ai_node_agentic_loop():
    tool_service = ToolService()
    tool_service.register_tool(CalculatorTool())
    config = DummyConfig(llm_config=LLMConfig(provider='openai', model='gpt-3.5-turbo', api_key='fake'), prompt="First, add {a} and {b}. Then, add the result to {c}.")
    node = AiNode(config, GraphContextManager(), config.llm_config, llm_service=MultiStepMockLLMService(), tool_service=tool_service)
    result: NodeExecutionResult = await node.execute({'a': 2, 'b': 3, 'c': 4}, max_steps=5)
    assert result.success
    assert '9' in str(result.output.values()).lower() or '9' in str(result.output).lower() 