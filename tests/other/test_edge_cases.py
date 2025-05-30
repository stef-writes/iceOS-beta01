import pytest
from unittest.mock import AsyncMock, patch
from app.models.node_models import NodeConfig
from app.models.config import LLMConfig
from app.nodes.ai_node import AiNode
from app.utils.context import GraphContextManager
from app.services.llm_service import LLMService

# 1. Malformed Node Config (missing required field)
def test_missing_required_field():
    bad_config = {
        "id": "bad1",
        "type": "ai",
        "prompt": "Say hi",
        "level": 0,
        "dependencies": []
        # "model" is missing!
    }
    with pytest.raises(Exception):  # Use ValidationError if using Pydantic
        NodeConfig(**bad_config)

# 2. Circular Dependencies (API test, requires running server)
try:
    import requests
    @pytest.mark.skipif('requests' not in globals(), reason="requests not installed")
    def test_circular_dependency():
        node_a = {
            "id": "A",
            "type": "ai",
            "model": "gpt-4",
            "prompt": "A",
            "level": 0,
            "dependencies": ["B"]
        }
        node_b = {
            "id": "B",
            "type": "ai",
            "model": "gpt-4",
            "prompt": "B",
            "level": 0,
            "dependencies": ["A"]
        }
        chain = {"nodes": [node_a, node_b], "persist_intermediate_outputs": True}
        response = requests.post("http://localhost:8000/api/v1/chains/execute", json=chain)
        assert response.status_code == 400
        assert "circular" in response.text.lower()
except ImportError:
    pass

# 3. Node Output Missing Expected Key (scaffold)
def test_output_missing_key():
    # TODO: Implement with a mock or test LLM handler
    # Simulate a node config where output_schema expects 'result' but LLM returns just a string
    pass

def make_basic_node_config():
    return NodeConfig(
        id="test-node",
        type="ai",
        prompt="Say hi to {name}",
        model="gpt-3.5-turbo",
        level=0,
        dependencies=[],
        input_schema={"name": "str"},
        output_schema={"text": "str"},
        context_rules={},
        format_specifications={},
        input_selection=None,
        templates={},
        metadata=None,
        tools=None
    )

def make_llm_config():
    return LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="fake-key",
        max_context_tokens=2048,
        max_tokens=256
    )

def test_ainode_calls_llmservice(monkeypatch):
    node_config = make_basic_node_config()
    llm_config = make_llm_config()
    context_manager = GraphContextManager({})
    mock_llm_service = AsyncMock(spec=LLMService)
    mock_llm_service.generate.return_value = ("Hello, John!", None, None)
    node = AiNode(node_config, context_manager, llm_config, llm_service=mock_llm_service)
    context = {"name": "John"}
    # Run execute and assert LLMService.generate was called
    import asyncio
    asyncio.run(node.execute(context))
    mock_llm_service.generate.assert_called_once()

def test_ainode_llmservice_failure(monkeypatch):
    node_config = make_basic_node_config()
    llm_config = make_llm_config()
    context_manager = GraphContextManager({})
    mock_llm_service = AsyncMock(spec=LLMService)
    mock_llm_service.generate.side_effect = Exception("Simulated LLM failure")
    node = AiNode(node_config, context_manager, llm_config, llm_service=mock_llm_service)
    context = {"name": "John"}
    import asyncio
    result = asyncio.run(node.execute(context))
    assert not result.success
    assert "Simulated LLM failure" in result.error
