import pytest
import os
import httpx
from unittest.mock import AsyncMock, patch
from app.models.node_models import NodeConfig
from app.models.config import LLMConfig
from app.nodes.ai_node import AiNode
from app.utils.context.manager import GraphContextManager
from app.services.llm_service import LLMService

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

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
@pytest.mark.asyncio
async def test_circular_dependency():
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
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/chains/execute", json=chain)
    assert response.status_code == 400
    assert "circular" in response.text.lower()

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

@pytest.mark.asyncio
async def test_ainode_calls_llmservice():
    node_config = {
        "id": "test-node",
        "type": "ai",
        "prompt": "Respond with only: Hi {name}. Do not add any extra words, punctuation, or explanation.",
        "model": "gpt-4",
        "output_schema": {"text": "str"},
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "input_schema": {"name": "str"},
        "context_rules": {},
        "format_specifications": {},
        "input_selection": None,
        "templates": {},
        "metadata": None,
        "tools": None
    }
    payload = {
        "config": node_config,
        "context": {"name": "John"}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/nodes/text-generation", json=payload)
    result = response.json()
    assert result["success"]
    assert "john" in result["output"]["text"].lower()

@pytest.mark.asyncio
async def test_ainode_llmservice_failure():
    # This test expects failure if no API key is available in config or environment.
    # If the environment variable OPENAI_API_KEY is set, the app will succeed (intended behavior).
    node_config = {
        "id": "test-node",
        "type": "ai",
        "prompt": "Say hi to {name}",
        "model": "gpt-4",
        "output_schema": {"text": "str"},
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4"
            # No API key
        },
        "input_schema": {"name": "str"},
        "context_rules": {},
        "format_specifications": {},
        "input_selection": None,
        "templates": {},
        "metadata": None,
        "tools": None
    }
    payload = {
        "config": node_config,
        "context": {"name": "John"}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/nodes/text-generation", json=payload)
    result = response.json()
    if os.getenv("OPENAI_API_KEY"):
        assert result["success"]
    else:
        assert not result["success"]
        assert "missing" in (result["error"] or "").lower() or "api key" in (result["error"] or "").lower()
