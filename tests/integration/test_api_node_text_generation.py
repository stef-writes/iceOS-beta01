import pytest
from fastapi.testclient import TestClient
from app.main import app
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

client = TestClient(app)

@pytest.fixture(autouse=True)
def patch_llm_service(monkeypatch):
    from app.services.llm_service import LLMService
    call_count = {"n": 0}
    async def mock_generate(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ('{"function_call": {"name": "calculator", "arguments": {"a": 2, "b": 3}}}', None, None)
        else:
            return ("The result is 5.", None, None)
    monkeypatch.setattr(LLMService, "generate", mock_generate)

def test_text_generation_node_with_tool():
    node_config = {
        "id": "test-node",
        "name": "Test Node",
        "type": "ai",
        "model": "dummy-model",
        "prompt": "Calculate the sum of {a} and {b}.",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "api_key": "fake"
        },
        "output_schema": {"result": "int"}
    }
    payload = {
        "config": node_config,
        "context": {"a": 2, "b": 3}
    }
    response = client.post("/api/v1/nodes/text-generation", json=payload)
    data = response.json()
    print(data)  # Debug output
    assert data["success"]
    assert data["output"]["result"] == 5 