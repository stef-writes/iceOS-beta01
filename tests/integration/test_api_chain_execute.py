import pytest
from fastapi.testclient import TestClient
from app.main import app

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

def test_chain_execute_with_tool():
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
        "nodes": [node_config],
        "context": {"a": 2, "b": 3},
        "persist_intermediate_outputs": True
    }
    response = client.post("/api/v1/chains/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    print(data)  # Debug output
    assert data["success"]
    assert data["output"]["test-node"]["output"]["result"] == 5 