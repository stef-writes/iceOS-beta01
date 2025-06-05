import pytest
import os
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

@pytest.mark.asyncio
async def test_output_schema_validation_success():
    node_config = {
        "id": "dummy",
        "type": "ai",
        "model": "gpt-4",
        "provider": "openai",
        "prompt": "Respond with only the number 42. Do not add any words, punctuation, or explanation. Output only: 42",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "output_schema": {"result": "int"},
        "input_schema": {},
        "level": 0,
        "dependencies": [],
        "templates": {},
        "input_mappings": {},
        "context_rules": {},
        "format_specifications": {},
        "token_management": {
            "truncate": True,
            "preserve_sentences": True,
            "max_context_tokens": 4096,
            "max_completion_tokens": 1024
        }
    }
    payload = {
        "config": node_config,
        "context": {}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/nodes/text-generation", json=payload)
    result = response.json()
    assert result["success"]
    assert isinstance(result["output"]["result"], int)

@pytest.mark.asyncio
async def test_output_schema_validation_failure():
    node_config = {
        "id": "dummy",
        "type": "ai",
        "model": "gpt-4.1",
        "provider": "openai",
        "prompt": "Respond with only the string 'hello'. Do not add any numbers, punctuation, or explanation.",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4.1",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "output_schema": {"result": "int"},
        "input_schema": {},
        "level": 0,
        "dependencies": [],
        "templates": {},
        "input_mappings": {},
        "context_rules": {},
        "format_specifications": {},
        "token_management": {
            "truncate": True,
            "preserve_sentences": True,
            "max_context_tokens": 4096,
            "max_completion_tokens": 1024
        }
    }
    payload = {
        "config": node_config,
        "context": {}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/nodes/text-generation", json=payload)
    result = response.json()
    assert not result["success"]
    # Accept either a schema validation error or a max agentic steps error
    error_msg = (result["error"] or '').lower()
    assert ("validation" in error_msg or "failed" in error_msg or "max agentic steps" in error_msg) 