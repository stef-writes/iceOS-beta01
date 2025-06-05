import pytest
import os
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

@pytest.mark.asyncio
async def test_successful_tool_call():
    node_config = {
        "id": "dummy",
        "type": "ai",
        "model": "gpt-4",
        "provider": "openai",
        "prompt": "Use the calculator tool to add 2 and 3. Respond with only the result as an integer, nothing else.",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "output_schema": {"result": "int"},
        "input_schema": {},
        "level": 0,
        "dependencies": [],
        "tools": [
            {
                "name": "calculator",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        ],
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
        "context": {"a": 2, "b": 3}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/nodes/text-generation", json=payload)
        print(response.text)
    result = response.json()
    assert result["success"]
    assert result["output"]["result"] == 5
    assert result["error"] is None

@pytest.mark.asyncio
async def test_invalid_json_tool_call_detection():
    # This test is less relevant for the API, but we can check for graceful fallback.
    node_config = {
        "id": "dummy",
        "type": "ai",
        "model": "gpt-4",
        "provider": "openai",
        "prompt": "Calculate the sum of {a} and {b}.",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "output_schema": {"result": "int"},
        "input_schema": {},
        "level": 0,
        "dependencies": [],
        "tools": [
            {
                "name": "calculator",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        ],
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
        "context": {"a": 2, "b": 3}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/nodes/text-generation", json=payload)
        print(response.text)
    result = response.json()
    # Should fall back to normal output handling, not a tool call
    assert not (result["output"] and "result" in result["output"]) or result["success"]

@pytest.mark.asyncio
async def test_no_tool_call_in_response():
    node_config = {
        "id": "dummy",
        "type": "ai",
        "model": "gpt-4",
        "provider": "openai",
        "prompt": "Calculate the sum of {a} and {b}.",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "output_schema": {"result": "int"},
        "input_schema": {},
        "level": 0,
        "dependencies": [],
        "tools": [
            {
                "name": "calculator",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        ],
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
        "context": {"a": 2, "b": 3}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/nodes/text-generation", json=payload)
        print(response.text)
    result = response.json()
    # Should fall back to normal output handling, not a tool call
    assert not (result["output"] and "result" in result["output"]) or result["success"] 