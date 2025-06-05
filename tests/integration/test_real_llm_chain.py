import os
import requests
import json
from dotenv import load_dotenv
import time

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def make_api_request(method, endpoint, data=None):
    url = f"{API_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    resp = requests.request(method, url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()

def skip_if_missing(key, name):
    if not key:
        print(f"[SKIP] {name} API key not found. Skipping test.")
        return True
    return False

def test_single_openai_node():
    if skip_if_missing(OPENAI_API_KEY, "OpenAI"): return
    node_config = {
        "id": f"openai_test_{int(time.time())}",
        "type": "ai",
        "model": "gpt-4.1",
        "prompt": "Respond with only: {\"text\": \"Paris\"}",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4.1",
            "api_key": OPENAI_API_KEY
        },
        "output_schema": {"text": "str"}
    }
    payload = {"config": node_config, "context": {}}
    result = make_api_request("POST", "/nodes/text-generation", payload)
    print("Single OpenAI node result:", result)
    assert result["success"]
    assert "Paris".lower() in result["output"]["text"].lower()

def test_chain_with_tool():
    # Use ToolNode for deterministic tool execution
    node_config = {
        "id": f"calc_test_{int(time.time())}",
        "type": "tool",
        "name": "calculator",  # Must match the registered tool name
        "model": "tool",      # Required by schema, value doesn't matter for ToolNode
        "prompt": "Run calculator",  # Required by schema, not used by ToolNode
        "input_schema": {"a": "int", "b": "int"},
        "output_schema": {"result": "int"},
        "level": 0,
        "dependencies": []
    }
    payload = {"nodes": [node_config], "context": {"a": 7, "b": 8}, "persist_intermediate_outputs": True}
    result = make_api_request("POST", "/chains/execute", payload)
    print("Chain with tool result (ToolNode):", result)
    assert result["success"]
    node_id = node_config["id"]
    assert result["output"][node_id]["output"]["result"] == 15

def test_multi_provider_chain():
    if skip_if_missing(OPENAI_API_KEY, "OpenAI"): return
    if skip_if_missing(ANTHROPIC_API_KEY, "Anthropic"): return
    if skip_if_missing(GOOGLE_API_KEY, "Google"): return
    if skip_if_missing(DEEPSEEK_API_KEY, "DeepSeek"): return
    timestamp = int(time.time())
    node_oai_id = f"mp_openai_{timestamp}"
    node_anth_id = f"mp_anthropic_{timestamp}"
    node_goog_id = f"mp_google_{timestamp}"
    node_ds_id = f"mp_deepseek_{timestamp}"
    nodes = [
        {
            "id": node_oai_id,
            "type": "ai",
            "model": "gpt-4.1",
            "prompt": "What is the capital of France? Respond with only the city name.",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4.1",
                "api_key": OPENAI_API_KEY
            },
            "output_schema": {"text": "str"}
        },
        {
            "id": node_anth_id,
            "type": "ai",
            "model": "claude-3-haiku-20240307",
            "prompt": "The capital of France is {capital_from_openai}. What is a famous landmark there? Respond with only the landmark name.",
            "dependencies": [node_oai_id],
            "input_mappings": {"capital_from_openai": {"source_node_id": node_oai_id, "source_output_key": "text"}},
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "api_key": ANTHROPIC_API_KEY
            },
            "output_schema": {"text": "str"}
        },
        {
            "id": node_goog_id,
            "type": "ai",
            "model": "gemini-1.5-flash-latest",
            "prompt": "A famous landmark in {capital_from_openai} is {landmark_from_anthropic}. Describe it very briefly (1 sentence).",
            "dependencies": [node_oai_id, node_anth_id],
            "input_mappings": {
                "capital_from_openai": {"source_node_id": node_oai_id, "source_output_key": "text"},
                "landmark_from_anthropic": {"source_node_id": node_anth_id, "source_output_key": "text"}
            },
            "llm_config": {
                "provider": "google",
                "model": "gemini-1.5-flash-latest",
                "api_key": GOOGLE_API_KEY
            },
            "output_schema": {"text": "str"}
        },
        {
            "id": node_ds_id,
            "type": "ai",
            "model": "deepseek-chat",
            "prompt": "Based on this description: '{description_from_google}', suggest a good month to visit. Respond with only the month name.",
            "dependencies": [node_goog_id],
            "input_mappings": {"description_from_google": {"source_node_id": node_goog_id, "source_output_key": "text"}},
            "llm_config": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": DEEPSEEK_API_KEY
            },
            "output_schema": {"text": "str"}
        }
    ]
    payload = {"nodes": nodes, "context": {}, "persist_intermediate_outputs": True}
    result = make_api_request("POST", "/chains/execute", payload)
    print("Multi-provider chain result:", json.dumps(result, indent=2))
    assert result["success"]
    # Optionally, add more detailed assertions for each node's output

def test_context_management():
    if skip_if_missing(OPENAI_API_KEY, "OpenAI"): return
    node_id = f"context_test_{int(time.time())}"
    node_config = {
        "id": node_id,
        "type": "ai",
        "model": "gpt-4.1",
        "prompt": "Say hello to {name}.",
        "llm_config": {
            "provider": "openai",
            "model": "gpt-4.1",
            "api_key": OPENAI_API_KEY
        },
        "output_schema": {"text": "str"}
    }
    payload = {"config": node_config, "context": {"name": "Alice"}}
    result = make_api_request("POST", "/nodes/text-generation", payload)
    print("Context management node result:", result)
    assert result["success"]
    # Now retrieve context
    context = make_api_request("GET", f"/nodes/{node_id}/context")
    print("Retrieved context:", context)
    assert context["text"].lower().startswith("hello")

if __name__ == "__main__":
    print("\n=== REAL LLM INTEGRATION TESTS ===\n")
    test_single_openai_node()
    test_chain_with_tool()
    test_multi_provider_chain()
    test_context_management()
    print("\n=== ALL REAL LLM TESTS COMPLETED ===\n") 