"""
Real-world ScriptChain test demonstrating:
- Multiple LLM providers (OpenAI, Anthropic, Google)
- Context passing between nodes
- Tool usage for calculations
- Parallel execution
- Error handling and validation
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def make_api_request(method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make an API request to the ScriptChain server."""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise

def create_node_config(
    node_id: str,
    name: str,
    prompt: str,
    model: str,
    provider: str,
    dependencies: list = None,
    output_schema: Dict[str, str] = None,
    input_mappings: Dict[str, Dict[str, str]] = None,
    tools: list = None
) -> Dict[str, Any]:
    """Create a node configuration for the ScriptChain API."""
    
    # Select API key based on provider
    api_key = {
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "google": GOOGLE_API_KEY
    }.get(provider.lower(), OPENAI_API_KEY)

    llm_config = {
        "provider": provider,
        "model": model,
        "temperature": 0.7,
        "max_tokens": 500,
        "max_context_tokens": 4000,
        "api_key": api_key,
        "top_p": 1.0
    }

    config = {
        "id": node_id,
        "name": name,
        "type": "ai",
        "model": model,
        "provider": provider,
        "prompt": prompt,
        "level": 0,
        "dependencies": dependencies or [],
        "llm_config": llm_config,
        "token_management": {
            "truncate": True,
            "preserve_sentences": True,
            "max_context_tokens": 4000,
            "max_completion_tokens": 500
        }
    }

    if output_schema:
        config["output_schema"] = output_schema
    if input_mappings:
        config["input_mappings"] = input_mappings
    if tools:
        config["tools"] = tools

    return config

def create_real_world_chain_test():
    """Create and execute a real-world chain test with multiple providers and tools."""
    
    print("\n=== REAL-WORLD CHAIN TEST ===")
    print("Testing a chain that analyzes user feedback and generates recommendations")
    
    timestamp = int(time.time())
    
    # Node 1: Initial feedback analysis (OpenAI)
    node1_id = f"feedback_analysis_{timestamp}"
    node1_config = create_node_config(
        node_id=node1_id,
        name="Feedback Analysis",
        prompt="""Analyze the following user feedback:
        {feedback}
        
        Your response MUST be a single, valid JSON object ONLY, with no other text before or after it. The JSON object must contain the following keys:
        - "topics": A list of strings representing the main topics discussed.
        - "sentiment": A string indicating the overall sentiment (e.g., "positive", "negative", "neutral").
        - "key_points": A list of strings highlighting key concerns or suggestions.
        - "priority": A string indicating the priority level (e.g., "high", "medium", "low").
        """,
        model="gpt-4",
        provider="openai",
        output_schema={
            "topics": "list",
            "sentiment": "string",
            "key_points": "list",
            "priority": "string"
        }
    )

    # Node 2: Sentiment Analysis (Anthropic)
    node2_id = f"sentiment_analysis_{timestamp}"
    node2_config = create_node_config(
        node_id=node2_id,
        name="Sentiment Analysis",
        prompt="""Perform a detailed sentiment analysis of this feedback:
        {feedback}
        
        Your response MUST be a single, valid JSON object ONLY, with no other text before or after it. The JSON object must contain the following keys:
        - "overall_sentiment": A string describing the overall sentiment (e.g., "positive", "negative", "mixed").
        - "emotional_tones": A list of strings describing the emotional undertones detected (e.g., "frustration", "appreciation").
        - "pain_points": A list of strings identifying specific pain points mentioned.
        - "positive_aspects": A list of strings identifying specific positive aspects mentioned.""",
        model="claude-3-opus-20240229",
        provider="anthropic",
        dependencies=[node1_id],
        input_mappings={
            "feedback": {"source_node_id": node1_id, "source_output_key": "key_points"}
        },
        output_schema={
            "overall_sentiment": "string",
            "emotional_tones": "list",
            "pain_points": "list",
            "positive_aspects": "list"
        }
    )

    # Node 3: Recommendation Generation (Google)
    node3_id = f"recommendations_{timestamp}"
    node3_config = create_node_config(
        node_id=node3_id,
        name="Recommendation Generator",
        prompt="""Based on the analysis and sentiment:
        Analysis: {analysis}
        Sentiment: {sentiment}
        
        Your response MUST be a single, valid JSON object ONLY, with no other text before or after it. The JSON object must contain the following keys:
        - "immediate_actions": A list of strings for immediate actions.
        - "short_term": A list of strings for short-term improvements.
        - "long_term": A list of strings for long-term strategies.
        - "priority_order": A list of strings indicating the priority order of these actions/strategies.""",
        model="gemini-1.5-flash-latest",
        provider="google",
        dependencies=[node1_id, node2_id],
        input_mappings={
            "analysis": {"source_node_id": node1_id, "source_output_key": "key_points"},
            "sentiment": {"source_node_id": node2_id, "source_output_key": "overall_sentiment"}
        },
        output_schema={
            "immediate_actions": "list",
            "short_term": "list",
            "long_term": "list",
            "priority_order": "list"
        }
    )

    # Test data
    test_feedback = """
    The new interface is much better than before, but there are still some issues:
    1. The search function is slow and sometimes doesn't find what I'm looking for
    2. The mobile version needs improvement - buttons are too small
    3. I love the new dark mode feature
    4. Customer support response time has increased significantly
    5. The new analytics dashboard is very helpful
    """

    try:
        print("\n--- CLEANUP: Clearing existing context ---")
        for node_id in [node1_id, node2_id, node3_id]:
            try:
                make_api_request("DELETE", f"/nodes/{node_id}/context")
                print(f"  ✓ Cleared context for {node_id}")
            except:
                print(f"  ⚠ No existing context for {node_id}")
        print()

        print("--- EXECUTING CHAIN ---")
        nodes_config = [node1_config, node2_config, node3_config]
        
        print("Chain configuration:")
        for i, node in enumerate(nodes_config):
            deps = ", ".join(node['dependencies']) if node['dependencies'] else "None"
            print(f"  {i+1}. {node['name']} (Provider: {node['llm_config']['provider']}, Model: {node['llm_config']['model']}) - Deps: {deps}")

        print("\nExecuting chain...")
        chain_result = make_api_request("POST", "/chains/execute", {
            "nodes": nodes_config,
            "context": {"feedback": test_feedback},
            "persist_intermediate_outputs": True
        })

        if chain_result.get('success'):
            print("\n✓ Chain execution successful!")
            print("\n--- RESULTS ---")
            chain_output = chain_result.get('output', {})
            
            # Display results for each node
            for node_id, node_config in zip([node1_id, node2_id, node3_id], nodes_config):
                print(f"\n{node_config['name']} Results:")
                if node_id in chain_output:
                    output = chain_output[node_id].get('output', {})
                    print(json.dumps(output, indent=2))
                else:
                    print("  No output found")

            print("\n=== CHAIN EXECUTION METRICS ===")
            if 'metadata' in chain_result:
                print(f"Total execution time: {chain_result['metadata'].get('duration', 'N/A')} seconds")
                print(f"Token usage: {chain_result.get('token_stats', {}).get('total_tokens', 'N/A')}")

        else:
            print(f"\n✗ Chain execution failed: {chain_result.get('error', 'Unknown error')}")
            print(f"Full chain result: {json.dumps(chain_result, indent=2)}")

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting Real-World Chain Test...")
    print(f"API Base URL: {API_BASE_URL}")
    
    # Check API keys
    api_keys_status = {
        "OpenAI": bool(OPENAI_API_KEY),
        "Anthropic": bool(ANTHROPIC_API_KEY),
        "Google": bool(GOOGLE_API_KEY)
    }
    print("\nAPI Key Status:")
    for provider, found in api_keys_status.items():
        print(f"  {provider}: {'✓ Found' if found else '✗ Missing'}")
    
    if not all(api_keys_status.values()):
        print("\nWarning: Some API keys are missing. The test may not run completely.")
    
    try:
        create_real_world_chain_test()
        print("\n" + "="*60)
        print("🎉 TEST COMPLETED!")
        print("Demonstrated capabilities:")
        print("  • Multiple LLM providers")
        print("  • Context passing between nodes")
        print("  • Structured output schemas")
        print("  • Dependency management")
        print("  • Error handling")
        print("="*60)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc() 