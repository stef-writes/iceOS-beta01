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
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
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

def create_complex_mixed_provider_chain_test():
    """Test a complex chain with fan-out/fan-in, mixed providers, and different configs."""
    print("\n=== COMPLEX MIXED-PROVIDER CHAIN TEST (FAN-OUT/FAN-IN) ===")
    print("Testing: NodeA -> (NodeB, NodeC, NodeD) -> NodeE")

    timestamp = int(time.time())

    # --- Node Configurations --- #

    # Node A: Initial Data Generation (e.g., OpenAI)
    # Output: A list of three distinct concepts or items.
    node_a_id = f"initial_generator_A_{timestamp}"
    node_a_config = create_node_config(
        node_id=node_a_id,
        name="Initial Concept Generator (OpenAI)",
        prompt="""Generate three distinct, one-word technical concepts (e.g., 'authentication', 'database', 'API').
        Your response MUST be a single, valid JSON object ONLY, with a single key "concepts" which is a list of three strings.""",
        model="gpt-3.5-turbo", # Using a faster model for this simple task
        provider="openai",
        output_schema={"concepts": "list"}
    )

    # Fan-out Nodes (processing items from Node A)

    # Node B: Process Concept 1 (e.g., Anthropic)
    node_b_id = f"process_concept1_B_{timestamp}"
    node_b_config = create_node_config(
        node_id=node_b_id,
        name="Define Concept 1 (Anthropic)",
        prompt="""Provide a concise, one-sentence definition for the concept: {concept1}.
        Your response MUST be a single, valid JSON object ONLY, with a single key "definition" which is a string.""",
        model="claude-3-haiku-20240307", # Using a faster/cheaper Claude model
        provider="anthropic",
        dependencies=[node_a_id],
        input_mappings={"concept1": {"source_node_id": node_a_id, "source_output_key": "concepts.0"}}, # Assuming concepts is a list
        output_schema={"definition": "string"}
    )

    # Node C: Process Concept 2 (e.g., Google Gemini)
    node_c_id = f"process_concept2_C_{timestamp}"
    node_c_config = create_node_config(
        node_id=node_c_id,
        name="Use Case for Concept 2 (Google)",
        prompt="""Describe a primary use case for the concept: {concept2}.
        Your response MUST be a single, valid JSON object ONLY, with a single key "use_case" which is a string.""",
        model="gemini-1.5-flash-latest",
        provider="google",
        dependencies=[node_a_id],
        input_mappings={"concept2": {"source_node_id": node_a_id, "source_output_key": "concepts.1"}},
        output_schema={"use_case": "string"}
    )

    # Node D: Process Concept 3 (e.g., DeepSeek)
    # Ensure you have DEEPSEEK_API_KEY in your .env for this to work
    node_d_id = f"related_tech_concept3_D_{timestamp}"
    node_d_config = create_node_config(
        node_id=node_d_id,
        name="Related Tech for Concept 3 (DeepSeek)",
        prompt="""List one related technology for the concept: {concept3}.
        Your response MUST be a single, valid JSON object ONLY, with a single key "related_technology" which is a string.""",
        model="deepseek-chat", # Or another valid DeepSeek model name
        provider="deepseek",
        dependencies=[node_a_id],
        input_mappings={"concept3": {"source_node_id": node_a_id, "source_output_key": "concepts.2"}},
        output_schema={"related_technology": "string"}
    )

    # Fan-in Node

    # Node E: Synthesize Information (e.g., OpenAI GPT-4 for better synthesis)
    node_e_id = f"synthesizer_E_{timestamp}"
    node_e_config = create_node_config(
        node_id=node_e_id,
        name="Synthesize Concepts Report (OpenAI GPT-4)",
        prompt="""Compile a brief report from the following information:
        Concept 1 Definition: {definition1}
        Concept 2 Use Case: {use_case2}
        Concept 3 Related Technology: {related_tech3}
        Your response MUST be a single, valid JSON object ONLY, with a single key "report" which is a string summarizing all inputs.""",
        model="gpt-4",
        provider="openai",
        dependencies=[node_b_id, node_c_id, node_d_id],
        input_mappings={
            "definition1": {"source_node_id": node_b_id, "source_output_key": "definition"},
            "use_case2": {"source_node_id": node_c_id, "source_output_key": "use_case"},
            "related_tech3": {"source_node_id": node_d_id, "source_output_key": "related_technology"}
        },
        output_schema={"report": "string"}
    )

    nodes_data = [
        node_a_config, node_b_config, node_c_config, node_d_config, node_e_config
    ]

    # Add DEEPSEEK_API_KEY check similar to others if you intend to run this frequently
    # For now, assuming it's set if the user wants to test DeepSeek.

    try:
        print("\n--- CLEANUP: Clearing existing context for complex chain ---")
        for node_config in nodes_data:
            node_id = node_config['id']
            try:
                make_api_request("DELETE", f"/nodes/{node_id}/context")
                print(f"  ✓ Cleared context for {node_id}")
            except:
                print(f"  ⚠ No existing context for {node_id}")
        print()

        print("--- EXECUTING COMPLEX CHAIN ---")
        print("Chain configuration:")
        for i, node in enumerate(nodes_data):
            deps = ", ".join(node['dependencies']) if node['dependencies'] else "None"
            prov = node.get('provider', node['llm_config']['provider']) # Handle provider key location
            mod = node.get('model', node['llm_config']['model']) # Handle model key location
            print(f"  {i+1}. {node['name']} (Provider: {prov}, Model: {mod}) - Deps: {deps}")
        print("\nExecuting chain...")

        chain_result = make_api_request("POST", "/chains/execute", {
            "nodes": nodes_data,
            # No initial global context needed as Node A generates it.
            "persist_intermediate_outputs": True 
        })

        if chain_result.get('success'):
            print("\n✓ Complex chain execution successful!")
            print("\n--- RESULTS OF COMPLEX CHAIN ---")
            chain_output = chain_result.get('output', {})
            
            for node_config in nodes_data:
                node_id = node_config['id']
                node_name = node_config['name']
                print(f"\n{node_name} (ID: {node_id}) Results:")
                if node_id in chain_output and chain_output[node_id].get('success'):
                    node_actual_output = chain_output[node_id].get('output', {})
                    print(json.dumps(node_actual_output, indent=2))
                elif node_id in chain_output:
                    print(f"  Node failed: {chain_output[node_id].get('error')}")
                else:
                    print("  No output found for this node in chain result.")

            print("\n=== COMPLEX CHAIN EXECUTION METRICS ===")
            if 'metadata' in chain_result:
                print(f"Total execution time: {chain_result['metadata'].get('duration', 'N/A')} seconds")
            if 'token_stats' in chain_result and chain_result['token_stats']:
                print(f"Total token usage: {chain_result['token_stats'].get('total_tokens', 'N/A')}")
                print(f"Provider breakdown: {json.dumps(chain_result['token_stats'].get('provider_usage', {}), indent=2)}")

        else:
            print(f"\n✗ Complex chain execution failed: {chain_result.get('error', 'Unknown error')}")
            # Print full output for debugging if the chain itself failed
            print("Full chain result for failed execution:")
            print(json.dumps(chain_result, indent=2))

    except Exception as e:
        print(f"\nComplex chain test failed with an exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting ScriptChain Test Suite...")
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
        create_complex_mixed_provider_chain_test()
        print("\n" + "="*60)
        print("🎉 ALL SCRIPTCHAIN TESTS COMPLETED! 🎉")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc() 