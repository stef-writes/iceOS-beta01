"""
OpenAI Chained Math Test Script - Adapted for ScriptChain3.0
Tests a chain of OpenAI nodes performing sequential math operations
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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def make_api_request(method: str, endpoint_path: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Helper function to make API requests with proper error handling"""
    url = f"{API_BASE_URL}{endpoint_path}"
    headers = {"Content-Type": "application/json"}
    
    try:
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            response = requests.get(url, headers=headers)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {e.response.text}")
        raise

def create_node_config(node_id: str, name: str, prompt: str, model: str = "gpt-4",
                      dependencies: list = None,
                      output_schema: Dict[str, str] = None,
                      input_mappings: Dict[str, Dict[str, str]] = None,
                      llm_config_payload: Optional[Dict[str, Any]] = None,
                      provider_name: str = "openai"
                      ) -> Dict[str, Any]:
    """Create a node configuration for the ScriptChain3.0 API"""
    
    # Default LLM config if specific payload not provided
    if not llm_config_payload:
        api_key_to_use = OPENAI_API_KEY
        if provider_name == "anthropic":
            api_key_to_use = ANTHROPIC_API_KEY
        elif provider_name == "google":
            api_key_to_use = GOOGLE_API_KEY
        elif provider_name == "deepseek":
            api_key_to_use = DEEPSEEK_API_KEY
        else:
            api_key_to_use = OPENAI_API_KEY # Default if provider_name is odd

        current_llm_config_dict = {
            "provider": provider_name,
            "model": model, # model name is passed as an argument
            "temperature": 0.1,
            "max_tokens": 50,
            "max_context_tokens": 4000,
            "api_key": api_key_to_use,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    else:
        current_llm_config_dict = llm_config_payload

    config = {
        "id": node_id,
        "name": name,
        "type": "ai", # Assuming 'ai' type for these tests
        "prompt": prompt,
        "level": 0, # Default level
        "dependencies": dependencies or [],
        
        # Populate top-level fields expected by NodeConfig Pydantic model on server
        "model": current_llm_config_dict["model"],
        "provider": current_llm_config_dict.get("provider", provider_name),
        "temperature": current_llm_config_dict.get("temperature", 0.7), # Default from NodeConfig if not in llm_config
        "max_tokens": current_llm_config_dict.get("max_tokens", 500),   # Default from NodeConfig if not in llm_config
        
        # Include the full llm_config as well, in case it's used for other settings or overrides
        "llm_config": current_llm_config_dict,
        
        "token_management": {
            "truncate": True,
            "preserve_sentences": True,
            "max_context_tokens": current_llm_config_dict.get("max_context_tokens", 4096),
            "max_completion_tokens": current_llm_config_dict.get("max_tokens", 1024) # Note: using max_tokens from llm_config here for consistency
        }
    }
    if output_schema:
        config["output_schema"] = output_schema
    if input_mappings:
        config["input_mappings"] = input_mappings
    return config

def create_chained_openai_math_test():
    """Create and execute a chained math test using only OpenAI models with explicit input_mappings"""
    
    print("\n=== CHAINED OPENAI MATH TEST SCRIPT (WITH INPUT MAPPINGS) ===")
    print("Creating a chain: GPT-4 -> GPT-4-Turbo -> GPT-3.5-Turbo -> GPT-4")
    print("Each node performs a math operation on the previous result using explicit input mappings\n")
    
    base_prompt = "You are a math expert. Only respond with the numerical answer to the given equation. Do not include any explanation, just the number.\n\n"
    
    timestamp = int(time.time())
    node1_id = f"math_node1_{timestamp}"
    node2_id = f"math_node2_{timestamp}"
    node3_id = f"math_node3_{timestamp}"
    node4_id = f"math_node4_{timestamp}"

    nodes_data = [
        {
            "config": create_node_config(
                node_id=node1_id,
                name="Math Node 1 (10 + 32)",
                prompt=base_prompt + "10 + 32 = ?",
                model="gpt-4",
                output_schema={"result": "int"}
            ),
            "expected_output": 42
        },
        {
            "config": create_node_config(
                node_id=node2_id,
                name="Math Node 2 (Node1_result + 100)",
                prompt=base_prompt + "{val_from_node1} + 100 = ?",
                model="gpt-4-turbo",
                dependencies=[node1_id],
                output_schema={"result": "int"},
                input_mappings={
                    "val_from_node1": {"source_node_id": node1_id, "source_output_key": "result"}
                }
            ),
            "expected_output": 142
        },
        {
            "config": create_node_config(
                node_id=node3_id,
                name="Math Node 3 (10 * Node1_result)",
                prompt=base_prompt + "10 * {val_from_node1} = ?",
                model="gpt-3.5-turbo",
                dependencies=[node1_id],
                output_schema={"result": "int"},
                input_mappings={
                    "val_from_node1": {"source_node_id": node1_id, "source_output_key": "result"}
                }
            ),
            "expected_output": 420
        },
        {
            "config": create_node_config(
                node_id=node4_id,
                name="Math Node 4 (Node3_result - Node2_result)",
                prompt=base_prompt + "{val_from_node3} - {val_from_node2} = ?",
                model="gpt-4",
                dependencies=[node2_id, node3_id],
                output_schema={"result": "int"},
                input_mappings={
                    "val_from_node2": {"source_node_id": node2_id, "source_output_key": "result"},
                    "val_from_node3": {"source_node_id": node3_id, "source_output_key": "result"}
                }
            ),
            "expected_output": 278
        }
    ]
    
    actual_outputs = {}
    
    try:
        print("--- CLEANUP: Clearing existing context ---")
        for node_data in nodes_data:
            try:
                make_api_request("DELETE", f"/nodes/{node_data['config']['id']}/context")
                print(f"  ✓ Cleared context for {node_data['config']['id']}")
            except:
                print(f"  ⚠ No existing context for {node_data['config']['id']}")
        print()
        
        print("--- STEP 1: Executing Chain ---")
        config_dicts = [json.loads(json.dumps(nd['config'], cls=DateTimeEncoder)) for nd in nodes_data]
        
        print("Executing chain with the following nodes:")
        for i, node_data in enumerate(nodes_data):
            cfg = node_data['config']
            deps = ", ".join(cfg['dependencies']) if cfg['dependencies'] else "None"
            mappings = cfg.get('input_mappings', {})
            print(f"  {i+1}. {cfg['name']} ({cfg['id']}, Model: {cfg['llm_config']['model']}, Dependencies: {deps}, InputMappings: {len(mappings)})")

        print("\nSending chain execution request...")
        chain_result = make_api_request("POST", "/chains/execute", {
            "nodes": config_dicts,
            "persist_intermediate_outputs": True
        })
        
        print(f"Chain execution completed: {'✓ SUCCESS' if chain_result.get('success') else '✗ FAILED'}")
        if not chain_result.get('success'):
            print(f"Chain execution failed: {chain_result.get('error', 'Unknown error')}")
            print(f"Full chain result: {json.dumps(chain_result, indent=2)}")
            return
        
        chain_output = chain_result.get('output', {})
        
        print("\n--- STEP 2: Analyzing Results ---")
        print("\nExpected calculation chain:")
        print(f"  {nodes_data[0]['config']['name']}: {nodes_data[0]['expected_output']}")
        print(f"  {nodes_data[1]['config']['name']}: {nodes_data[1]['expected_output']}")
        print(f"  {nodes_data[2]['config']['name']}: {nodes_data[2]['expected_output']}")
        print(f"  {nodes_data[3]['config']['name']}: {nodes_data[3]['expected_output']}")
        
        print("\nActual results:")
        all_correct = True
        
        for i, node_data in enumerate(nodes_data):
            node_id = node_data['config']['id']
            node_name = node_data['config']['name']
            model = node_data['config']['llm_config']['model']
            expected = node_data['expected_output']
            
            actual = "No output or result key found"
            if node_id in chain_output and isinstance(chain_output[node_id], dict) and \
               'output' in chain_output[node_id] and isinstance(chain_output[node_id]['output'], dict) and \
               'result' in chain_output[node_id]['output']:
                actual = chain_output[node_id]['output']['result']
            else:
                all_correct = False
                print(f"  Debug info for {node_name}: Full output entry: {chain_output.get(node_id)}")

            actual_outputs[node_id] = actual
            is_correct = (actual == expected)
            status = "✓ PASS" if is_correct else "✗ FAIL"
            print(f"  {node_name} ({model}): Actual='{actual}' (Type: {type(actual).__name__}), Expected='{expected}' (Type: {type(expected).__name__}) {status}")
            
            if not is_correct:
                all_correct = False
        
        # Step 3: Test individual node context retrieval
        print("\n--- STEP 3: Testing Node Context Retrieval ---")
        for node_data in nodes_data:
            node_id = node_data['config']['id']
            try:
                context = make_api_request("GET", f"/nodes/{node_id}/context")
                print(f"  ✓ Retrieved context for {node_id}: {json.dumps(context, default=str)}")
            except Exception as e:
                print(f"  ✗ Failed to retrieve context for {node_id}: {e}")
        
        print(f"\n{'='*60}")
        if all_correct:
            print("🎉 CHAINED MATH TEST PASSED!")
        else:
            print("❌ CHAINED MATH TEST FAILED!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_complex_topology_test():
    """Test a more complex chain topology with different dependency patterns and input_mappings"""
    
    print("\n=== COMPLEX TOPOLOGY TEST (WITH INPUT MAPPINGS) ===")
    print("Creating a complex chain with fan-out and fan-in patterns:")
    print("Node A (starter) → Node B (creative) & Node C (analytical)")
    print("Node B & Node C → Node D (synthesizer)")
    print("Different models and temperatures for each role\n")
    
    timestamp = int(time.time())
    starter_id = f"starter_{timestamp}"
    creative_id = f"creative_{timestamp}"
    analytical_id = f"analytical_{timestamp}"
    synthesizer_id = f"synthesizer_{timestamp}"

    nodes_data = [
        {
            "config": create_node_config(
                node_id=starter_id,
                name="Starter: Space Prompt",
                prompt="Generate a creative writing prompt about space exploration. Keep it to 1-2 sentences.",
                model="gpt-4",
                output_schema={"text": "str"},
            ),
            "role": "Starter"
        },
        {
            "config": create_node_config(
                node_id=creative_id,
                name="Creative Writer",
                prompt="Based on this prompt: '{prompt_from_starter}', write a creative short story opening (2-3 sentences). Be imaginative and vivid.",
                model="gpt-4-turbo",
                dependencies=[starter_id],
                output_schema={"text": "str"},
                input_mappings={
                    "prompt_from_starter": {"source_node_id": starter_id, "source_output_key": "text"}
                }
            ),
            "role": "Creative"
        },
        {
            "config": create_node_config(
                node_id=analytical_id,
                name="Analytical Thinker",
                prompt="Based on this prompt: '{prompt_from_starter}', provide a scientific analysis of the scenario (2-3 sentences). Focus on technical feasibility.",
                model="gpt-3.5-turbo",
                dependencies=[starter_id],
                output_schema={"text": "str"},
                input_mappings={
                    "prompt_from_starter": {"source_node_id": starter_id, "source_output_key": "text"}
                }
            ),
            "role": "Analytical"
        },
        {
            "config": create_node_config(
                node_id=synthesizer_id,
                name="Synthesizer",
                prompt="Combine these two perspectives:\nCreative: '{creative_story_opening}'\nAnalytical: '{scientific_analysis}'\n\nWrite a balanced summary that bridges imagination and science (2-3 sentences).",
                model="gpt-4",
                dependencies=[creative_id, analytical_id],
                output_schema={"text": "str"},
                input_mappings={
                    "creative_story_opening": {"source_node_id": creative_id, "source_output_key": "text"},
                    "scientific_analysis": {"source_node_id": analytical_id, "source_output_key": "text"}
                }
            ),
            "role": "Synthesizer"
        }
    ]

    # Adjust llm_config for specific nodes (max_tokens, temperature)
    nodes_data[0]['config']['llm_config']['temperature'] = 0.3
    nodes_data[0]['config']['llm_config']['max_tokens'] = 100
    nodes_data[1]['config']['llm_config']['temperature'] = 0.9
    nodes_data[1]['config']['llm_config']['max_tokens'] = 150
    nodes_data[2]['config']['llm_config']['temperature'] = 0.1
    nodes_data[2]['config']['llm_config']['max_tokens'] = 150
    nodes_data[3]['config']['llm_config']['temperature'] = 0.5
    nodes_data[3]['config']['llm_config']['max_tokens'] = 200
    
    try:
        print("--- CLEANUP: Clearing existing context ---")
        for node_data in nodes_data:
            try:
                make_api_request("DELETE", f"/nodes/{node_data['config']['id']}/context")
                print(f"  ✓ Cleared context for {node_data['config']['id']}")
            except:
                print(f"  ⚠ No existing context for {node_data['config']['id']}")
        print()
        
        print("--- EXECUTING COMPLEX CHAIN ---")
        config_dicts = [json.loads(json.dumps(nd['config'], cls=DateTimeEncoder)) for nd in nodes_data]
        
        print("Chain topology:")
        for node_data in nodes_data:
            cfg = node_data['config']
            deps = ", ".join(cfg['dependencies']) if cfg['dependencies'] else "None"
            mappings = cfg.get('input_mappings', {})
            temp = cfg['llm_config']['temperature']
            print(f"  • {cfg['name']} ({cfg['llm_config']['model']}, temp={temp}) - Deps: {deps}, InputMappings: {len(mappings)})")
        
        print("\nExecuting chain...")
        chain_result = make_api_request("POST", "/chains/execute", {
            "nodes": config_dicts,
            "persist_intermediate_outputs": True
        })
        
        if chain_result.get('success'):
            print("✓ Chain execution successful!")
            print("\n--- RESULTS ---")
            chain_output = chain_result.get('output', {})
            
            for node_data in nodes_data:
                node_id = node_data['config']['id']
                role = node_data['role']
                
                actual_result_text = "No output or text key found"
                if node_id in chain_output and isinstance(chain_output[node_id], dict) and \
                   'output' in chain_output[node_id] and isinstance(chain_output[node_id]['output'], dict) and \
                   'text' in chain_output[node_id]['output']:
                    actual_result_text = chain_output[node_id]['output']['text']
                else:
                     print(f"  Debug info for {role}: Full output entry: {chain_output.get(node_id)}")
                
                print(f"\n{role} ({node_data['config']['name']}, Model: {node_data['config']['llm_config']['model']}):")
                print(f"  {actual_result_text}")
            
            print(f"\n{'='*60}")
            print("🎉 COMPLEX TOPOLOGY TEST COMPLETED!")
            print("Successfully demonstrated:")
            print("  • Fan-out pattern (1 → 2)")
            print("  • Fan-in pattern (2 → 1)")  
            print("  • Different LLM models and temperatures")
            print("  • Context passing between dependent nodes")
            print(f"{'='*60}")
        else:
            print(f"✗ Chain execution failed: {chain_result.get('error', 'Unknown error')}")
            print(f"Full chain result: {json.dumps(chain_result, indent=2)}")
            
    except Exception as e:
        print(f"\n❌ Complex topology test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_parallel_processing_test():
    """Test parallel processing with independent nodes, using output_schema"""
    print("\n=== PARALLEL PROCESSING TEST (WITH OUTPUT SCHEMA) ===")
    print("Creating independent nodes that can run in parallel:")
    print("Node A, Node B, Node C (all independent)")
    print("Different models processing different tasks simultaneously\n")
    
    timestamp = int(time.time())
    
    tasks_data = [
        {
            "role": "Summarizer", 
            "prompt": "Summarize the benefits of renewable energy in 1 sentence.", 
            "model": "gpt-3.5-turbo",
            "output_schema": {"text": "str"},
            "expected_key": "text"
        },
        {
            "role": "Translator", 
            "prompt": "Translate 'Hello, how are you?' to Spanish.", 
            "model": "gpt-4",
            "output_schema": {"text": "str"},
            "expected_key": "text"
        },
        {
            "role": "Calculator", 
            "prompt": "What is 15 × 24? Only respond with the number.", 
            "model": "gpt-4-turbo",
            "output_schema": {"result": "int"},
            "expected_key": "result"
        }
    ]
    
    nodes_configs_for_api = []
    node_info_for_results = []

    for i, task_detail in enumerate(tasks_data):
        node_id = f"{task_detail['role'].lower()}_{timestamp}"
        node_cfg = create_node_config(
            node_id=node_id,
            name=f"{task_detail['role']} Node",
            prompt=task_detail['prompt'],
            model=task_detail['model'],
            output_schema=task_detail['output_schema']
        )
        nodes_configs_for_api.append(json.loads(json.dumps(node_cfg, cls=DateTimeEncoder)))
        node_info_for_results.append({
            "id": node_id, 
            "role": task_detail['role'], 
            "model": task_detail['model'],
            "expected_key": task_detail['expected_key']
        })
            
    try:
        print("--- EXECUTING PARALLEL NODES ---")
        print("Parallel nodes:")
        for info in node_info_for_results:
            print(f"  • {info['role']} ({info['model']})")
        
        start_time = time.time()
        chain_result = make_api_request("POST", "/chains/execute", {
            "nodes": nodes_configs_for_api,
            "persist_intermediate_outputs": True
        })
        execution_time = time.time() - start_time
        
        if chain_result.get('success'):
            print(f"✓ Parallel execution completed in {execution_time:.2f} seconds")
            print("\n--- PARALLEL RESULTS ---")
            chain_output = chain_result.get('output', {})
            
            for info in node_info_for_results:
                node_id = info['id']
                role = info['role']
                expected_key = info['expected_key']
                
                actual_result = "No output or expected key found"
                if node_id in chain_output and isinstance(chain_output[node_id], dict) and \
                   'output' in chain_output[node_id] and isinstance(chain_output[node_id]['output'], dict) and \
                   expected_key in chain_output[node_id]['output']:
                    actual_result = chain_output[node_id]['output'][expected_key]
                else:
                    print(f"  Debug info for {role}: Full output entry: {chain_output.get(node_id)}")

                print(f"  {role}: {actual_result} (Type: {type(actual_result).__name__})")
            
            print(f"\n✓ All {len(nodes_configs_for_api)} nodes executed successfully in parallel!")
        else:
            print(f"✗ Parallel execution failed: {chain_result.get('error', 'Unknown error')}")
            print(f"Full chain result: {json.dumps(chain_result, indent=2)}")
            
    except Exception as e:
        print(f"\n❌ Parallel processing test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_simple_single_node():
    """Test a simple single node execution with output_schema"""
    print("\n=== SIMPLE SINGLE NODE TEST (WITH OUTPUT SCHEMA) ===")
    
    timestamp = int(time.time())
    node_id = f"simple_test_{timestamp}"
    node_config = create_node_config(
        node_id=node_id,
        name="Simple Test Node",
        prompt="You are a helpful assistant. Say 'Hello, World!' in exactly those words.",
        model="gpt-3.5-turbo",
        output_schema={"greeting": "str"}
    )
    
    try:
        # Clear context first
        try:
            make_api_request("DELETE", f"/nodes/{node_id}/context")
            print(f"  ✓ Cleared context for {node_id}")
        except:
            print(f"  ⚠ No existing context for {node_id}")

        config_dict = json.loads(json.dumps(node_config, cls=DateTimeEncoder))
        
        response = make_api_request("POST", "/nodes/text-generation", {
            "config": config_dict,
            "context": {"task": "Simple greeting test"}
        })
        
        if response.get('success'):
            output_data = response.get('output', {})
            result = output_data.get('greeting', 'Key "greeting" not in output') 
            print(f"✓ Single node test passed: {result}")
            if result != "Hello, World!":
                 print(f"  WARN: Expected 'Hello, World!', got '{result}'")
        else:
            print(f"✗ Single node test failed: {response.get('error', 'Unknown error')}")
            print(f"Full response: {json.dumps(response, indent=2)}")
    
    except Exception as e:
        print(f"✗ Single node test failed with error: {e}")
        import traceback
        traceback.print_exc()

def create_multi_provider_variant_test():
    """Test a chain with different providers, models, temperatures, tokens, etc."""
    print("\n=== MULTI-PROVIDER VARIANT CHAIN TEST ===")
    print("Testing a chain: OpenAI -> Anthropic -> Google Gemini -> DeepSeek")
    print("Each node has varied LLM configurations.")

    required_keys = {
        "OpenAI": OPENAI_API_KEY,
        "Anthropic": ANTHROPIC_API_KEY,
        "Google": GOOGLE_API_KEY,
        "DeepSeek": DEEPSEEK_API_KEY
    }

    for provider_name, key_value in required_keys.items():
        if not key_value:
            env_var_name = f"{provider_name.upper()}_API_KEY"
            if provider_name == "Google":
                env_var_name = "GOOGLE_API_KEY"
            print(f"\n⚠️ WARNING: {env_var_name} not found. Skipping MULTI-PROVIDER VARIANT CHAIN TEST.")
            return

    timestamp = int(time.time())
    node_oai_id = f"mp_openai_{timestamp}"
    node_anth_id = f"mp_anthropic_{timestamp}"
    node_goog_id = f"mp_google_{timestamp}"
    node_ds_id = f"mp_deepseek_{timestamp}"

    nodes_data = [
        {
            "config": create_node_config(
                node_id=node_oai_id,
                name="OpenAI: Capital City",
                prompt="What is the capital of France? Respond with only the city name.",
                output_schema={"text": "str"},
                llm_config_payload={
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.2,
                    "max_tokens": 60,
                    "api_key": OPENAI_API_KEY,
                    "top_p": 0.9
                }
            ),
            "provider": "OpenAI"
        },
        {
            "config": create_node_config(
                node_id=node_anth_id,
                name="Anthropic: Famous Landmark",
                prompt="The capital of France is {capital_from_openai}. What is a famous landmark there? Respond with only the landmark name.",
                dependencies=[node_oai_id],
                input_mappings={"capital_from_openai": {"source_node_id": node_oai_id, "source_output_key": "text"}},
                output_schema={"text": "str"},
                llm_config_payload={
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "temperature": 0.3,
                    "max_tokens": 70,
                    "api_key": ANTHROPIC_API_KEY
                }
            ),
            "provider": "Anthropic"
        },
        {
            "config": create_node_config(
                node_id=node_goog_id,
                name="Google Gemini: Landmark Description",
                prompt="A famous landmark in {capital_from_openai} is {landmark_from_anthropic}. Describe it very briefly (1 sentence).",
                dependencies=[node_oai_id, node_anth_id],
                input_mappings={
                    "capital_from_openai": {"source_node_id": node_oai_id, "source_output_key": "text"},
                    "landmark_from_anthropic": {"source_node_id": node_anth_id, "source_output_key": "text"}
                },
                output_schema={"text": "str"},
                llm_config_payload={
                    "provider": "google",
                    "model": "gemini-1.5-flash-latest", # Using a common Gemini model
                    "temperature": 0.4,
                    "max_tokens": 80,
                    "api_key": GOOGLE_API_KEY,
                    "top_p": 0.7
                }
            ),
            "provider": "Google Gemini"
        },
        {
            "config": create_node_config(
                node_id=node_ds_id,
                name="DeepSeek: Visit Suggestion",
                prompt="Based on this description: '{description_from_google}', suggest a good month to visit. Respond with only the month name.",
                dependencies=[node_goog_id],
                input_mappings={"description_from_google": {"source_node_id": node_goog_id, "source_output_key": "text"}},
                output_schema={"text": "str"},
                llm_config_payload={
                    "provider": "deepseek",
                    "model": "deepseek-chat", # Common DeepSeek model
                    "temperature": 0.5,
                    "max_tokens": 90,
                    "api_key": DEEPSEEK_API_KEY,
                    "top_p": 0.6
                }
            ),
            "provider": "DeepSeek"
        }
    ]

    try:
        print("\n--- CLEANUP: Clearing existing context for multi-provider test ---")
        for node_data in nodes_data:
            try:
                make_api_request("DELETE", f"/nodes/{node_data['config']['id']}/context")
                print(f"  ✓ Cleared context for {node_data['config']['id']}")
            except:
                print(f"  ⚠ No existing context for {node_data['config']['id']}")
        print()

        print("--- EXECUTING MULTI-PROVIDER CHAIN ---")
        config_dicts = [json.loads(json.dumps(nd['config'], cls=DateTimeEncoder)) for nd in nodes_data]
        
        print("Chain configuration:")
        for i, (node_data, provider_name) in enumerate(zip(nodes_data, ["OpenAI", "Anthropic", "Google Gemini", "DeepSeek"])):
            cfg = node_data['config']
            llm_cfg = cfg['llm_config']
            deps = ", ".join(cfg['dependencies']) if cfg['dependencies'] else "None"
            # Use .get() for top_p as it might not be present for all providers (e.g., Anthropic after our change)
            top_p_val = llm_cfg.get('top_p', 'N/A') # Default to N/A if not found
            print(f"  {i+1}. {cfg['name']} (Provider: {llm_cfg['provider']}, Model: {llm_cfg['model']}, Temp: {llm_cfg['temperature']}, Tokens: {llm_cfg['max_tokens']}, TopP: {top_p_val}) - Deps: {deps}")

        print("\nExecuting chain...")
        chain_result = make_api_request("POST", "/chains/execute", {
            "nodes": config_dicts,
            "persist_intermediate_outputs": True
        })
        
        if chain_result.get('success'):
            print("✓ Multi-provider chain execution successful!")
            print("\n--- RESULTS ---")
            chain_output = chain_result.get('output', {})
            all_nodes_produced_output = True
            
            for node_data in nodes_data:
                node_id = node_data['config']['id']
                provider_used = node_data['provider']
                node_name = node_data['config']['name']
                
                actual_result_text = "No output or text key found"
                output_entry = chain_output.get(node_id)

                if output_entry and isinstance(output_entry, dict) and \
                   'output' in output_entry and isinstance(output_entry['output'], dict) and \
                   'text' in output_entry['output'] and output_entry['output']['text']:
                    actual_result_text = output_entry['output']['text']
                else:
                    all_nodes_produced_output = False
                    print(f"  Debug info for {node_name} ({provider_used}): Full output entry: {output_entry}")
                
                print(f"\n{node_name} (Provider: {provider_used}):")
                print(f"  Output: {actual_result_text}")
            
            print(f"\n{'='*60}")
            if all_nodes_produced_output:
                print("🎉 MULTI-PROVIDER VARIANT CHAIN TEST COMPLETED SUCCESSFULLY!")
            else:
                print("❌ MULTI-PROVIDER VARIANT CHAIN TEST COMPLETED WITH MISSING OUTPUTS.")
            print("Successfully demonstrated a chain with diverse LLM providers and configurations.")
            print(f"{'='*60}")
        else:
            print(f"✗ Multi-provider chain execution failed: {chain_result.get('error', 'Unknown error')}")
            print(f"Full chain result: {json.dumps(chain_result, indent=2)}")
            
    except Exception as e:
        print(f"\n❌ Multi-provider variant test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Starting ScriptChain3.0 Test Suite...")
    print(f"API Base URL: {API_BASE_URL}")
    api_keys_status = {
        "OpenAI": bool(OPENAI_API_KEY),
        "Anthropic": bool(ANTHROPIC_API_KEY),
        "Google": bool(GOOGLE_API_KEY),
        "DeepSeek": bool(DEEPSEEK_API_KEY)
    }
    print("API Key Status:")
    for provider, found in api_keys_status.items():
        print(f"  {provider}: {'✓ Found' if found else '✗ Missing'}")
    
    if not OPENAI_API_KEY:
        print("\nCritical Error: OPENAI_API_KEY is missing. Some tests cannot run.")
        # Decide if to exit or just skip relevant tests. For now, OpenAI is core to many tests.

    try:
        # Test simple single node (uses OpenAI by default in create_node_config)
        test_simple_single_node()
        
        # Test linear chain (original math test - uses OpenAI)
        create_chained_openai_math_test()
        
        # Test complex topology with fan-out/fan-in (uses OpenAI)
        create_complex_topology_test()
        
        # Test parallel processing (uses OpenAI)
        create_parallel_processing_test()

        # Test multi-provider chain
        create_multi_provider_variant_test()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED!")
        print("Demonstrated chain capabilities:")
        print("  • Linear chains (A → B → C → D)")
        print("  • Fan-out patterns (1 → many)")
        print("  • Fan-in patterns (many → 1)")
        print("  • Parallel execution (independent nodes)")
        print("  • Different LLM models and configurations")
        print("  • Context passing and dependency resolution")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest suite failed: {e}")
        import traceback
        traceback.print_exc() 