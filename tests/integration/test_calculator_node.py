"""
Test a single node using the calculator tool with different providers.
"""

import os
import sys
import requests
import time
from typing import Dict, Any

# Add src to sys.path so 'from app...' imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from app.models.config import ModelProvider


def test_calculator_node(provider: ModelProvider = ModelProvider.OPENAI):
    """Test a single node that uses the calculator tool with the specified provider."""
    
    base_url = "http://localhost:8000"
    
    # Map providers to their appropriate models
    model_map = {
        ModelProvider.OPENAI: "gpt-4",
        ModelProvider.ANTHROPIC: "claude-3-opus-20240229",
        ModelProvider.DEEPSEEK: "deepseek-chat"
    }
    
    # Create a single node configuration that uses the calculator
    node_config = {
        "nodes": [
            {
                "id": "math_solver",
                "name": "Math Problem Solver",
                "type": "ai",
                "model": model_map[provider],
                "provider": provider.value,  # Use enum value for API
                "prompt": "Solve this math problem: What is 42 + 1337? Use the calculator tool to get the exact answer. After you get the result from the calculator, explain the answer in a complete sentence and DO NOT call the calculator again.",
                "level": 0,
                "dependencies": [],
                "input_mappings": {},
                "output_schema": {"text": "str"},
                "llm_config": {
                    "model": model_map[provider],
                    "provider": provider.value,  # Use enum value for API
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            }
        ]
    }
    
    print(f"\n🚀 Testing calculator with {provider.value.upper()} provider...")
    print(f"📡 Server: {base_url}")
    
    start_time = time.time()
    
    # Make the API request
    response = requests.post(
        f"{base_url}/api/v1/chains/execute",
        json=node_config,
        timeout=30  # 30 second timeout
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️  Total time: {duration:.2f} seconds")
    print(f"📈 Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return
        
    result = response.json()
    
    if not result.get("success"):
        print(f"❌ Node failed: {result.get('error')}")
        return
        
    print("✅ Node execution successful!")
    
    # Print results
    output = result.get("output", {})
    print("\n" + "="*80)
    print(f"CALCULATOR NODE RESULTS ({provider.value.upper()})")
    print("="*80)
    
    if "math_solver" in output:
        print(f"\n📝 Generated response:")
        print(f"{output['math_solver']['output']['text']}")
    
    # Print token stats
    token_stats = result.get("token_stats", {})
    print(f"\n📊 Total tokens: {token_stats.get('total_tokens', 0)}")
    print(f"💰 Provider usage: {token_stats.get('provider_usage', {})}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test with each provider
    for provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.DEEPSEEK]:
        test_calculator_node(provider) 