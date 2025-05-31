"""
Test philosophical essay chain via API server (so you can see server logs).
"""

import requests
import time
from typing import Dict, Any


def test_philosophical_essay_via_api():
    """Test the philosophical essay chain via HTTP API calls."""
    
    base_url = "http://localhost:8000"
    
    # Create the chain configuration
    chain_config = {
        "nodes": [
            {
                "id": "philosophical_generator",
                "name": "Philosophical Statement Generator",
                "type": "ai",
                "model": "gpt-4",
                "provider": "openai",
                "prompt": "Generate a single, thought-provoking philosophical statement about the nature of reality, consciousness, or existence. Make it concise but profound.",
                "level": 0,
                "dependencies": [],
                "input_mappings": {},
                "output_schema": {"text": "str"},
                "llm_config": {
                    "model": "gpt-4",
                    "provider": "openai",
                    "temperature": 0.8,
                    "max_tokens": 150
                }
            },
            {
                "id": "axiom_analyzer",
                "name": "Axiom Analyzer", 
                "type": "ai",
                "model": "deepseek-chat",
                "provider": "deepseek",
                "prompt": "What are the fundamental axioms underlying this philosophical statement: '{statement}'. List 3-5 core axioms that this statement assumes to be true.",
                "level": 1,
                "dependencies": ["philosophical_generator"],
                "input_mappings": {
                    "statement": {
                        "source_node_id": "philosophical_generator",
                        "source_output_key": "text"
                    }
                },
                "output_schema": {"text": "str"},
                "llm_config": {
                    "model": "deepseek-chat", 
                    "provider": "deepseek",
                    "temperature": 0.7,
                    "max_tokens": 300
                }
            },
            {
                "id": "assumption_analyzer",
                "name": "Assumption Analyzer",
                "type": "ai", 
                "model": "claude-3-5-sonnet-20241022",
                "provider": "anthropic",
                "prompt": "What are the key assumptions in this philosophical statement: '{statement}'. Identify implicit beliefs, presuppositions, and unstated premises.",
                "level": 1,
                "dependencies": ["philosophical_generator"],
                "input_mappings": {
                    "statement": {
                        "source_node_id": "philosophical_generator",
                        "source_output_key": "text"
                    }
                },
                "output_schema": {"text": "str"},
                "llm_config": {
                    "model": "claude-3-5-sonnet-20241022",
                    "provider": "anthropic", 
                    "temperature": 0.6,
                    "max_tokens": 300
                }
            },
            {
                "id": "sentiment_analyzer",
                "name": "Sentiment Analyzer",
                "type": "ai",
                "model": "gpt-4", 
                "provider": "openai",
                "prompt": "Analyze the emotional tone and sentiment of this philosophical statement: '{statement}'. Consider its optimism/pessimism, certainty/uncertainty, and overall philosophical mood.",
                "level": 1,
                "dependencies": ["philosophical_generator"],
                "input_mappings": {
                    "statement": {
                        "source_node_id": "philosophical_generator",
                        "source_output_key": "text"
                    }
                },
                "output_schema": {"text": "str"},
                "llm_config": {
                    "model": "gpt-4",
                    "provider": "openai",
                    "temperature": 0.5,
                    "max_tokens": 200
                }
            },
            {
                "id": "outline_creator",
                "name": "Essay Outline Creator",
                "type": "ai",
                "model": "gpt-4",
                "provider": "openai", 
                "prompt": "Create a detailed outline for an essay analyzing this philosophical statement: '{statement}'\\n\\nBased on these analyses:\\nAxioms: {axioms}\\nAssumptions: {assumptions}\\nSentiment: {sentiment}\\n\\nProvide only a structured outline with main points and sub-points.",
                "level": 2,
                "dependencies": ["philosophical_generator", "axiom_analyzer", "assumption_analyzer", "sentiment_analyzer"],
                "input_mappings": {
                    "statement": {
                        "source_node_id": "philosophical_generator",
                        "source_output_key": "text"
                    },
                    "axioms": {
                        "source_node_id": "axiom_analyzer",
                        "source_output_key": "text"
                    },
                    "assumptions": {
                        "source_node_id": "assumption_analyzer", 
                        "source_output_key": "text"
                    },
                    "sentiment": {
                        "source_node_id": "sentiment_analyzer",
                        "source_output_key": "text"
                    }
                },
                "output_schema": {"text": "str"},
                "llm_config": {
                    "model": "gpt-4",
                    "provider": "openai",
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            },
            {
                "id": "essay_writer",
                "name": "Essay Writer",
                "type": "ai",
                "model": "gpt-4",
                "provider": "openai",
                "prompt": "Using this outline: '{outline}'\\n\\nWrite a comprehensive essay analyzing the philosophical statement: '{statement}'\\n\\nIncorporate these analyses:\\nAxioms: {axioms}\\nAssumptions: {assumptions}\\nSentiment: {sentiment}\\n\\nWrite a well-structured, academic-style essay with introduction, body paragraphs, and conclusion.",
                "level": 3,
                "dependencies": ["philosophical_generator", "axiom_analyzer", "assumption_analyzer", "sentiment_analyzer", "outline_creator"],
                "input_mappings": {
                    "statement": {
                        "source_node_id": "philosophical_generator", 
                        "source_output_key": "text"
                    },
                    "axioms": {
                        "source_node_id": "axiom_analyzer",
                        "source_output_key": "text"
                    },
                    "assumptions": {
                        "source_node_id": "assumption_analyzer",
                        "source_output_key": "text"
                    },
                    "sentiment": {
                        "source_node_id": "sentiment_analyzer",
                        "source_output_key": "text"
                    },
                    "outline": {
                        "source_node_id": "outline_creator",
                        "source_output_key": "text"
                    }
                },
                "output_schema": {"text": "str"},
                "llm_config": {
                    "model": "gpt-4",
                    "provider": "openai",
                    "temperature": 0.8,
                    "max_tokens": 1500
                }
            }
        ]
    }
    
    print("🚀 Making API request to execute philosophical essay chain...")
    print(f"📡 Server: {base_url}")
    print(f"📊 Nodes: {len(chain_config['nodes'])}")
    
    start_time = time.time()
    
    # Make the API request
    response = requests.post(
        f"{base_url}/api/v1/chains/execute",
        json=chain_config,
        timeout=120  # 2 minute timeout
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
        print(f"❌ Chain failed: {result.get('error')}")
        return
        
    print("✅ Chain execution successful!")
    
    # Print results with real-time preview
    output = result.get("output", {})
    print("\n" + "="*80)
    print("PHILOSOPHICAL ESSAY CHAIN RESULTS (VIA API)")
    print("="*80)
    
    # Level 0
    if "philosophical_generator" in output:
        print(f"\n🎯 Level 0 - Philosophical Statement:")
        print(f"📝 {output['philosophical_generator']['output']['text']}")
    
    # Level 1 (Parallel)
    print("\n🎯 Level 1 - Analysis (Parallel):")
    if "axiom_analyzer" in output:
        print(f"\n📊 Axioms (DeepSeek):")
        print(f"📝 {output['axiom_analyzer']['output']['text']}")
    if "assumption_analyzer" in output:
        print(f"\n📊 Assumptions (Claude):")
        print(f"📝 {output['assumption_analyzer']['output']['text']}")
    if "sentiment_analyzer" in output:
        print(f"\n📊 Sentiment (GPT-4):")
        print(f"📝 {output['sentiment_analyzer']['output']['text']}")
    
    # Level 2
    if "outline_creator" in output:
        print("\n🎯 Level 2 - Outline:")
        print(f"📝 {output['outline_creator']['output']['text']}")
    
    # Level 3
    if "essay_writer" in output:
        print("\n🎯 Level 3 - Final Essay:")
        print(f"📝 {output['essay_writer']['output']['text']}")
    
    # Print token stats
    token_stats = result.get("token_stats", {})
    print(f"\n📊 Total tokens: {token_stats.get('total_tokens', 0)}")
    print(f"💰 Provider usage: {token_stats.get('provider_usage', {})}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_philosophical_essay_via_api() 