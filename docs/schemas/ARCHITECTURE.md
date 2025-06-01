# iceOS: Current Implementation Architecture

## Core Components

### 1. Node System
Currently implemented with:
- `BaseNode`: Abstract base class defining node interface
- `AiNode`: Concrete implementation for AI-powered nodes
- Node Factory: Creates appropriate node instances based on type

```python
# Current node types
node_types = {
    "ai": "AiNode",  # Implemented
    "tool": "ToolNode",  # Planned
    "router": "RouterNode"  # Planned
}
```

### 2. LLM Integration
Implemented providers:
- OpenAI
- Anthropic
- Google Gemini
- DeepSeek

Each provider has:
- Handler class implementing `BaseLLMHandler`
- Configuration management
- API key handling
- Error handling

### 3. Context Management
`GraphContextManager` provides:
- Context storage and retrieval
- Dependency tracking
- Token management
- File-based persistence

### 4. Tool System
Current implementation:
- `BaseTool` abstract class
- Tool registration system
- Parameter validation
- Output schema support

### 5. Chain System
`ScriptChain` orchestrates:
- Node execution
- Dependency management
- Parallel processing
- Error handling

## Data Flow

1. **API Layer** (`main.py`)
   - FastAPI application
   - Environment configuration
   - API key management
   - CORS handling

2. **Node Execution**
   ```python
   # Current flow
   API Request -> ScriptChain -> Node Factory -> AiNode -> LLM Handler -> Response
   ```

3. **Context Management**
   ```python
   # Context flow
   Node Execution -> GraphContextManager -> Context Store -> Next Node
   ```

## Current Limitations

1. **Node Types**
   - Only `AiNode` is fully implemented
   - `ToolNode` and `RouterNode` are planned

2. **Tool System**
   - Basic tool framework in place
   - Limited tool implementations
   - Need more specialized tools

3. **UI/UX**
   - No visual interface yet
   - API-only interaction
   - Limited monitoring capabilities

## Next Steps

1. **Immediate**
   - Implement remaining node types
   - Add more specialized tools
   - Enhance error handling
   - Improve monitoring

2. **Short-term**
   - Add visual interface
   - Implement workflow editor
   - Add debugging tools
   - Enhance security

3. **Long-term**
   - Spatial computing support
   - Quantum readiness
   - Autonomous optimization
   - Advanced security features

## Technical Details

### Node Configuration
```python
node_config = {
    "metadata": {
        "node_id": "unique_id",
        "node_type": "ai",
        "name": "node_name",
        "description": "node_description"
    },
    "input_schema": {
        "field_name": "type_string"
    },
    "output_schema": {
        "field_name": "type_string"
    },
    "templates": {
        "system": "template_string",
        "user": "template_string"
    }
}
```

### LLM Configuration
```python
llm_config = {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "optional_key",
    "max_tokens": 4000,
    "temperature": 0.7
}
```

### Tool Definition
```python
tool_config = {
    "name": "tool_name",
    "description": "tool_description",
    "parameters_schema": {
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string",
                "description": "param_description"
            }
        }
    }
}
```

## Security Considerations

1. **API Key Management**
   - Environment variable based
   - Per-node configuration
   - Secure storage

2. **Context Security**
   - File-based storage
   - Locking mechanism
   - Version tracking

3. **Execution Security**
   - Input validation
   - Output validation
   - Error handling

## Monitoring and Metrics

1. **Current Capabilities**
   - Basic logging
   - Token usage tracking
   - Execution timing

2. **Planned Features**
   - Performance metrics
   - Cost tracking
   - Usage analytics
   - Health monitoring

## Development Guidelines

1. **Code Structure**
   - Modular design
   - Clear interfaces
   - Type hints
   - Documentation

2. **Testing**
   - Unit tests
   - Integration tests
   - Performance tests

3. **Documentation**
   - Code comments
   - API documentation
   - Architecture docs
   - User guides 