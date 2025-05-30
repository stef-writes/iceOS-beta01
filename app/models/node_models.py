"""
Data models for node configurations and metadata
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, validator
from datetime import datetime
from .config import LLMConfig, MessageTemplate, ModelProvider
from enum import Enum

class ContextFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    CUSTOM = "custom"

class InputMapping(BaseModel):
    """Mapping configuration for node inputs"""
    source_node_id: str = Field(..., description="Source node ID (UUID of the dependency)")
    source_output_key: str = Field(..., description="Key from the source node's output object to use (e.g., 'text', 'result')")
    rules: Dict[str, Any] = Field(default_factory=dict, description="Optional mapping/transformation rules (currently unused)")

class ContextRule(BaseModel):
    """Rule for handling context in a node"""
    include: bool = Field(default=True, description="Whether to include this context")
    format: ContextFormat = Field(default=ContextFormat.TEXT, description="Format of the context")
    required: bool = Field(default=False, description="Whether this context is required")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens allowed for this context")
    truncate: bool = Field(default=True, description="Whether to truncate if over token limit")

class NodeMetadata(BaseModel):
    """Metadata model for node versioning and ownership"""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Type of node (ai)")
    version: str = Field("1.0.0", pattern=r"^\d+\.\d+\.\d+$",
                        description="Semantic version of node configuration")
    owner: Optional[str] = Field(None, description="Node owner/maintainer")
    created_at: datetime = Field(default_factory=datetime.utcnow,
                               description="Creation timestamp")
    modified_at: datetime = Field(default_factory=datetime.utcnow,
                                description="Last modification timestamp")
    description: Optional[str] = Field(None, description="Description of the node")
    error_type: Optional[str] = Field(None, description="Type of error if execution failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow,
                               description="Execution timestamp")
    start_time: Optional[datetime] = Field(None, description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    duration: Optional[float] = Field(None, description="Execution duration in seconds")
    provider: Optional[ModelProvider] = Field(None, description="Model provider used by the node")

    @model_validator(mode='before')
    @classmethod
    def set_modified_at(cls, values):
        """Update modified_at timestamp on any change."""
        values['modified_at'] = datetime.utcnow()
        return values

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

class ToolConfig(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class NodeConfig(BaseModel):
    """Configuration for a node in the workflow."""
    id: str = Field(..., description="Unique identifier for the node")
    type: str = Field(..., description="Type of node (e.g., 'ai')")
    model: str = Field(..., description="Model to use for the node")
    prompt: str = Field(..., description="Prompt template for the node")
    name: Optional[str] = Field(None, description="Human-readable name for the node")
    level: int = Field(default=0, description="Execution level for parallel processing")
    dependencies: List[str] = Field(default_factory=list, description="List of node IDs this node depends on")
    timeout: Optional[float] = Field(None, description="Optional timeout in seconds")
    templates: Dict[str, Any] = Field(default_factory=dict, description="Message templates for the node")
    llm_config: Optional[LLMConfig] = Field(None, description="LLM configuration for the node")
    metadata: Optional[NodeMetadata] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input schema for the node")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output schema for the node")
    input_mappings: Dict[str, InputMapping] = Field(default_factory=dict, description="Input mappings for the node's prompt placeholders")
    input_selection: Optional[List[str]] = None
    context_rules: Dict[str, ContextRule] = Field(default_factory=dict, description="Context rules for the node")
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    format_specifications: Dict[str, Any] = Field(default_factory=dict, description="Format specifications for the node")
    provider: ModelProvider = Field(default=ModelProvider.OPENAI, description="Model provider for the node")
    token_management: Dict[str, Any] = Field(
        default_factory=lambda: {
            "truncate": True,
            "preserve_sentences": True,
            "max_context_tokens": 4096,
            "max_completion_tokens": 1024
        },
        description="Token management configuration"
    )
    tools: Optional[List[ToolConfig]] = None

    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v: List[str], info) -> List[str]:
        """Validate that a node doesn't depend on itself."""
        node_id = info.data.get('id')
        if node_id and node_id in v:
            raise ValueError(f"Node {node_id} cannot depend on itself")
        return v

    @field_validator('token_management')
    @classmethod
    def validate_token_management(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate token management configuration."""
        if not isinstance(v, dict):
            raise ValueError("token_management must be a dictionary")
        
        # Validate max_context_tokens
        if 'max_context_tokens' in v:
            if not isinstance(v['max_context_tokens'], int) or v['max_context_tokens'] <= 0:
                raise ValueError("max_context_tokens must be a positive integer")
        
        # Validate max_completion_tokens
        if 'max_completion_tokens' in v:
            if not isinstance(v['max_completion_tokens'], int) or v['max_completion_tokens'] <= 0:
                raise ValueError("max_completion_tokens must be a positive integer")
        
        # Validate boolean fields
        if 'truncate' in v and not isinstance(v['truncate'], bool):
            raise ValueError("truncate must be a boolean")
        if 'preserve_sentences' in v and not isinstance(v['preserve_sentences'], bool):
            raise ValueError("preserve_sentences must be a boolean")
        
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if self.metadata is None:
            self.metadata = NodeMetadata(
                node_id=self.id,
                node_type=self.type,
                version="1.0.0",
                description=f"Node {self.id} of type {self.type}",
                provider=self.provider
            )
        # Check model compatibility for all templates
        if 'templates' in data and 'model' in data:
            model = data['model']
            for template in self.templates.values():
                if not template.is_compatible_with_model(model):
                    raise ValueError(f"Model {model} is too old for template requiring {template.min_model_version}")

    model_config = ConfigDict(extra="forbid")  # Prevent unexpected arguments

class NodeExecutionRecord(BaseModel):
    """Execution statistics and historical data"""
    node_id: str
    executions: int = Field(0, ge=0, description="Total execution attempts")
    successes: int = Field(0, ge=0, description="Successful executions")
    failures: int = Field(0, ge=0, description="Failed executions")
    avg_duration: float = Field(0.0, ge=0, description="Average execution time in seconds")
    last_executed: Optional[datetime] = None
    token_usage: Dict[str, int] = Field(default_factory=dict,
                                      description="Token usage by model version")
    provider_usage: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Token usage by provider and model"
    )

class NodeIO(BaseModel):
    """Input/Output schema for a node."""
    data_schema: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class UsageMetadata(BaseModel):
    """Usage metadata for a node execution."""
    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion")
    total_tokens: int = Field(default=0, description="Total number of tokens used")
    cost: float = Field(default=0.0, description="Cost of the API call in USD")
    api_calls: int = Field(default=1, description="Number of API calls made")
    model: str = Field(..., description="Model used for the execution")
    node_id: str = Field(..., description="ID of the node that generated this usage")
    provider: ModelProvider = Field(..., description="Provider used for the execution")
    token_limits: Dict[str, int] = Field(
        default_factory=lambda: {
            "context": 4096,
            "completion": 1024
        },
        description="Token limits for the execution"
    )

class NodeExecutionResult(BaseModel):
    """Result of a node execution."""
    success: bool = Field(default=True, description="Whether the execution was successful")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    output: Optional[Dict[str, Any]] = Field(None, description="Output data from the node")
    metadata: NodeMetadata = Field(..., description="Metadata about the execution")
    usage: Optional[UsageMetadata] = Field(None, description="Usage statistics from the execution")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    context_used: Optional[Dict[str, Any]] = Field(None, description="Context used for the execution")
    token_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Token statistics including truncation and limits"
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: str
        }
