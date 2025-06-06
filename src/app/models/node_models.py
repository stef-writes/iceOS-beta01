"""
Data models for node configurations and metadata
"""

from typing import Dict, List, Optional, Union, Any, Type, Literal, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import datetime
from .config import LLMConfig, MessageTemplate, ModelProvider
from enum import Enum
import json

class ContextFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    CUSTOM = "custom"

class InputMapping(BaseModel):
    """Mapping configuration for node inputs"""
    source_node_id: str = Field(..., description="Source node ID (UUID of the dependency)")
    source_output_key: str = Field(..., description="Key from the source node's output object to use (e.g., 'text', 'result', 'data.items.0')")
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
    name: Optional[str] = None
    version: str = Field("1.0.0", pattern=r"^\d+\.\d+\.\d+$", description="Semantic version of node configuration")
    owner: Optional[str] = Field(None, description="Node owner/maintainer")
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    description: Optional[str] = Field(None, description="Description of the node")
    error_type: Optional[str] = Field(None, description="Type of error if execution failed")
    timestamp: Optional[datetime] = None
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

    # TODO: Refactor or remove json_encoders for Pydantic v2 compatibility

    # Automatically compute duration if possible ------------------------------
    @model_validator(mode='after')
    def _set_duration(self):  # noqa: D401
        if self.start_time and self.end_time and self.duration is None:
            self.duration = (self.end_time - self.start_time).total_seconds()
        elif self.start_time and self.duration is not None and self.end_time is None:
            from datetime import timedelta
            self.end_time = self.start_time + timedelta(seconds=self.duration)
        return self

class ToolConfig(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

# ---------------------------------------------------------------------------
# NEW, SIMPLIFIED NODE CONFIG CLASSES
# ---------------------------------------------------------------------------

# NOTE: The original monolithic ``NodeConfig`` class has been renamed to
# ``AiNodeConfig``.  A lighter ``ToolNodeConfig`` was added and, for backwards
# compatibility, ``NodeConfig`` is now an alias::
#
#     NodeConfig = Union[AiNodeConfig, ToolNodeConfig]
#
# External code therefore keeps working while the new, clearer split is in
# place.  The *only* mandatory discriminator is the ``type`` field which must
# be either ``"ai"`` or ``"tool"``.


class BaseNodeConfig(BaseModel):
    """Common fields shared by all node configurations."""

    id: str = Field(..., description="Unique identifier for the node")
    type: str = Field(..., description="Type discriminator (ai | tool)")
    name: Optional[str] = Field(None, description="Human-readable name")
    dependencies: List[str] = Field(default_factory=list, description="IDs of prerequisite nodes")
    level: int = Field(default=0, description="Execution level for parallelism")
    metadata: Optional[NodeMetadata] = None
    provider: ModelProvider = Field(default=ModelProvider.OPENAI, description="Model provider for the node")
    # Maximum time (in seconds) the orchestrator will wait for this node to finish. ``None`` disables the timeout.
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Hard timeout for node execution in seconds (None = no timeout)"
    )

    # IO schemas are optional and can be provided as loose dicts or Pydantic models.
    input_schema: Union[Dict[str, Any], Type[BaseModel]] = Field(default_factory=dict)
    output_schema: Union[Dict[str, Any], Type[BaseModel]] = Field(default_factory=dict)

    # Mapping of placeholders in the prompt / template to dependency outputs.
    input_mappings: Dict[str, InputMapping] = Field(default_factory=dict)

    use_cache: bool = Field(
        default=True,
        description="Whether the orchestrator should reuse cached results when the context & config are unchanged."
    )

    input_selection: Optional[List[str]] = Field(default=None, description="List of input keys to include in the prompt (order preserved). If None, include all inputs.")

    @field_validator('dependencies')
    @classmethod
    def _no_self_dependency(cls, v: List[str], info):
        node_id = info.data.get('id')
        if node_id and node_id in v:
            raise ValueError(f"Node {node_id} cannot depend on itself")
        return v

    @model_validator(mode='after')
    def _ensure_metadata(self) -> 'BaseNodeConfig':
        if self.metadata is None:
            self.metadata = NodeMetadata(
                node_id=self.id,
                node_type=self.type,
                version="1.0.0",
                description=f"Node {self.id} (type={self.type})",
            )
        return self

    # ------------------------------------------------------------------
    # Common helpers preserved from the legacy implementation
    # ------------------------------------------------------------------

    @field_validator('input_mappings')
    @classmethod
    def _validate_input_mappings(cls, v: Dict[str, InputMapping], info):
        """Ensure that input mappings reference declared dependencies."""
        data = info.data
        dependencies = data.get('dependencies', [])
        if dependencies:
            for placeholder, mapping in v.items():
                # Allow literal/static values as-is.
                if not isinstance(mapping, InputMapping):
                    continue
                if mapping.source_node_id not in dependencies:
                    raise ValueError(
                        f"Input mapping for '{placeholder}' references non-existent dependency '{mapping.source_node_id}'. "
                        f"Available dependencies: {dependencies}"
                    )
        return v

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and coerce *input_data* according to *input_schema*."""
        if not self.input_schema:
            return input_data

        result: Dict[str, Any] = {}
        for key, expected_type in self.input_schema.items():
            if key not in input_data:
                raise ValueError(f"Missing required input field: {key}")

            value = input_data[key]

            # Only attempt coercion when the config advertises the flag.
            coerce = getattr(self, 'coerce_input_types', True)

            if not coerce:
                result[key] = value
                continue

            try:
                if expected_type == "int":
                    result[key] = int(value)
                elif expected_type == "float":
                    result[key] = float(value)
                elif expected_type == "bool":
                    result[key] = bool(value)
                elif expected_type == "str":
                    result[key] = str(value)
                else:
                    result[key] = value
            except (ValueError, TypeError):
                raise ValueError(
                    f"Validation error: Could not coerce {key}={value} to type {expected_type}"
                )

        return result

    def adapt_schema_from_context(self, context: Dict[str, Any]) -> None:  # noqa: D401
        """Hook for dynamic schema adaptation based on upstream context."""
        return None

    @staticmethod
    def is_pydantic_schema(schema) -> bool:  # noqa: D401
        from pydantic import BaseModel as _BaseModelChecker
        return isinstance(schema, type) and issubclass(schema, _BaseModelChecker)

class AiNodeConfig(BaseNodeConfig):
    """Configuration for an LLM-powered node."""

    type: Literal['ai'] = 'ai'

    # LLM-specific ----------------------------------------------------------------
    model: str = Field(..., description="Model name, e.g. gpt-3.5-turbo")
    prompt: str = Field(..., description="Prompt template")
    llm_config: LLMConfig = Field(..., description="Provider-specific parameters")

    # Optional quality-of-life flags kept for now (may be removed later).
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # Experimental / less frequently used knobs – scheduled for removal.
    context_rules: Dict[str, ContextRule] = Field(default_factory=dict)
    format_specifications: Dict[str, Any] = Field(default_factory=dict)
    coerce_output_types: bool = Field(default=True)
    coerce_input_types: bool = Field(default=True)

    tools: Optional[List[ToolConfig]] = Field(default=None, description="List of ToolConfig objects describing callable tools available to the node")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Default arguments for the tool when invoked via tool_name")

class ToolNodeConfig(BaseNodeConfig):
    """Configuration for a deterministic tool execution."""

    type: Literal['tool'] = 'tool'

    tool_name: str = Field(..., description="Registered name of the tool to invoke")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments to forward to the tool")


# Backwards-compatibility alias ----------------------------------------------------

from typing import Union as _UnionAlias  # local alias

NodeConfig = Annotated[
    _UnionAlias[AiNodeConfig, ToolNodeConfig],
    Field(discriminator='type'),
]

# ---------------------------------------------------------------------------
# END OF NEW CLASSES – the legacy giant NodeConfig definition was removed.
# ---------------------------------------------------------------------------

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

    # Auto-calculate totals ---------------------------------------------------
    @model_validator(mode='after')
    def _fill_totals(self):  # noqa: D401
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

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

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # TODO: Refactor or remove json_encoders for Pydantic v2 compatibility

# --- Pydantic Schemas for Node Outputs ---
class CatFactOutput(BaseModel):
    fact: str
    length: int

class WordCountOutput(BaseModel):
    word_count: int
    text: str

class TweetOutput(BaseModel):
    tweet: str

class ChainExecutionResult(BaseModel):
    """Result of a chain execution."""
    success: bool = Field(default=True, description="Whether the execution was successful")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    output: Optional[Dict[str, Any]] = Field(None, description="Output data from the final node in the chain")
    metadata: NodeMetadata = Field(..., description="Metadata about the chain execution")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    token_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Token statistics including truncation and limits"
    )
