"""
Concrete node implementations using the data models
"""

import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from openai import AsyncOpenAI, OpenAI
from openai import APIError, RateLimitError, Timeout
import logging
import uuid
import asyncio
import json
import traceback
import re
from pydantic import ValidationError
from anthropic import AsyncAnthropic
import google.generativeai as genai
from app.utils.type_coercion import coerce_types

# Existing imports
# This is a test comment to trigger reload
from app.models.node_models import (
    NodeConfig, 
    NodeExecutionResult,
    NodeExecutionRecord,
    NodeMetadata,
    UsageMetadata,
    ContextRule
)
from app.models.config import LLMConfig, MessageTemplate, ModelProvider
from app.utils.context import GraphContextManager
from app.nodes.base import BaseNode
from app.utils.logging import logger
from app.utils.tracking import track_usage
from app.utils.callbacks import ScriptChainCallback
from app.utils.token_counter import TokenCounter
from app.llm_providers import OpenAIHandler, AnthropicHandler, GoogleGeminiHandler, DeepSeekHandler
from app.services.llm_service import LLMService
from app.services.tool_service import ToolService
from app.nodes.prompt_builder import build_tool_preamble, prepare_prompt
from app.nodes.tool_call_utils import detect_tool_call, format_tool_output
from app.nodes.error_handling import OpenAIErrorHandler
from app.nodes.constants import TOOL_INSTRUCTION

logger = logging.getLogger(__name__)

class AiNode(BaseNode):
    """AI-powered node supporting LLM text generation and tool/function calling"""
    
    def __init__(
        self,
        config: NodeConfig,
        context_manager: GraphContextManager,
        llm_config: LLMConfig,
        callbacks: Optional[List[ScriptChainCallback]] = None,
        llm_service: Optional[LLMService] = None,
        tool_service: Optional[ToolService] = None
    ):
        """Initialize AI node with LLM and Tool services."""
        super().__init__(config)
        self.context_manager = context_manager
        self._llm_config = llm_config
        self.callbacks = callbacks or []
        self.llm_service = llm_service or LLMService()
        self.tool_service = tool_service or ToolService()
    
    def build_tool_preamble(self, tools: list) -> str:
        """Generate a human-readable preamble describing available tools, their parameters, and usage examples."""
        return build_tool_preamble(tools)

    async def prepare_prompt(self, inputs: Dict[str, Any]) -> str:
        """Prepare prompt with selected context from inputs, including a tool preamble if tools are available."""
        return prepare_prompt(self.config, self.context_manager, self._llm_config, self.tool_service, inputs)

    def _truncate_prompt(self, prompt: str, current_tokens: int) -> str:
        """Truncate prompt to fit within token limit.
        
        Args:
            prompt: Prompt to truncate
            current_tokens: Current token count
            
        Returns:
            Truncated prompt
        """
        max_tokens = self._llm_config.max_context_tokens if self._llm_config.max_context_tokens is not None else 4096
        if current_tokens <= max_tokens:
            return prompt
        
        # Calculate how many tokens to remove
        tokens_to_remove = current_tokens - max_tokens
        
        # Estimate characters to remove (roughly 4 chars per token)
        chars_to_remove = tokens_to_remove * 4
        
        # Truncate from the end, preserving complete sentences
        truncated = prompt[:-chars_to_remove]
        last_period = truncated.rfind('.')
        if last_period > 0:
            truncated = truncated[:last_period + 1]
        
        return truncated

    def _prepare_messages(self, context: Optional[Dict] = None) -> List[Dict[str, str]]:
        """Prepare messages for the API call.
        
        Args:
            context: Optional context data
            
        Returns:
            List of message dictionaries
        """
        messages = []
        context = context or {}
        
        # Add system message if available
        if self.config.templates and "system" in self.config.templates:
            try:
                system_content = self.config.templates["system"].format(**context)
                messages.append({
                    "role": "system",
                    "content": system_content
                })
            except KeyError as e:
                logger.warning(f"Missing context key for system message: {e}")
        
        # Add user message
        try:
            user_content = self.config.prompt.format(**context)
            messages.append({
                "role": "user",
                "content": user_content
            })
        except KeyError as e:
            raise ValueError(f"Missing required context key: {e}")
        
        return messages

    async def execute(self, context: Dict[str, Any] = None, max_steps: int = 5) -> NodeExecutionResult:
        """
        Execute the node with a single-step tool call approach:
        - Send the prompt to the LLM.
        - If the LLM requests a tool call, execute the tool ONCE and return the tool's output as the final answer.
        - If the LLM does not request a tool call, return the LLM's answer as the final answer.
        (max_steps is kept for future agentic support, but is unused)
        """
        start_time = datetime.utcnow()
        usage_metadata = None  # Always initialize
        context = context or {}
        output_format = getattr(self.config, 'output_format', 'plain')
        # Add strong system prompt for output format
        system_prompt = None
        if output_format == 'json':
            system_prompt = 'Respond ONLY with a JSON object: {"tweet": "<your tweet here>"}'
        elif output_format == 'function_call':
            system_prompt = 'If you want to use a tool, respond with a function_call JSON as shown in the examples below.'
        elif output_format == 'plain':
            system_prompt = 'Respond with a short answer in plain English.'
        error_message_prefix = f"Node '{self.config.id}' (Name: '{self.config.name}', Provider: {self._llm_config.provider}, Model: {self._llm_config.model}): "
        try:
            # Prepare the initial prompt
            try:
                prompt_template_for_handler = await self.prepare_prompt(context)
                if system_prompt:
                    prompt_template_for_handler = system_prompt + "\n" + prompt_template_for_handler
            except KeyError as e:
                logger.error(f"{error_message_prefix}Missing key '{str(e)}' for prompt template. Context: {json.dumps(context, indent=2, default=str)}", exc_info=True)
                return NodeExecutionResult(
                    success=False,
                    error=f"{error_message_prefix}Prompt formatting error: Missing key '{str(e)}'.",
                    metadata=self._create_error_metadata(start_time, "PromptFormattingError"),
                    execution_time=(datetime.utcnow() - start_time).total_seconds()
                )

            # Pass tools to the LLM service if present
            tools = [tool.model_dump() for tool in self.config.tools] if self.config.tools else None

            # Single LLM call
            generated_text, usage_dict, handler_error = await self.llm_service.generate(
                self._llm_config,
                prompt_template_for_handler,
                context=context,
                tools=tools
            )
            logger.info(f"{error_message_prefix}Raw LLM output: {repr(generated_text)}")

            # Tool/function call detection
            tool_name, tool_args = detect_tool_call(generated_text)
            tool_call_detected = tool_name is not None

            if tool_call_detected:
                logger.info(f"Tool/function call detected: {tool_name} with args: {tool_args}")
                tool_exec_result = await self.tool_service.execute(tool_name, tool_args or {})
                if not tool_exec_result["success"]:
                    tool_error = tool_exec_result["error"]
                    end_time = datetime.utcnow()
                    duration = (end_time - start_time).total_seconds()
                    return NodeExecutionResult(
                        success=False,
                        output=None,
                        error=f"{error_message_prefix}{tool_error}" if tool_error else None,
                        metadata=NodeMetadata(
                            node_id=self.config.id,
                            node_type=self.config.type,
                            version=self.config.metadata.version if self.config.metadata else "1.0.0",
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            provider=self.config.provider,
                            error_type="ToolExecutionError"
                        ),
                        usage=usage_metadata,
                        execution_time=duration
                    )
                # Return the tool output as the final answer
                output_data, validation_success, validation_error = self._process_and_validate_output(json.dumps(tool_exec_result["output"]))
                final_success = validation_success
                final_error = validation_error if not validation_success else None
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                if usage_dict:
                    try:
                        usage_metadata = UsageMetadata(
                            prompt_tokens=usage_dict.get("prompt_tokens", 0),
                            completion_tokens=usage_dict.get("completion_tokens", 0),
                            total_tokens=usage_dict.get("total_tokens", 0),
                            model=self._llm_config.model,
                            node_id=self.config.id,
                            provider=self.config.provider
                        )
                    except Exception as e_usage:
                        logger.error(f"{error_message_prefix}Failed to create UsageMetadata: {str(e_usage)}", exc_info=True)
                return NodeExecutionResult(
                    success=final_success,
                    output=output_data,
                    error=f"{error_message_prefix}{final_error}" if final_error else None,
                    metadata=NodeMetadata(
                        node_id=self.config.id,
                        node_type=self.config.type,
                        version=self.config.metadata.version if self.config.metadata else "1.0.0",
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        provider=self.config.provider,
                        error_type="SchemaValidationError" if not final_success and validation_error else None
                    ),
                    usage=usage_metadata,
                    execution_time=duration
                )

            # Output parsing/validation logic based on output_format
            if output_format == 'json':
                # Try to parse as JSON, fallback to single-field wrap
                try:
                    output_data, validation_success, validation_error = self._process_and_validate_output(generated_text)
                except Exception as e:
                    output_data, validation_success, validation_error = { }, False, str(e)
                if not validation_success and self.config.output_schema and len(self.config.output_schema) == 1:
                    # Fallback: wrap as {field: value}
                    key = list(self.config.output_schema.keys())[0]
                    logger.warning(f"{error_message_prefix}Output was not valid JSON, wrapping as single-field: {key}")
                    output_data = {key: generated_text.strip()}
                    validation_success = True
                    validation_error = None
                final_success = validation_success
                final_error = validation_error if not validation_success else None
            elif output_format == 'plain':
                # Treat as plain text, wrap if single-field schema
                if self.config.output_schema and len(self.config.output_schema) == 1:
                    key = list(self.config.output_schema.keys())[0]
                    output_data = {key: generated_text.strip()}
                    final_success = True
                    final_error = None
                else:
                    output_data, final_success, final_error = self._process_and_validate_output(generated_text)
            else:
                # Default/fallback: use existing logic
                output_data, final_success, final_error = self._process_and_validate_output(generated_text)

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            if usage_dict:
                try:
                    usage_metadata = UsageMetadata(
                        prompt_tokens=usage_dict.get("prompt_tokens", 0),
                        completion_tokens=usage_dict.get("completion_tokens", 0),
                        total_tokens=usage_dict.get("total_tokens", 0),
                        model=self._llm_config.model,
                        node_id=self.config.id,
                        provider=self.config.provider
                    )
                except Exception as e_usage:
                    logger.error(f"{error_message_prefix}Failed to create UsageMetadata: {str(e_usage)}", exc_info=True)
            return NodeExecutionResult(
                success=final_success,
                output=output_data,
                error=f"{error_message_prefix}{final_error}" if final_error else None,
                metadata=NodeMetadata(
                    node_id=self.config.id,
                    node_type=self.config.type,
                    version=self.config.metadata.version if self.config.metadata else "1.0.0",
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    provider=self.config.provider,
                    error_type="SchemaValidationError" if not final_success and final_error else None
                ),
                usage=usage_metadata,
                execution_time=duration
            )
        except Exception as e:
            logger.error(f"{error_message_prefix}Unexpected error in execute method: {str(e)}", exc_info=True)
            return NodeExecutionResult(
                success=False,
                error=f"{error_message_prefix}Execution Error: {str(e)}",
                metadata=self._create_error_metadata(start_time, e.__class__.__name__),
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input data against schema.
        
        Args:
            inputs: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.config.input_schema:
            return True
            
        for key, expected_type in self.config.input_schema.items():
            if key not in inputs:
                return False
            if not isinstance(inputs[key], eval(expected_type)):
                return False
        return True
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate the cost of the API call.
        
        Args:
            usage: Token usage data
            
        Returns:
            Cost in USD
        """
        # TODO: Implement cost calculation based on model and provider
        return 0.0
    
    @property
    def input_keys(self) -> List[str]:
        """Get list of input keys."""
        return list(self.config.input_schema.keys()) if self.config.input_schema else []
    
    @property
    def output_keys(self) -> List[str]:
        """Get list of output keys."""
        return list(self.config.output_schema.keys()) if self.config.output_schema else []
    
    def get_template(self, role: str) -> MessageTemplate:
        """Get message template for a role.
        
        Args:
            role: Message role
            
        Returns:
            MessageTemplate object
        """
        if not self.config.templates or role not in self.config.templates:
            return None
        return self.config.templates[role]
    
    def _build_messages(self, inputs: Dict, context: Dict) -> List[Dict]:
        """Build message list for API call.
        
        Args:
            inputs: Input data
            context: Context data
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message if available
        system_template = self.get_template("system")
        if system_template:
            messages.append({
                "role": "system",
                "content": system_template.format(**context)
            })
        
        # Add user message
        user_template = self.get_template("user")
        if user_template:
            messages.append({
                "role": "user",
                "content": user_template.format(**inputs)
            })
        else:
            messages.append({
                "role": "user",
                "content": self.config.prompt.format(**inputs)
            })
        
        return messages
    
    def _truncate_messages(self, messages: List[Dict], current_tokens: int) -> List[Dict]:
        """Truncate messages to fit token limit.
        
        Args:
            messages: List of message dictionaries
            current_tokens: Current token count
            
        Returns:
            Truncated list of messages
        """
        if current_tokens <= self._llm_config.max_context_tokens:
            return messages
            
        # Remove oldest messages first
        while current_tokens > self._llm_config.max_context_tokens and len(messages) > 1:
            removed = messages.pop(0)
            try:
                removed_tokens = TokenCounter.count_tokens(
                    removed["content"],
                    self._llm_config.model,
                    self._llm_config.provider
                )
                current_tokens -= removed_tokens
            except ValueError:
                # If we can't count tokens, estimate
                current_tokens -= len(removed["content"]) // 4
                
        return messages
    
    def _process_response(self, response: Dict) -> Tuple[str, UsageMetadata]:
        """Process API response.
        
        Args:
            response: API response
            
        Returns:
            Tuple of (text, usage_metadata)
        """
        text_content = response.choices[0].message.content
        text = text_content.strip() if text_content else ""
        usage = UsageMetadata(
            model=self._llm_config.model,
            node_id=self.config.id,
            provider=self._llm_config.provider,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        return text, usage
    
    def _format_error(self, error: Exception) -> str:
        """Format error message."""
        return OpenAIErrorHandler.format_error_message(error)
    
    def _update_execution_stats(self, result: NodeExecutionResult):
        """Update execution statistics."""
        if result.usage:
            self.metrics["total_tokens"] += result.usage["total_tokens"]
            self.metrics["token_usage"][self._llm_config.provider] = result.usage

    def _create_error_metadata(self, start_time: datetime, error_type_str: str) -> NodeMetadata:
        """Helper to create metadata for error results."""
        return NodeMetadata(
            node_id=self.config.id,
            node_type=self.config.type,
            version=self.config.metadata.version if self.config.metadata else "1.0.0",
            start_time=start_time,
            end_time=datetime.utcnow(),
            duration=(datetime.utcnow() - start_time).total_seconds(),
            error_type=error_type_str,
            provider=self.config.provider
        )

    def _process_and_validate_output(self, generated_text: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        from pydantic import BaseModel, ValidationError, create_model
        logger.debug(f"[_PVP] Raw generated_text: '{generated_text}'")
        output = {}
        # Step 1: Parse output as JSON or fallback to dict for single-field schemas
        try:
            parsed_json = json.loads(generated_text.strip())
            if isinstance(parsed_json, dict):
                output = parsed_json
            else:
                # If not a dict, treat as plain text for single-field schemas
                if self.config.output_schema and len(self.config.output_schema) == 1:
                    key = list(self.config.output_schema.keys())[0]
                    output = {key: parsed_json}
                else:
                    return {}, False, "Output is not a dict and schema is not single-field."
        except Exception:
            # Fallback: treat as plain text for single-field schemas
            if self.config.output_schema and len(self.config.output_schema) == 1:
                key = list(self.config.output_schema.keys())[0]
                output = {key: generated_text.strip()}
            else:
                return {}, False, "Output is not valid JSON and schema is not single-field."

        # Step 2: Type coercion before Pydantic validation
        if self.config.output_schema:
            try:
                if getattr(self.config, "coerce_output_types", True):
                    output = coerce_types(output, self.config.output_schema)
                # Map string type names to Python types
                type_map = {"str": str, "int": int, "float": float, "bool": bool}
                fields = {}
                for k, v in self.config.output_schema.items():
                    py_type = type_map.get(v, str)  # Default to str if unknown
                    fields[k] = (py_type, ...)
                OutputModel = create_model("OutputModel", **fields)
                validated = OutputModel(**output)
                return validated.dict(), True, None
            except ValueError as e:
                return {}, False, f"Type coercion failed: {e}"
            except ValidationError as e:
                return {}, False, f"Output schema validation failed: {e}"
            except Exception as e:
                return {}, False, f"Output schema validation error: {e}"
        else:
            # No output schema, just return the output as-is
            return output, True, None