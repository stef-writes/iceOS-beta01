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

# Remove langchain imports
# from langchain_community.chat_models import ChatOpenAI
# from langchain_core.messages import (
#     HumanMessage,
#     SystemMessage,
#     AIMessage
# )

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
        self.llm_config = llm_config
        self.callbacks = callbacks or []
        self.llm_service = llm_service or LLMService()
        self.tool_service = tool_service or ToolService()
        if not llm_config.api_key:
            raise ValueError(f"API key is required for provider {llm_config.provider}")
    
    def build_tool_preamble(self, tools: list) -> str:
        """Generate a human-readable preamble describing available tools, their parameters, and usage examples."""
        return build_tool_preamble(tools)

    async def prepare_prompt(self, inputs: Dict[str, Any]) -> str:
        """Prepare prompt with selected context from inputs, including a tool preamble if tools are available."""
        return prepare_prompt(self.config, self.context_manager, self.llm_config, self.tool_service, inputs)

    def _truncate_prompt(self, prompt: str, current_tokens: int) -> str:
        """Truncate prompt to fit within token limit.
        
        Args:
            prompt: Prompt to truncate
            current_tokens: Current token count
            
        Returns:
            Truncated prompt
        """
        max_tokens = self.llm_config.max_context_tokens
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
        Execute the text generation node using the appropriate LLM service and handle tool/function calls.
        Implements an agentic 'thought loop':
        - If the LLM requests a tool call, execute the tool and send the result back to the LLM for further reasoning.
        - Repeat up to max_steps, or until the LLM returns a normal text answer.
        """
        start_time = datetime.utcnow()
        context = context or {}
        error_message_prefix = f"Node '{self.config.id}' (Name: '{self.config.name}', Provider: {self.llm_config.provider}, Model: {self.llm_config.model}): "
        messages = []
        tool_outputs = []
        step = 0
        try:
            # Prepare the initial prompt
            try:
                prompt_template_for_handler = await self.prepare_prompt(context)
            except KeyError as e:
                logger.error(f"{error_message_prefix}Missing key '{str(e)}' for prompt template. Context: {json.dumps(context, indent=2, default=str)}", exc_info=True)
                return NodeExecutionResult(
                    success=False,
                    error=f"{error_message_prefix}Prompt formatting error: Missing key '{str(e)}'.",
                    metadata=self._create_error_metadata(start_time, "PromptFormattingError"),
                    execution_time=(datetime.utcnow() - start_time).total_seconds()
                )

            # Pass tools to the LLM service if present
            tools = None
            if self.config.tools:
                tools = [tool.model_dump() for tool in self.config.tools]
            else:
                tools = self.tool_service.list_tools_with_schemas() if hasattr(self, 'tool_service') else []

            # Start the agentic loop
            last_llm_output = None
            usage_metadata = None
            while step < max_steps:
                step += 1
                logger.debug(f"{error_message_prefix}Agentic loop step {step}")
                if step == 1:
                    llm_input = prompt_template_for_handler
                else:
                    # Use the utility for tool output formatting
                    tool_output_strs = [
                        format_tool_output(t['tool_name'], t['output']) for t in tool_outputs if t['success']
                    ]
                    llm_input = prompt_template_for_handler + "\n" + "\n".join(tool_output_strs)

                generated_text, usage_dict, handler_error = await self.llm_service.generate(
                    llm_config=self.llm_config,
                    prompt=llm_input,
                    context=context,
                    tools=tools
                )
                last_llm_output = generated_text

                # --- Tool/function call detection and handling ---
                tool_name, tool_args = detect_tool_call(generated_text)
                tool_call_detected = tool_name is not None
                tool_result = None
                tool_error = None

                if tool_call_detected:
                    logger.info(f"Tool/function call detected: {tool_name} with args: {tool_args}")
                    tool_exec_result = await self.tool_service.execute(tool_name, tool_args or {})
                    tool_outputs.append(tool_exec_result)
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
                            usage=None,
                            execution_time=duration
                        )
                    # Continue the loop: send tool output back to LLM for further reasoning
                    continue
                # --- End tool/function call handling ---

                if handler_error:
                    logger.error(f"{error_message_prefix}Handler error: {handler_error}")
                    return NodeExecutionResult(
                        success=False,
                        error=f"{error_message_prefix}{handler_error}",
                        metadata=self._create_error_metadata(start_time, "LLMHandlerError"),
                        execution_time=(datetime.utcnow() - start_time).total_seconds()
                    )

                # If no tool call detected, treat as final answer
                output_data, validation_success, validation_error = self._process_and_validate_output(generated_text)
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
                            model=self.llm_config.model,
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
            # If we reach here, max_steps was hit
            logger.warning(f"{error_message_prefix}Max agentic steps ({max_steps}) reached.")
            return NodeExecutionResult(
                success=False,
                output=None,
                error=f"{error_message_prefix}Max agentic steps ({max_steps}) reached.",
                metadata=self._create_error_metadata(start_time, "MaxStepsReached"),
                execution_time=(datetime.utcnow() - start_time).total_seconds()
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
        if current_tokens <= self.llm_config.max_context_tokens:
            return messages
            
        # Remove oldest messages first
        while current_tokens > self.llm_config.max_context_tokens and len(messages) > 1:
            removed = messages.pop(0)
            try:
                removed_tokens = TokenCounter.count_tokens(
                    removed["content"],
                    self.llm_config.model,
                    self.llm_config.provider
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
            model=self.llm_config.model,
            node_id=self.config.id,
            provider=self.llm_config.provider,
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
            self.metrics["token_usage"][self.llm_config.provider] = result.usage

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
        """Processes the raw text from LLM and validates against output_schema."""
        output = {}
        node_succeeded = True
        final_error_message = None
        error_message_prefix = f"Node '{self.config.id}' (Name: '{self.config.name}'): "

        if self.config.output_schema:
            for key, type_str_in_schema in self.config.output_schema.items():
                # Try to parse common types like int from the generated_text
                # This is a simplified parsing logic. More robust parsing might be needed.
                parsed_value = None
                if type_str_in_schema == "int":
                    try:
                        numbers = re.findall(r'-?\d+', generated_text)
                        if numbers:
                            parsed_value = int(numbers[0])
                    except (ValueError, IndexError):
                        pass # Will be caught by validation if key is missing or type mismatch
                elif type_str_in_schema == "float":
                    try:
                        # More robust float parsing might be needed
                        numbers = re.findall(r'-?\d*\.?\d+', generated_text)
                        if numbers:
                            parsed_value = float(numbers[0])
                    except (ValueError, IndexError):
                        pass
                # Add other type parsers as needed (e.g., bool, list from string)
                
                # If a specific parser was successful for the key, use its value.
                # Otherwise, if the schema expects a string for this key, use the raw generated_text.
                if parsed_value is not None:
                    output[key] = parsed_value
                elif type_str_in_schema == "str":
                    output[key] = generated_text # Assign raw text if schema is str and no specific parser hit
                # If schema is not str and no specific parser hit, output[key] will remain unset here.

            # Strict validation of the populated 'output' against 'self.config.output_schema'
            for key_in_schema, type_str_in_schema in self.config.output_schema.items():
                if key_in_schema not in output:
                    # If the key is missing, but schema expects string, and we haven't assigned it yet
                    # (e.g. it wasn't 'int' or 'float'), assign the raw text now if it was the *only* string field.
                    # This is a bit heuristic for cases where LLM just returns the value for a single-field schema.
                    if type_str_in_schema == "str" and len(self.config.output_schema) == 1:
                        output[key_in_schema] = generated_text
                    else:
                        node_succeeded = False
                        final_error_message = f"Output schema validation failed: Key '{key_in_schema}' (expected type '{type_str_in_schema}') is missing from node output."
                        logger.error(f"{error_message_prefix}{final_error_message} LLM raw response: '{generated_text}', Parsed output attempt: {output}")
                        break
               
                # Proceed with type check if key is present
                if node_succeeded and key_in_schema in output:
                    try:
                        actual_expected_type = eval(type_str_in_schema)
                        if not isinstance(output[key_in_schema], actual_expected_type):
                            # Attempt type conversion for basic types if direct instance check fails
                            can_convert = False
                            try:
                                if actual_expected_type is int:
                                    output[key_in_schema] = int(output[key_in_schema])
                                    can_convert = True
                                elif actual_expected_type is float:
                                    output[key_in_schema] = float(output[key_in_schema])
                                    can_convert = True
                                elif actual_expected_type is str:
                                    output[key_in_schema] = str(output[key_in_schema])
                                    can_convert = True
                                # Add other conversions if necessary
                            except (ValueError, TypeError):
                                pass # Conversion failed

                            if not can_convert or not isinstance(output[key_in_schema], actual_expected_type):
                                node_succeeded = False
                                final_error_message = (
                                    f"Output schema validation failed: Key '{key_in_schema}' has value '{output[key_in_schema]}' "
                                    f"of type '{type(output[key_in_schema]).__name__}', but schema expects type '{type_str_in_schema}'. Attempted conversion failed."
                                )
                                logger.error(f"{error_message_prefix}{final_error_message} LLM raw response: '{generated_text}'")
                                break
                    except NameError:
                        node_succeeded = False
                        final_error_message = f"Output schema validation failed: Invalid type string '{type_str_in_schema}' for key '{key_in_schema}' in output_schema."
                        logger.error(f"{error_message_prefix}{final_error_message}")
                        break
        else:
            # No output_schema defined, default to putting raw text in "text" key
            output["text"] = generated_text
        
        # Bulletproof: If schema is a single string key and output is missing, assign it
        if (
            self.config.output_schema
            and len(self.config.output_schema) == 1
            and list(self.config.output_schema.values())[0] == "str"
            and list(self.config.output_schema.keys())[0] not in output
        ):
            output[list(self.config.output_schema.keys())[0]] = generated_text
            node_succeeded = True
            final_error_message = None

        return output, node_succeeded, final_error_message