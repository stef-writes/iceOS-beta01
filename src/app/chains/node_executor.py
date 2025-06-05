from typing import Any, Dict, Tuple, Optional, List
from app.models.node_models import NodeExecutionResult, NodeMetadata
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NodeExecutor:
    """
    Handles execution of a single node in the ScriptChain, including context resolution,
    error/result creation, and callback handling.
    """
    def __init__(
        self,
        context_manager: Any,
        chain_id: str,
        persist_intermediate_outputs: bool = True,
        callbacks: Optional[List[Any]] = None,
        tool_service: Optional[Any] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ):
        self.context_manager = context_manager
        self.chain_id = chain_id
        self.persist_intermediate_outputs = persist_intermediate_outputs
        self.callbacks = callbacks or []
        self.tool_service = tool_service
        self.initial_context = initial_context or {}

    async def execute_node(self, node, accumulated_results: Dict[str, NodeExecutionResult]) -> Tuple[str, NodeExecutionResult]:
        start_time = datetime.utcnow()
        node_id = node.id
        try:
            # Context resolution
            context = {}
            missing_dependencies = []
            validation_errors = []
            for dep_id in node.dependencies:
                dep_result = accumulated_results.get(dep_id)
                if not dep_result or not dep_result.success:
                    missing_dependencies.append(dep_id)
                    continue
                if not dep_result.output:
                    validation_errors.append(f"Dependency '{dep_id}' produced no output")
                    continue
                # Input mappings
                if node.input_mappings:
                    for current_prompt_placeholder, mapping_config in node.input_mappings.items():
                        if mapping_config.source_node_id != dep_id:
                            continue
                        try:
                            value_from_dependency = self.resolve_nested_path(
                                dep_result.output,
                                mapping_config.source_output_key
                            )
                        except (KeyError, IndexError, TypeError) as e:
                            validation_errors.append(
                                f"Node '{node.id}': Failed to resolve path '{mapping_config.source_output_key}' "
                                f"in dependency '{dep_id}' output: {str(e)}"
                            )
                            continue
                        context[current_prompt_placeholder] = value_from_dependency
                else:
                    # Fallback: use dependency output directly
                    if isinstance(dep_result.output, dict):
                        if 'text' in dep_result.output:
                            context[dep_id] = str(dep_result.output['text'])
                        elif 'result' in dep_result.output:
                            context[dep_id] = dep_result.output['result']
                        else:
                            context[dep_id] = dep_result.output
                    else:
                        context[dep_id] = dep_result.output
            # Merge initial context if no dependencies
            if not node.dependencies:
                context = {**self.initial_context, **context}
            # Handle missing dependencies
            if missing_dependencies:
                error_msg = f"Node '{node.id}' skipped: Required dependencies failed or missing: {missing_dependencies}"
                logger.warning(error_msg)
                error_result = NodeExecutionResult(
                    success=False,
                    error=error_msg,
                    metadata=NodeMetadata(
                        node_id=node_id,
                        node_type=node.type,
                        start_time=start_time,
                        end_time=datetime.utcnow(),
                        error_type="MissingDependencyError",
                        provider=getattr(node, 'provider', None)
                    )
                )
                await self._trigger_callbacks('node_error', error_result)
                return node_id, error_result
            # Handle validation errors
            if validation_errors:
                error_msg = f"Node '{node.id}' validation failed:\n" + "\n".join(validation_errors)
                logger.error(error_msg)
                error_result = NodeExecutionResult(
                    success=False,
                    error=error_msg,
                    metadata=NodeMetadata(
                        node_id=node_id,
                        node_type=node.type,
                        start_time=start_time,
                        end_time=datetime.utcnow(),
                        error_type="ValidationError",
                        provider=getattr(node, 'provider', None)
                    )
                )
                await self._trigger_callbacks('node_error', error_result)
                return node_id, error_result
            # Execute node
            logger.debug(f"Node '{node.id}' (Name: '{getattr(node, 'name', '')}') executing with context: {json.dumps(context, indent=2, default=str)}")
            result = await node.execute(context)
            # Update context and metrics
            if result.success and self.persist_intermediate_outputs:
                self.context_manager.update_context(
                    node_id,
                    result.output,
                    execution_id=self.chain_id
                )
            await self._trigger_callbacks('node_end', result)
            return node_id, result
        except Exception as e:
            error_result = NodeExecutionResult(
                success=False,
                error=str(e),
                metadata=NodeMetadata(
                    node_id=node_id,
                    node_type=getattr(node, 'type', None),
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                    error_type=e.__class__.__name__,
                    provider=getattr(node, 'provider', None)
                )
            )
            await self._trigger_callbacks('node_error', error_result)
            return node_id, error_result

    async def _trigger_callbacks(self, event: str, data: Any) -> None:
        for callback in self.callbacks:
            try:
                if event == 'node_start':
                    await callback.on_node_start(data)
                elif event == 'node_end':
                    await callback.on_node_end(data)
                elif event == 'node_error':
                    await callback.on_node_error(data)
            except Exception as e:
                logger.error(f"Error in callback {callback.__class__.__name__}: {str(e)}")

    @staticmethod
    def resolve_nested_path(data: Any, path: str) -> Any:
        if not path:
            return data
        parts = path.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"Key '{part}' not found in dict. Available keys: {list(current.keys())}")
                current = current[part]
            elif isinstance(current, (list, tuple)):
                try:
                    index = int(part)
                    if index < 0 or index >= len(current):
                        raise IndexError(f"Index {index} out of bounds for array of length {len(current)}")
                    current = current[index]
                except ValueError:
                    raise TypeError(f"Cannot use non-integer key '{part}' to index array")
            else:
                raise TypeError(f"Cannot access '{part}' on type {type(current)}")
        return current
