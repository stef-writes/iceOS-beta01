"""
Enhanced workflow orchestration system with level-based parallel execution and robust context management
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from pydantic import BaseModel, ValidationError
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.models.config import LLMConfig, MessageTemplate, ModelProvider
from app.utils.context import GraphContextManager
from app.nodes.base import BaseNode
from app.nodes.factory import node_factory
from app.utils.callbacks import ScriptChainCallback
from app.utils.token_counter import TokenCounter
import logging
from uuid import uuid4
import asyncio
import os
import traceback
import json
from collections import deque

logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class ScriptChainError(Exception):
    """Base exception class for ScriptChain errors"""
    pass

class CircularDependencyError(ScriptChainError):
    """Exception raised when circular dependencies are detected"""
    pass

class ScriptChain:
    """Advanced workflow orchestrator with level-based parallel execution"""
    
    def __init__(
        self,
        nodes: List[NodeConfig],
        context_manager: Optional[GraphContextManager] = None,
        callbacks: Optional[List[ScriptChainCallback]] = None,
        max_parallel: int = 5,
        persist_intermediate_outputs: bool = True,
        tool_service: Optional[Any] = None
    ):
        """Initialize the script chain.
        
        Args:
            nodes: List of node configurations
            context_manager: Optional context manager for global context persistence.
            callbacks: Optional list of callbacks
            max_parallel: Maximum number of parallel executions
            persist_intermediate_outputs: If True, persist output of each node in the chain
                                          to the global context manager.
            tool_service: Optional tool service for node execution
        """
        self.nodes = {node.id: node for node in nodes}
        self.global_context_manager = context_manager or GraphContextManager()
        self.callbacks = callbacks or []
        self.max_parallel = max_parallel
        self.chain_id = str(uuid4())
        self.persist_intermediate_outputs = persist_intermediate_outputs
        self.metrics = {
            'total_tokens': 0,
            'node_execution_times': {},
            'provider_usage': {},
            'token_usage': {}
        }
        self.tool_service = tool_service
        
        # Build dependency graph
        self.graph = nx.DiGraph()
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, level=0)
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Dependency {dep} not found for node {node_id}")
                self.graph.add_edge(dep, node_id)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                cycle_str = " -> ".join(cycles[0])
                raise CircularDependencyError(f"Circular dependency detected: {cycle_str}")
        except nx.NetworkXNoCycle:
            pass
            
        # Assign levels for parallel execution
        for node_id in nx.topological_sort(self.graph):
            node = self.nodes[node_id]
            node.level = max(
                (self.nodes[dep].level for dep in node.dependencies),
                default=-1
            ) + 1
            
        # Group nodes by level
        self.levels = {}
        for node_id, node in self.nodes.items():
            level = node.level
            if level not in self.levels:
                self.levels[level] = []
            self.levels[level].append(node_id)
            
        logger.info(f"Initialized ScriptChain with {len(nodes)} nodes in {len(self.levels)} levels")
        
    async def execute(self) -> NodeExecutionResult:
        """Execute the workflow.
        
        Returns:
            NodeExecutionResult containing the final output and metadata
        """
        start_time = datetime.utcnow()
        results = {}
        errors = []
        
        # Execute each level in sequence
        for level_num in sorted(self.levels.keys()):
            level_start_time = datetime.utcnow()
            level_node_outputs = await self._execute_level(level_num, results)
            
            # Process level results
            for node_id, result_obj in level_node_outputs.items():
                if result_obj.success:
                    results[node_id] = result_obj
                    # Update metrics
                    if result_obj.usage:
                        self._update_metrics(node_id, result_obj)
                else:
                    errors.append(f"Node {node_id} failed: {result_obj.error}")
                    
            # Check if we should continue
            if errors and not self._should_continue(errors):
                break
                
        # Prepare final result
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return NodeExecutionResult(
            success=len(errors) == 0,
            output=results,
            error="\n".join(errors) if errors else None,
            metadata=NodeMetadata(
                node_id=self.chain_id,
                node_type="script_chain",
                version="1.0.0",
                start_time=start_time,
                end_time=end_time,
                duration=duration
            ),
            execution_time=duration,
            token_stats={
                "total_tokens": self.metrics["total_tokens"],
                "provider_usage": self.metrics["provider_usage"],
                "token_usage": self.metrics["token_usage"]
            }
        )
        
    async def _execute_level(self, level: int, accumulated_results: Dict[str, NodeExecutionResult]) -> Dict[str, NodeExecutionResult]:
        """Execute all nodes in a level in parallel.
        
        Args:
            level: Level number to execute
            accumulated_results: Dictionary of results from previously completed levels
            
        Returns:
            Dictionary mapping node IDs to their execution results for the current level
        """
        level_start_time = datetime.utcnow()
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def process_node(node_id: str) -> Tuple[str, NodeExecutionResult]:
            async with semaphore:
                start_time = datetime.utcnow()
                node = self.nodes[node_id]
                
                # Create node instance
                try:
                    node_instance = node_factory(
                        node_config=node,
                        context_manager=self.global_context_manager,
                        llm_config=node.llm_config,
                        callbacks=self.callbacks,
                        tool_service=self.tool_service
                    )
                except Exception as e:
                    raise ValueError(f"Failed to instantiate node '{node.id}': {e}")
                
                try:
                    # Get context for node
                    context = {}
                    missing_dependencies = []
                    
                    # Add dependencies' outputs to context
                    for dep_id in node.dependencies:
                        dep_result = accumulated_results.get(dep_id)
                        if not dep_result or not dep_result.success:
                            missing_dependencies.append(dep_id)
                            continue
                            
                        if dep_result.output:
                            if node.input_mappings: # NodeConfig.input_mappings
                                # Explicit mapping: current_prompt_placeholder -> InputMapping(source_node_id, source_output_key)
                                for current_prompt_placeholder, mapping_config in node.input_mappings.items():
                                    if mapping_config.source_node_id == dep_id:
                                        value_from_dependency = dep_result.output.get(mapping_config.source_output_key)
                                        if value_from_dependency is not None:
                                            context[current_prompt_placeholder] = value_from_dependency
                                        else:
                                            logger.warning(
                                                f"Node '{node.id}': Source output key '{mapping_config.source_output_key}' not found "
                                                f"in dependency '{dep_id}' output for placeholder '{current_prompt_placeholder}'. "
                                                f"Available keys: {list(dep_result.output.keys())}"
                                            )
                            else:
                                # No explicit input_mappings defined for this node.
                                # Default behavior: try to get 'text' or 'result' from dependency output,
                                # and key it by the dependency's ID for the prompt.
                                # This is useful for simple chains where prompt placeholders match dependency IDs.
                                if isinstance(dep_result.output, dict):
                                    if 'text' in dep_result.output:
                                        context[dep_id] = str(dep_result.output['text'])
                                    elif 'result' in dep_result.output: # For nodes with numeric/structured results
                                        context[dep_id] = dep_result.output['result']
                                    else:
                                        # If neither 'text' nor 'result' is found, but it's a dict,
                                        # log a warning and provide the whole object. This might be noisy.
                                        logger.warning(
                                            f"Node '{node.id}': No explicit input_mappings and dependency '{dep_id}' output has no 'text' or 'result' key. "
                                            f"Using entire output object for placeholder '{dep_id}'. Output keys: {list(dep_result.output.keys())}"
                                        )
                                        context[dep_id] = dep_result.output
                                elif isinstance(dep_result.output, (str, int, float, bool)):
                                    # If the output is already a primitive type
                                    context[dep_id] = dep_result.output
                                else:
                                    logger.warning(
                                        f"Node '{node.id}': Dependency '{dep_id}' produced an unexpected output type: {type(dep_result.output)}. Cannot auto-map."
                                    )
                    
                    # Check if we have missing critical dependencies
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
                                provider=node.provider
                            )
                        )
                        await self._trigger_callbacks('node_error', error_result)
                        return node_id, error_result
                    
                    # Execute node
                    logger.debug(f"Node '{node.id}' (Name: '{node.name}') executing with context: {json.dumps(context, indent=2, default=str)}")
                    result = await node_instance.execute(context)
                    
                    # Update context and metrics
                    if result.success:
                        if self.persist_intermediate_outputs:
                            self.global_context_manager.update_context(
                                node_id,
                                result.output,
                                execution_id=self.chain_id
                            )
                        self._update_metrics(node_id, result)
                    
                    await self._trigger_callbacks('node_end', result)
                    
                    return node_id, result
                    
                except Exception as e:
                    error_result = NodeExecutionResult(
                        success=False,
                        error=str(e),
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node.type,
                            start_time=start_time,
                            end_time=datetime.utcnow(),
                            error_type=e.__class__.__name__,
                            provider=node.provider
                        )
                    )
                    await self._trigger_callbacks('node_error', error_result)
                    return node_id, error_result
        
        # Execute all nodes in the level
        tasks = [process_node(node_id) for node_id in self.levels[level]]
        current_level_node_outputs = dict(await asyncio.gather(*tasks))
        
        return current_level_node_outputs
        
    def _update_metrics(self, node_id: str, result: NodeExecutionResult) -> None:
        """Update chain metrics with node execution results"""
        if result.usage:
            # Update total tokens
            self.metrics['total_tokens'] += result.usage.total_tokens
            
            # Update provider usage
            provider = result.metadata.provider
            if provider not in self.metrics['provider_usage']:
                self.metrics['provider_usage'][provider] = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            self.metrics['provider_usage'][provider]['prompt_tokens'] += result.usage.prompt_tokens
            self.metrics['provider_usage'][provider]['completion_tokens'] += result.usage.completion_tokens
            self.metrics['provider_usage'][provider]['total_tokens'] += result.usage.total_tokens
            
            # Update token usage by model
            model = result.usage.model
            if model not in self.metrics['token_usage']:
                self.metrics['token_usage'][model] = 0
            self.metrics['token_usage'][model] += result.usage.total_tokens
            
        self.metrics['node_execution_times'][node_id] = datetime.utcnow()
        
    def _should_continue(self, errors: List[str]) -> bool:
        """Determine if execution should continue after errors.
        
        This method implements intelligent error handling:
        - Continue if there are nodes that can still execute (don't depend on failed nodes)
        - Stop only if all remaining nodes are blocked by failed dependencies
        
        Args:
            errors: List of error messages
            
        Returns:
            True if execution should continue, False otherwise
        """
        if not errors:
            return True
            
        # Get list of failed node IDs from error messages
        failed_nodes = set()
        for error in errors:
            # Extract node ID from error messages like "Node xyz failed: ..."
            if "Node " in error and " failed:" in error:
                try:
                    node_id = error.split("Node ")[1].split(" failed:")[0]
                    failed_nodes.add(node_id)
                except (IndexError, AttributeError):
                    continue
        
        # Check if there are any remaining levels that can still execute
        # A level can execute if none of its nodes depend on failed nodes
        for level_num in sorted(self.levels.keys()):
            for node_id in self.levels[level_num]:
                node = self.nodes[node_id]
                
                # Skip nodes that have already been processed
                if node_id in failed_nodes:
                    continue
                    
                # Check if this node's dependencies include any failed nodes
                depends_on_failed_node = any(dep_id in failed_nodes for dep_id in node.dependencies)
                
                # If this node doesn't depend on failed nodes, we can continue
                if not depends_on_failed_node:
                    logger.info(f"Chain execution continuing: Node '{node_id}' can still execute independently")
                    return True
        
        # If we get here, all remaining nodes depend on failed nodes
        logger.warning(f"Chain execution stopping: All remaining nodes depend on failed nodes: {failed_nodes}")
        return False
        
    async def _trigger_callbacks(self, event: str, data: Any) -> None:
        """Trigger callbacks for an event.
        
        Args:
            event: Event name
            data: Event data
        """
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
                
    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get list of dependencies for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of dependency node IDs
        """
        return list(self.graph.predecessors(node_id))
        
    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get list of nodes that depend on this node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of dependent node IDs
        """
        return list(self.graph.successors(node_id))
        
    def get_node_level(self, node_id: str) -> int:
        """Get execution level for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Execution level
        """
        return self.nodes[node_id].level
        
    def get_level_nodes(self, level: int) -> List[str]:
        """Get list of nodes in a level.
        
        Args:
            level: Level number
            
        Returns:
            List of node IDs in the level
        """
        return self.levels.get(level, [])
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current chain metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics