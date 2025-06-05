"""
Enhanced workflow orchestration system with level-based parallel execution and robust context management
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4
import asyncio
import logging
from .dependency_graph import DependencyGraph
from .node_executor import NodeExecutor
from .metrics import ChainMetrics
from .events import EventDispatcher
from .errors import ScriptChainError, CircularDependencyError
from .utils import resolve_nested_path
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.nodes.factory import node_factory

logger = logging.getLogger(__name__)

class ScriptChain:
    """
    Orchestrates workflow execution using level-based parallelism and robust context management.
    Delegates graph, node execution, metrics, and event logic to dedicated modules.
    """
    def __init__(
        self,
        nodes: List[NodeConfig],
        name: Optional[str] = None,
        context_manager: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        max_parallel: int = 5,
        persist_intermediate_outputs: bool = True,
        tool_service: Optional[Any] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ):
        self.nodes = {node.id: node for node in nodes}
        self.chain_id = str(uuid4())
        self.name = name or f"chain-{self.chain_id[:8]}"
        self.context_manager = context_manager
        self.max_parallel = max_parallel
        self.persist_intermediate_outputs = persist_intermediate_outputs
        self.tool_service = tool_service
        self.initial_context = initial_context or {}
        self.callbacks = callbacks or []
        # Modular components
        self.graph = DependencyGraph(nodes)
        self.levels = self.graph.get_level_nodes()
        self.metrics = ChainMetrics(self.name)
        self.events = EventDispatcher(self.callbacks)
        # Instantiate all nodes using node_factory
        self.node_instances = {
            node_id: node_factory(
                node_config=node,
                context_manager=self.context_manager,
                llm_config=getattr(node, 'llm_config', None),
                callbacks=self.callbacks,
                tool_service=self.tool_service
            )
            for node_id, node in self.nodes.items()
        }
        logger.info(f"Initialized ScriptChain with {len(nodes)} nodes in {len(self.levels)} levels")

    async def execute(self) -> NodeExecutionResult:
        start_time = datetime.utcnow()
        results = {}
        errors = []
        node_executor = NodeExecutor(
            context_manager=self.context_manager,
            chain_id=self.chain_id,
            persist_intermediate_outputs=self.persist_intermediate_outputs,
            callbacks=self.callbacks,
            tool_service=self.tool_service,
            initial_context=self.initial_context
        )
        logger.info(f"Starting execution of chain '{self.name}' (ID: {self.chain_id})")
        for level_num in sorted(self.levels.keys()):
            level_node_ids = self.levels[level_num]
            level_nodes = [self.node_instances[node_id] for node_id in level_node_ids]
            level_results = await self._execute_level(level_nodes, node_executor, results)
            for node_id, result_obj in level_results.items():
                if result_obj.success:
                    results[node_id] = result_obj
                    if hasattr(result_obj, 'usage') and result_obj.usage:
                        self.metrics.update(node_id, result_obj)
                else:
                    errors.append(f"Node {node_id} failed: {result_obj.error}")
            if errors and not self._should_continue(errors):
                break
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Completed execution of chain '{self.name}' (ID: {self.chain_id}) in {duration:.2f} seconds")
        return NodeExecutionResult(
            success=len(errors) == 0,
            output=results,
            error="\n".join(errors) if errors else None,
            metadata=NodeMetadata(
                node_id=self.chain_id,
                node_type="script_chain",
                name=self.name,
                version="1.0.0",
                start_time=start_time,
                end_time=end_time,
                duration=duration
            ),
            execution_time=duration,
            token_stats=self.metrics.as_dict()
        )

    async def _execute_level(self, level_nodes: List[Any], node_executor: NodeExecutor, accumulated_results: Dict[str, NodeExecutionResult]) -> Dict[str, NodeExecutionResult]:
        semaphore = asyncio.Semaphore(self.max_parallel)
        async def process_node(node):
            async with semaphore:
                return await node_executor.execute_node(node, accumulated_results)
        tasks = [process_node(node) for node in level_nodes]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def _should_continue(self, errors: List[str]) -> bool:
        if not errors:
            return True
        failed_nodes = set()
        for error in errors:
            if "Node " in error and " failed:" in error:
                try:
                    node_id = error.split("Node ")[1].split(" failed:")[0]
                    failed_nodes.add(node_id)
                except (IndexError, AttributeError):
                    continue
        for level_num in sorted(self.levels.keys()):
            for node_id in self.levels[level_num]:
                node = self.nodes[node_id]
                if node_id in failed_nodes:
                    continue
                depends_on_failed_node = any(dep_id in failed_nodes for dep_id in getattr(node, 'dependencies', []))
                if not depends_on_failed_node:
                    logger.info(f"Chain execution continuing: Node '{node_id}' can still execute independently")
                    return True
        logger.warning(f"Chain execution stopping: All remaining nodes depend on failed nodes: {failed_nodes}")
        return False

    # Expose graph queries for external use
    def get_node_dependencies(self, node_id: str) -> List[str]:
        return self.graph.get_node_dependencies(node_id)
    def get_node_dependents(self, node_id: str) -> List[str]:
        return self.graph.get_node_dependents(node_id)
    def get_node_level(self, node_id: str) -> int:
        return self.graph.get_node_level(node_id)
    def get_level_nodes(self, level: int) -> List[str]:
        return self.levels.get(level, [])
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.as_dict()