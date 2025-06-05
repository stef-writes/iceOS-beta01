import networkx as nx
from typing import List, Dict, Any
from .errors import CircularDependencyError

class DependencyGraph:
    """
    Handles dependency graph construction, cycle detection, level assignment, and queries for ScriptChain.
    """
    def __init__(self, nodes: List[Any]):
        self.graph = nx.DiGraph()
        self.node_levels = {}
        self._build_graph(nodes)
        self._assign_levels(nodes)

    def _build_graph(self, nodes: List[Any]):
        node_ids = {node.id for node in nodes}
        for node in nodes:
            self.graph.add_node(node.id, level=0)
            for dep in getattr(node, 'dependencies', []):
                if dep not in node_ids:
                    raise ValueError(f"Dependency {dep} not found for node {node.id}")
                self.graph.add_edge(dep, node.id)
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                cycle_str = " -> ".join(cycles[0])
                raise CircularDependencyError(f"Circular dependency detected: {cycle_str}")
        except nx.NetworkXNoCycle:
            pass

    def _assign_levels(self, nodes: List[Any]):
        node_map = {node.id: node for node in nodes}
        for node_id in nx.topological_sort(self.graph):
            node = node_map[node_id]
            node.level = max(
                (node_map[dep].level for dep in getattr(node, 'dependencies', [])),
                default=-1
            ) + 1
            self.node_levels[node_id] = node.level

    def get_level_nodes(self) -> Dict[int, List[str]]:
        levels = {}
        for node_id, level in self.node_levels.items():
            if level not in levels:
                levels[level] = []
            levels[level].append(node_id)
        return levels

    def get_node_dependencies(self, node_id: str) -> List[str]:
        return list(self.graph.predecessors(node_id))

    def get_node_dependents(self, node_id: str) -> List[str]:
        return list(self.graph.successors(node_id))

    def get_node_level(self, node_id: str) -> int:
        return self.node_levels[node_id]
