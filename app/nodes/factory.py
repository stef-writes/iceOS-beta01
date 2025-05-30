from app.nodes.ai_node import AiNode
# from app.nodes.tool_node import ToolNode  # (future)
# from app.nodes.router_node import RouterNode  # (future)

def node_factory(node_config, context_manager, llm_config=None, callbacks=None, tool_service=None):
    """
    Factory function to instantiate the correct node class based on node_config.type.
    Extend this as you add more node types.
    """
    if node_config.type == "ai":
        return AiNode(node_config, context_manager, llm_config, callbacks, tool_service=tool_service)
    # elif node_config.type == "tool":
    #     return ToolNode(node_config, ...)
    # elif node_config.type == "router":
    #     return RouterNode(node_config, ...)
    else:
        raise ValueError(f"Unknown node type: {node_config.type}") 