"""
API routes for the workflow engine
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ValidationError
from app.models.node_models import NodeConfig, NodeExecutionResult
from app.models.config import LLMConfig, MessageTemplate
from app.nodes.factory import node_factory
from app.chains.script_chain import ScriptChain
from app.chains.errors import CircularDependencyError
from app.utils.context.manager import GraphContextManager
from app.utils.logging import logger
from app.services.tool_service import ToolService
import traceback

router = APIRouter(prefix="/api/v1")

# Create a singleton context manager
context_manager = GraphContextManager()

# Create a singleton ToolService and register default tools
singleton_tool_service = ToolService()
ToolService.register_default_tools(singleton_tool_service)

class NodeRequest(BaseModel):
    """Request model for node operations"""
    config: NodeConfig
    context: Optional[Dict[str, Any]] = None

class ChainRequest(BaseModel):
    """Request model for chain operations"""
    nodes: List[NodeConfig]
    context: Optional[Dict[str, Any]] = None
    persist_intermediate_outputs: bool = True

@router.post("/nodes/text-generation", response_model=NodeExecutionResult)
async def create_text_generation_node(request: NodeRequest):
    """Create and execute a text generation node"""
    try:
        # Extract llm_config from the config object
        llm_config = request.config.llm_config
        # Remove API key check; assume server-side env vars are used
        # if not llm_config.api_key:
        #     raise HTTPException(status_code=422, detail="API key is required")
        node = node_factory(request.config, context_manager, llm_config, tool_service=singleton_tool_service)
        result = await node.execute(request.context or {})
        
        # If execution was successful, update the context manager with its output
        if result.success and result.output:
            context_manager.update_context(request.config.id, result.output)
            
        return result
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in text generation node: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/chains/execute", response_model=NodeExecutionResult)
async def execute_chain(request: ChainRequest):
    """Execute a chain of nodes"""
    try:
        # Create chain with nodes, passing the shared context_manager and initial_context
        chain = ScriptChain(
            nodes=request.nodes,
            context_manager=context_manager,
            persist_intermediate_outputs=request.persist_intermediate_outputs,
            tool_service=singleton_tool_service,
            initial_context=request.context or {}
        )
        
        # Execute chain
        result = await chain.execute()
        return result
    except CircularDependencyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing chain: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/nodes/{node_id}/context")
async def get_node_context(node_id: str):
    """Get context for a specific node"""
    try:
        context = context_manager.get_context(node_id)
        if not context:
            raise HTTPException(status_code=404, detail=f"Context not found for node {node_id}")
        return context
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node context: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/nodes/{node_id}/context")
async def clear_node_context(node_id: str):
    """Clear context for a specific node"""
    logger.info(f"API HANDLER: clear_node_context CALLED for node_id: {node_id}")
    try:
        # Attempt to clear the context. The manager's clear_context
        # should handle cases where the node_id doesn't exist gracefully.
        context_manager.clear_context(node_id)
        logger.info(f"API HANDLER: clear_node_context SUCCESS for node_id: {node_id}")
        return {"message": f"Context cleared for node {node_id}"}
    except Exception as e:
        logger.error(f"API HANDLER: Error clearing node context for {node_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")