from app.nodes.base import BaseNode
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata
from app.services.tool_service import ToolService
from datetime import datetime
from typing import Dict, Any, Optional

class ToolNode(BaseNode):
    """Node for executing a tool via ToolService."""
    def __init__(self, config: NodeConfig, tool_service: Optional[ToolService] = None):
        super().__init__(config)
        self.tool_service = tool_service or ToolService()

    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        start_time = datetime.utcnow()
        tool_name = self.config.name or self.config.id
        tool_args = context
        try:
            result = await self.tool_service.execute(tool_name, tool_args)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            if result.get("success"):
                return NodeExecutionResult(
                    success=True,
                    output=result.get("output"),
                    metadata=NodeMetadata(
                        node_id=self.config.id,
                        node_type=self.config.type,
                        version=self.config.metadata.version if self.config.metadata else "1.0.0",
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        provider=self.config.provider
                    ),
                    execution_time=duration
                )
            else:
                return NodeExecutionResult(
                    success=False,
                    error=result.get("error", "Unknown error"),
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
                    execution_time=duration
                )
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            return NodeExecutionResult(
                success=False,
                error=str(e),
                metadata=NodeMetadata(
                    node_id=self.config.id,
                    node_type=self.config.type,
                    version=self.config.metadata.version if self.config.metadata else "1.0.0",
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    provider=self.config.provider,
                    error_type=type(e).__name__
                ),
                execution_time=duration
            ) 