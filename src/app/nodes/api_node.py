from app.nodes.base import BaseNode
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata
from datetime import datetime
from typing import Dict, Any, Optional
import httpx

class ApiNode(BaseNode):
    """Node for making an API call."""
    def __init__(self, config: NodeConfig):
        super().__init__(config)

    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        start_time = datetime.utcnow()
        try:
            # Expect API details in config (e.g., url, method, headers, params, data)
            url = self.config.templates.get('url') or self.config.prompt
            method = self.config.templates.get('method', 'GET').upper()
            headers = self.config.templates.get('headers', {})
            params = context.get('params', {})
            data = context.get('data', {})
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, headers=headers, params=params, json=data)
                response.raise_for_status()
                output = {
                    'status_code': response.status_code,
                    'body': response.json() if 'application/json' in response.headers.get('content-type', '') else response.text
                }
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            return NodeExecutionResult(
                success=True,
                output=output,
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