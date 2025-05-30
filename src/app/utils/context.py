"""
Context management for LLM node execution
"""

from typing import Dict, List, Optional, Any, Union
import networkx as nx
from app.models.node_models import NodeExecutionResult, NodeMetadata, ContextFormat, ContextRule, InputMapping
from app.utils.logging import logger
import json
import os
import fcntl
from datetime import datetime
from uuid import uuid4

class GraphContextManager:
    """Manages context for graph-based LLM node execution"""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        graph: Optional[nx.DiGraph] = None,
        context_store_path: Optional[str] = None
    ):
        """Initialize context manager.
        
        Args:
            max_tokens: Maximum number of tokens to include in context
            graph: Optional graph structure for dependency tracking
            context_store_path: Optional path to the context store JSON file.
                                Defaults to SCRIPTCHAIN_CONTEXT_STORE_PATH env var,
                                then app/data/context_store.json in the workspace.
        """
        self.max_tokens = max_tokens
        self.graph = graph or nx.DiGraph()
        self.context_cache = {}
        
        # Determine context_store_path
        if context_store_path:
            self.context_store_path = context_store_path
        elif os.getenv("SCRIPTCHAIN_CONTEXT_STORE_PATH"):
            self.context_store_path = os.environ["SCRIPTCHAIN_CONTEXT_STORE_PATH"]
        else:
            # Default path relative to a discoverable project root or current working directory
            # For robustness, ensure this default path is well-defined.
            # Using current working directory as a fallback if project root isn't easily found.
            # A more robust solution might involve searching upwards for a .git folder or project marker.
            default_workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.context_store_path = os.path.join(default_workspace_root, "app", "data", "context_store.json")
            logger.info(f"Context store path not specified, defaulting to: {self.context_store_path}")

        self.format_handlers = {
            ContextFormat.TEXT: self._handle_text_format,
            ContextFormat.JSON: self._handle_json_format,
            ContextFormat.MARKDOWN: self._handle_markdown_format,
            ContextFormat.CODE: self._handle_code_format,
            ContextFormat.CUSTOM: self._handle_custom_format
        }
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.context_store_path), exist_ok=True)
        
        # Initialize empty context store if it doesn't exist
        if not os.path.exists(self.context_store_path):
            with open(self.context_store_path, 'w') as f:
                json.dump({}, f)
        
        # Load existing context
        self._load_context()
        
    def _load_context(self) -> None:
        """Load context from the store file"""
        try:
            if os.path.exists(self.context_store_path):
                with open(self.context_store_path, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        self.context_cache = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding context store: {str(e)}")
            self.context_cache = {}
        except Exception as e:
            logger.error(f"Error loading context store: {str(e)}")
            self.context_cache = {}
            
    def _save_context(self) -> None:
        """Save context to the store file"""
        try:
            with open(self.context_store_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(self.context_cache, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Error saving context store: {str(e)}")
            raise

    def _handle_text_format(self, content: Any) -> str:
        """Handle text format conversion"""
        return str(content)

    def _handle_json_format(self, content: Any) -> str:
        """Handle JSON format conversion"""
        if isinstance(content, str):
            try:
                return json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                return content
        return json.dumps(content, indent=2)

    def _handle_markdown_format(self, content: Any) -> str:
        """Handle markdown format conversion"""
        if isinstance(content, str):
            return content
        return f"```markdown\n{content}\n```"

    def _handle_code_format(self, content: Any) -> str:
        """Handle code format conversion"""
        if isinstance(content, str):
            return f"```\n{content}\n```"
        return f"```\n{str(content)}\n```"

    def _handle_custom_format(self, content: Any, format_spec: Dict[str, Any]) -> str:
        """Handle custom format conversion"""
        # Implement custom formatting logic based on format_spec
        return str(content)

    def format_context(self, content: Any, rule: ContextRule, format_specs: Optional[Dict[str, Any]] = None) -> str:
        """Format context according to specified rules"""
        if rule.format in self.format_handlers:
            if rule.format == ContextFormat.CUSTOM and format_specs:
                return self.format_handlers[rule.format](content, format_specs)
            return self.format_handlers[rule.format](content)
        return str(content)

    def get_node_output(self, node_id: str) -> Any:
        """Get the output of a specific node from persistent storage"""
        return self.context_cache.get(node_id, {}).get('data', {})

    def get_context(self, node_id: str) -> Dict[str, Any]:
        """Get context for a specific node.
        
        Args:
            node_id: ID of the node to get context for
            
        Returns:
            Dictionary containing context data for the node
        """
        # First check cache
        if node_id in self.context_cache:
            return self.context_cache[node_id].get('data', {})
            
        # Then check file
        try:
            if os.path.exists(self.context_store_path):
                with open(self.context_store_path, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                        context = data.get(node_id, {}).get('data', {})
                        # Update cache
                        self.context_cache[node_id] = {
                            'data': context,
                            'version': data.get(node_id, {}).get('version', str(uuid4())),
                            'timestamp': data.get(node_id, {}).get('timestamp', datetime.utcnow().isoformat())
                        }
                        return context
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Error reading context for node {node_id}: {str(e)}")
            
        return {}
        
    def set_context(self, node_id: str, context: Dict[str, Any]) -> None:
        """Set context for a node.
        
        Args:
            node_id: ID of the node to set context for
            context: Context dictionary to set
        """
        try:
            self.context_cache[node_id] = {
                'data': context,
                'version': str(uuid4()),
                'timestamp': datetime.utcnow().isoformat()
            }
            self._save_context()
        except Exception as e:
            logger.error(f"Error setting context for node {node_id}: {str(e)}")
            raise

    def clear_context(self, node_id: Optional[str] = None) -> None:
        """Clear context cache for a specific node or all nodes"""
        try:
            if node_id:
                # Clear from cache
                self.context_cache.pop(node_id, None)
                
                # Clear from file
                if os.path.exists(self.context_store_path):
                    with open(self.context_store_path, 'r+') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        try:
                            try:
                                data = json.load(f)
                            except json.JSONDecodeError: # Handle case where file is empty or malformed
                                data = {} # Treat as empty dict, item to pop won't be there
                            
                            data.pop(node_id, None)
                            f.seek(0)
                            json.dump(data, f, indent=2)
                            f.truncate()
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                # Clear all
                self.context_cache.clear()
                if os.path.exists(self.context_store_path):
                    with open(self.context_store_path, 'w') as f:
                        json.dump({}, f)
        except Exception as e:
            logger.error(f"Error clearing context for node {node_id}: {str(e)}")
            raise

    def update_context(self, node_id: str, content: Any, execution_id: Optional[str] = None) -> None:
        """Update the context for a specific node"""
        try:
            context_entry = {
                'data': content,
                'version': str(uuid4()),
                'timestamp': datetime.utcnow().isoformat()
            }
            if execution_id:
                context_entry['execution_id'] = execution_id
            
            # Update cache
            self.context_cache[node_id] = context_entry
            
            # Update file
            # Ensure file is created if it doesn't exist, especially if clear_context cleared everything
            # or if this is the very first write.
            mode = 'r+' if os.path.exists(self.context_store_path) else 'w'
            # Using 'w' for creation might truncate if used incorrectly with 'r+' logic,
            # so careful handling is needed. Best to ensure file exists before 'r+'.
            # The __init__ ensures the file and dir exist, but if all context was cleared,
            # the file might be empty {}.
            
            # Simplified file update: load, update, save.
            # This is less efficient than r+ for large files but safer for JSON.
            current_data = {}
            if os.path.exists(self.context_store_path):
                try:
                    with open(self.context_store_path, 'r') as f_read:
                        fcntl.flock(f_read.fileno(), fcntl.LOCK_SH)
                        try:
                            current_data = json.load(f_read)
                        except json.JSONDecodeError:
                            logger.warning(f"Context store file {self.context_store_path} is malformed. Will overwrite with new context.")
                            current_data = {} # Reset if malformed
                        finally:
                            fcntl.flock(f_read.fileno(), fcntl.LOCK_UN)
                except FileNotFoundError:
                     # This case should ideally be handled by __init__, but as a safeguard:
                    logger.info(f"Context store file {self.context_store_path} not found during update. Creating new one.")
                    pass # File will be created by 'w' mode below
                except Exception as e_read:
                    logger.error(f"Error reading context store for update: {str(e_read)}. Proceeding to overwrite.")
                    current_data = {}


            current_data[node_id] = self.context_cache[node_id] # Update with the new entry

            with open(self.context_store_path, 'w') as f_write: # Overwrite the file
                fcntl.flock(f_write.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(current_data, f_write, indent=2)
                finally:
                    fcntl.flock(f_write.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            logger.error(f"Error updating context for node {node_id}: {str(e)}")
            raise

    def validate_context_rules(self, rules: Dict[str, ContextRule]) -> bool:
        """Validate context rules for consistency"""
        for node_id, rule in rules.items():
            if node_id not in self.graph.nodes:
                logger.warning(f"Context rule specified for non-existent node: {node_id}")
                return False
            if rule.max_tokens and rule.max_tokens > self.max_tokens:
                logger.warning(f"Context rule max_tokens exceeds system limit for node: {node_id}")
                return False
        return True

    def log_error(self, node_id: str, error: Exception) -> None:
        """Log error for a node.
        
        Args:
            node_id: ID of the node that encountered an error
            error: Exception that was raised
        """
        logger.error(f"Error in node {node_id}: {str(error)}")