from typing import Any, Dict, Optional, Callable, List
from app.models.node_models import ContextFormat

class BaseContextFormatter:
    """Abstract base class for context formatters."""
    def format(self, content: Any, rule, format_specs: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

class ContextFormatter(BaseContextFormatter):
    """Handles formatting of context data for nodes, with hooks and optional schema validation."""
    def __init__(self):
        self.format_handlers = {
            ContextFormat.TEXT: self._handle_text_format,
            ContextFormat.JSON: self._handle_json_format,
            ContextFormat.MARKDOWN: self._handle_markdown_format,
            ContextFormat.CODE: self._handle_code_format,
            ContextFormat.CUSTOM: self._handle_custom_format
        }
        self.hooks: List[Callable[[str, Any], None]] = []  # hooks for observability

    def register_hook(self, hook: Callable[[str, Any], None]):
        """Register a hook to be called on every format operation."""
        self.hooks.append(hook)

    def _run_hooks(self, format_type: str, content: Any):
        for hook in self.hooks:
            hook(format_type, content)

    def _handle_text_format(self, content: Any) -> str:
        return str(content)

    def _handle_json_format(self, content: Any) -> str:
        import json
        if isinstance(content, str):
            try:
                return json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                return content
        return json.dumps(content, indent=2)

    def _handle_markdown_format(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return f"```markdown\n{content}\n```"

    def _handle_code_format(self, content: Any) -> str:
        if isinstance(content, str):
            return f"```\n{content}\n```"
        return f"```\n{str(content)}\n```"

    def _handle_custom_format(self, content: Any, format_spec: Dict[str, Any]) -> str:
        return str(content)

    def format(self, content: Any, rule, format_specs: Optional[Dict[str, Any]] = None) -> str:
        format_type = getattr(rule, 'format', None)
        self._run_hooks(str(format_type), content)
        if format_type in self.format_handlers:
            if getattr(format_type, 'name', None) == "CUSTOM" and format_specs:
                return self.format_handlers[format_type](content, format_specs)
            return self.format_handlers[format_type](content)
        return str(content)

    def validate_schema(self, content: Any, schema: Optional[Dict[str, str]] = None) -> bool:
        """Simple schema validation: checks keys and types if schema is provided."""
        if not schema:
            return True
        if not isinstance(content, dict):
            return False
        for key, typ in schema.items():
            if key not in content:
                return False
            if not isinstance(content[key], eval(typ)):
                return False
        return True
