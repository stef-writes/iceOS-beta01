# Public interfaces -------------------------------------------------------

from ice_sdk.base_node import BaseNode
from ice_sdk.base_tool import BaseTool
from app.models.node_models import (
    NodeConfig,
    NodeExecutionResult,
    NodeMetadata,
    NodeExecutionRecord,
    UsageMetadata,
)

__all__ = [
    "BaseNode",
    "BaseTool",
    "ToolService",
    "NodeConfig",
    "NodeExecutionResult",
    "NodeMetadata",
    "NodeExecutionRecord",
    "UsageMetadata",
]

# The SDK also exposes registries for plugins (tools, nodes, etc.)
from importlib import metadata as _metadata
from typing import Dict, Type

_TOOL_ENTRYPOINT_GROUP = "ice.tools"
_NODE_ENTRYPOINT_GROUP = "ice.nodes"

_tools_cache: Dict[str, Type[BaseTool]] | None = None
_nodes_cache: Dict[str, Type[BaseNode]] | None = None

def _load_entrypoints(group: str):
    """Helper to lazily load entry points from *group*."""
    try:
        eps = _metadata.entry_points()
    except Exception:  # pragma: no cover – importlib.metadata behaviour differs <3.10
        eps = {}
    return eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])

def iter_tool_classes():
    global _tools_cache
    if _tools_cache is None:
        _tools_cache = {}
        for ep in _load_entrypoints(_TOOL_ENTRYPOINT_GROUP):
            try:
                cls = ep.load()
                if hasattr(cls, "name"):
                    _tools_cache[cls.name] = cls
            except Exception:  # pylint: disable=broad-except
                # Ignore broken entrypoints but keep going
                continue

    # -----------------------------------------------------------------
    # Fallback: ensure built-in tools packaged in `app.tools.builtins` are
    # available even when entry points are not configured (e.g., in a dev
    # checkout).  This keeps backward-compatibility while encouraging
    # external packages to rely on entry-points in production.
    # -----------------------------------------------------------------
    if not _tools_cache:
        try:
            from importlib import import_module
            import pkgutil

            builtins_pkg = import_module("app.tools.builtins")
            for _, mod_name, _ in pkgutil.iter_modules(builtins_pkg.__path__):
                try:
                    mod = import_module(f"app.tools.builtins.{mod_name}")
                    for attr in mod.__dict__.values():
                        if isinstance(attr, type) and issubclass(attr, BaseTool) and attr is not BaseTool:
                            _tools_cache[attr.name] = attr
                except Exception:
                    continue
        except ModuleNotFoundError:
            # builtins package missing – ignore
            pass

    return _tools_cache.values()

def get_tool_class(name: str) -> Type[BaseTool] | None:
    if _tools_cache is None:
        list(iter_tool_classes())  # populate cache
    return _tools_cache.get(name)

def iter_node_classes():
    global _nodes_cache
    if _nodes_cache is None:
        _nodes_cache = {}
        for ep in _load_entrypoints(_NODE_ENTRYPOINT_GROUP):
            try:
                cls = ep.load()
                # Use the declared discriminator (e.g., 'ai', 'tool') if
                # it exists, otherwise default to the lowercase class
                # name.  This aligns with `NodeConfig.type` semantics.
                discriminator = getattr(cls, "type", cls.__name__.lower())
                _nodes_cache[discriminator] = cls
            except Exception:
                continue
    return _nodes_cache.values()

def get_node_class(name: str):
    if _nodes_cache is None:
        list(iter_node_classes())
    return _nodes_cache.get(name)

# ---------------------------------------------------------------------------
# Default structlog configuration – JSON to stdout when not configured by
# the host application.  This keeps the change backwards-compatible and
# ensures we always have structured logs for cloud ingestion.
# ---------------------------------------------------------------------------

import os as _os
import logging as _logging

import structlog as _structlog

if not _structlog.is_configured():  # Avoid overriding app-specific config.
    # Resolve logging level string (e.g., "INFO", "DEBUG") to its numeric value.
    _level_name = _os.getenv("ICE_LOG_LEVEL", "INFO").upper()
    _level = getattr(_logging, _level_name, _logging.INFO)

    _structlog.configure(
        processors=[
            _structlog.processors.TimeStamper(fmt="iso"),
            _structlog.processors.add_log_level,
            _structlog.processors.StackInfoRenderer(),
            _structlog.processors.format_exc_info,
            _structlog.processors.JSONRenderer(),
        ],
        wrapper_class=_structlog.make_filtering_bound_logger(min_level=_level),
        cache_logger_on_first_use=True,
    )

# Deferred import to avoid circular dependency with tool_service accessing
# iter_tool_classes during its module import.
from ice_sdk.tool_service import ToolService 