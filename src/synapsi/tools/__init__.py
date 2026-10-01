"""Tools agents can call. Every tool result becomes evidence with provenance.

Arbitrary code execution is intentionally not built in. If you need a Python
tool, register your own sandboxed implementation with :class:`Tool`.
"""

from synapsi.tools.base import Tool, ToolResult, ToolSource, tool
from synapsi.tools.calculator import calculator_tool
from synapsi.tools.documents import DocumentStore, document_search_tool
from synapsi.tools.sql import sql_tool
from synapsi.tools.web import (
    BraveSearch,
    StaticSearch,
    TavilySearch,
    fetch_url_tool,
    web_search_tool,
)

__all__ = [
    "BraveSearch",
    "DocumentStore",
    "StaticSearch",
    "TavilySearch",
    "Tool",
    "ToolResult",
    "ToolSource",
    "calculator_tool",
    "document_search_tool",
    "fetch_url_tool",
    "sql_tool",
    "tool",
    "web_search_tool",
]
