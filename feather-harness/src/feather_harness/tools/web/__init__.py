"""Provider-neutral public web tools."""

from typing import Protocol

from .fetch import WebFetchProvider, web_fetch_tool
from .search import WebSearchProvider, web_search_tool

WEB_TOOL_NAMES = ("web_search", "web_fetch")


class WebProvider(WebSearchProvider, WebFetchProvider, Protocol):
    pass


__all__ = ["WEB_TOOL_NAMES", "WebProvider", "web_fetch_tool", "web_search_tool"]
