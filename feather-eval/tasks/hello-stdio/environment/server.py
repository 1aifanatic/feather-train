#!/usr/bin/env python3
"""Stdio MCP server for the in-tree E2B Harbor smoke."""

from mcp.server.fastmcp import FastMCP

TOKEN = "harbor-stdio-ok"
server = FastMCP("token")


@server.tool()
def get_token() -> str:
    """Return the task token."""
    return TOKEN


if __name__ == "__main__":
    server.run()
