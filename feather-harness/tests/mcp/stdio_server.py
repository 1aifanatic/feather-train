"""A real local MCP server for the stdio adapter tests."""

from __future__ import annotations

import asyncio

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("stdio-echo")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="echo",
            description="Echo a value",
            inputSchema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name, arguments):
    return [TextContent(type="text", text=arguments["value"])]


async def main() -> None:
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
