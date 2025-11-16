"""
Inspect MCP Server Tools

This script connects to the MCP server and lists all available tools with their descriptions.
"""

import asyncio

from agent_framework import MCPStreamableHTTPTool
from rich import print
from rich.console import Console
from rich.table import Table

console = Console()


async def main():
    # Connect to the agent MCP server
    mcp_tool = MCPStreamableHTTPTool(
        name="agent_payment_server",
        url="http://127.0.0.1:8001/mcp",
    )

    async with mcp_tool:
        print("\n[bold green]🔍 Inspecting Agent MCP Server[/bold green]\n")

        # Get the list of available functions (tools)
        tools = mcp_tool.functions

        if not tools:
            print("[yellow]No tools found![/yellow]")
            return

        # Create a table to display the tools
        table = Table(
            title="Available MCP Tools", show_header=True, header_style="bold magenta"
        )
        table.add_column("Tool Name", style="cyan", width=30)
        table.add_column("Description", style="white", width=60)
        table.add_column("Parameters", style="green", width=40)

        for tool in tools:
            # Get tool metadata
            name = tool.name
            description = tool.description or "(no description)"

            # Get parameter info
            if hasattr(tool, "function") and hasattr(tool.function, "__annotations__"):
                params = list(tool.function.__annotations__.keys())
                params_str = ", ".join(params) if params else "(no parameters)"
            else:
                params_str = "(info not available)"

            table.add_row(name, description, params_str)

        console.print(table)

        # Print detailed info for each tool
        print("\n[bold]Detailed Tool Information:[/bold]\n")
        for i, tool in enumerate(tools, 1):
            print(f"[bold cyan]{i}. {tool.name}[/bold cyan]")
            print(f"   Description: {tool.description or '(none)'}")
            if hasattr(tool, "function"):
                print(f"   Function: {tool.function.__name__}")
                if hasattr(tool.function, "__doc__") and tool.function.__doc__:
                    print(f"   Docstring: {tool.function.__doc__.strip()}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
