"""
Client for Agent MCP StreamableHTTP Server

This example demonstrates how to connect to the Agent MCP server via StreamableHTTP transport.
"""

import asyncio
import os

from agent_framework import MCPStreamableHTTPTool
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from rich import print

load_dotenv()


def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    return f"The weather in {location} is sunny and warm."


async def main():
    # Connect to the StreamableHTTP-based MCP server
    mcp_tool = MCPStreamableHTTPTool(
        name="agent_payment_server",
        url="http://localhost:8001/mcp",  # StreamableHTTP endpoint
        description="Agent-based payment server with balance checking and payment processing",
    )

    async with (
        AzureCliCredential() as credential,
        mcp_tool,
        AzureAIAgentClient(async_credential=credential).create_agent(
            instructions="""
            You are a helpful assistant that helps users with their payments.
            You can check balances and process payments.
            Keep responses concise (max 50 words).
            """,
            name="PaymentClientAgent",
            tools=[mcp_tool, get_weather],
        ) as agent,
    ):
        print("[bold green]🚀 Agent MCP StreamableHTTP Client Demo[/bold green]\n")

        # Test 1: Check balance
        print("[bold]Test 1: Check Balance[/bold]")
        print("-" * 50)
        query = "What's my current balance?"
        print(f"[bold yellow]User:[/bold yellow] {query}")
        result = await agent.run(query)
        print(f"[cyan]Agent:[/cyan] {result.text}\n")

        # Test 2: Make payment
        print("[bold]Test 2: Make Payment[/bold]")
        print("-" * 50)
        query = "Make a payment of $30 for groceries"
        print(f"[bold yellow]User:[/bold yellow] {query}")
        result = await agent.run(query)
        print(f"[cyan]Agent:[/cyan] {result.text}\n")

        # Test 3: Check balance again
        print("[bold]Test 3: Check Balance After Payment[/bold]")
        print("-" * 50)
        query = "What's my balance now?"
        print(f"[bold yellow]User:[/bold yellow] {query}")
        result = await agent.run(query)
        print(f"[cyan]Agent:[/cyan] {result.text}\n")

        # Test 4: Get weather (not using the agent MCP server)
        print("[bold]Test 4: Get Weather (not using the agent MCP server)[/bold]")
        print("-" * 50)
        query = "What's the weather like in Zurich?"
        print(f"[bold yellow]User:[/bold yellow] {query}")
        result = await agent.run(query)
        print(f"[cyan]Agent:[/cyan] {result.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
