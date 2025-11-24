"""
Agent as MCP Server
===================

This script demonstrates how to turn an Agent Framework agent into an MCP server
that can be accessed remotely via StreamableHTTP protocol.

Key Concepts:
-------------
- **Agent-as-a-Service**: Expose agent capabilities as MCP tools over HTTP
- **StreamableHTTPSessionManager**: MCP protocol handler for HTTP transport
- **Starlette/Uvicorn**: ASGI web framework for serving the MCP endpoint
- **Tool Wrapping**: Convert agent.run() into an MCP-compatible tool interface

Architecture:
-------------
Agent Framework Agent → MCP Server Wrapper → HTTP Endpoint → Remote Clients
                     ↓
              [get_balance, make_payment tools]

"""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from agent_framework import ai_function
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import Field
from starlette.applications import Starlette
from starlette.routing import Route

load_dotenv()

balance: float = 100.0


def get_balance() -> str:
    """Get the balance for a given user."""
    return f"The balance for is ${balance}."


@ai_function(
    name="make_payment",
    description="Make a payment for a given user.",
)
def _make_payment(
    amount: Annotated[float, Field(description="The amount to pay.")],
    reason: Annotated[str, Field(description="The reason for the payment.")],
) -> str:
    global balance
    balance -= amount
    return f"Processing payment of ${amount} for {reason}."


# Create agent and MCP server at module level
agent = AzureAIAgentClient(async_credential=AzureCliCredential()).create_agent(
    instructions="""
    You are a helpful assistant, your task is to help user making orders online.
    You allow users to check their balance and make payments.
    If the balance is not sufficient, you should inform the user and not proceed with the payment.
    """,
    name="PaymentAgent",
    description=(
        "Useful for managing payments. "
        "It can check account balances and process payments for various services. "
        "The agent ensures sufficient balance before processing transactions."
    ),
    tools=[get_balance, _make_payment],
)

# Convert agent to MCP server with proper metadata
# The 'instructions' parameter provides a description for the MCP tool
mcp_server = agent.as_mcp_server()

# Create session manager
session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    json_response=False,
    stateless=False,
)


@asynccontextmanager
async def lifespan(app: Starlette):
    """Lifespan context manager for the Starlette app."""
    async with session_manager.run():
        yield


class MCPEndpoint:
    """ASGI app wrapper for the MCP endpoint."""

    async def __call__(self, scope, receive, send):
        """Handle ASGI requests."""
        await session_manager.handle_request(scope, receive, send)


# Create Starlette app
app = Starlette(
    routes=[
        Route("/mcp", endpoint=MCPEndpoint(), methods=["GET", "POST", "DELETE"]),
    ],
    lifespan=lifespan,
)


async def main():
    """Main entry point."""
    # Configure and run uvicorn server
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info",
    )

    server = uvicorn.Server(config)
    print("🚀 Agent MCP Server (StreamableHTTP) starting on http://127.0.0.1:8001")
    print("   MCP endpoint: http://127.0.0.1:8001/mcp")

    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
