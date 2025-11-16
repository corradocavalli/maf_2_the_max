import asyncio
import os

from agent_framework import HostedMCPTool, MCPStreamableHTTPTool
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from rich import print

load_dotenv()


async def main():
    mcp_server = MCPStreamableHTTPTool(
        name="payment_server",
        url="http://localhost:8000/mcp",
    )
    mcp_server.sampling_callback = lambda chunk: print(f"[green]{chunk}[/green]")
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(
            async_credential=credential,
            model_deployment_name=os.getenv(
                "AZURE_AI_MODEL_DEPLOYMENT_NAME_FOR_TOOLS", "gpt-4o"
            ),  # Note: For hosted MCP tool we need a model that supports chat-responses API
        ).create_agent(
            instructions="""
            You are a helpful assistant, your task is to help user making orders online and ask about current time.
            You allow users to check their balance and make payments.
            If the balance is not sufficient, you should inform the user and not proceed with the payment.
            Please provide concise answers (max 50 words).
            """,
            name="PaymentAgent",
            tools=mcp_server,
        ) as agent,
    ):
        # Check balance
        query = "Hello, I would like to check my balance."
        print(f"[bold][yellow]User:[/bold][/yellow] {query}")
        result = await agent.run(query)
        print(f"[yellow]{result.text}[/yellow]")

        # # Make payment
        query = "Make a payment of $45 for my internet subscription."
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")
        result = await agent.run(query)
        print(f"[yellow]{result.text}[/yellow]")

        # Use Learn MCP Server
        query = "What is current time in new york?"
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")
        result = await agent.run(
            query,
            tools=HostedMCPTool(
                name="Time MCP Server",
                url="https://time.mcp.inevitable.fyi/mcp",
                approval_mode="never_require",  # Note: Default is "always_require"
            ),
        )
        print(f"[yellow]{result}[/yellow]")


if __name__ == "__main__":
    print(f"[bold][yellow]MCP TOOLS DEMO[/bold][/yellow]\n")
    asyncio.run(main())
