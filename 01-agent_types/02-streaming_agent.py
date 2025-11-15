import asyncio

from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from rich import print

load_dotenv()


async def main():
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="StreamingAgent", instructions="You are a helpful assistant."
        ) as agent,
    ):
        async for chunk in agent.run_stream("Tell me a short story"):
            if chunk.text:
                print(f"[yellow]{chunk.text}[/yellow]", end="")


if __name__ == "__main__":
    asyncio.run(main())
