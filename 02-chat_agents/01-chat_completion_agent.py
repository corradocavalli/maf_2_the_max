import asyncio

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
from rich import print


async def main():
    agent = AzureOpenAIChatClient(credential=AzureCliCredential()).create_agent(
        instructions="You are good at telling jokes ", name="Joker"
    )

    result = await agent.run("Tell me a joke about a programmer.")
    print(f"[green]{result.text}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
