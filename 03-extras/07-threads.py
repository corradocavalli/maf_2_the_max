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
            name="HelperAgent",
            instructions="You are a helpful assistant.",
            store=None,  # None/True= default/set to AI Foundry server side storage. False= Create a local in-memory storage(if not already provided)
        ) as agent,
    ):
        # Start a new thread for the agent conversation.
        thread = agent.get_new_thread()

        query = "Who is Tom Cruise?"
        print(f"[bold]Test 1: {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query, thread=thread)
        print(f"[green]{result.text}\n[/green]")

        query = "And where is he from?"
        print(f"[bold]Test 2: {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query, thread=thread)
        print(f"[green]{result.text}\n[/green]")

        query = "And where is he from? (asked w/o the conversation thread)"
        print(f"[bold]Test 3 {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query)
        print(f"[green]{result.text}\n[/green]")

        serialized_thread = await thread.serialize()

        resumed_thread = await agent.deserialize_thread(serialized_thread)

        # You can now use the resumed_thread to continue the conversation.
        query = "What is his most famous movie?"
        print(f"[bold]Test 4: {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query, thread=resumed_thread)
        print(f"[green]{result.text}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
