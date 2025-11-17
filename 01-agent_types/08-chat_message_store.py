import asyncio
from collections.abc import Collection
from typing import Any

from agent_framework import ChatMessage, ChatMessageStoreProtocol
from agent_framework._threads import ChatMessageStoreState
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from core.agent import get_client
from dotenv import load_dotenv
from rich import print

load_dotenv()


class CustomChatMessageStore(ChatMessageStoreProtocol):

    def __init__(self, messages: Collection[ChatMessage] | None = None) -> None:
        self._messages: list[ChatMessage] = []
        if messages:
            self._messages.extend(messages)

    async def add_messages(self, messages: Collection[ChatMessage]) -> None:
        print(
            f"[magenta]Adding messages to CustomChatMessageStore: {len(messages)}[/magenta]"
        )
        self._messages.extend(messages)

    async def list_messages(self) -> list[ChatMessage]:
        print(
            f"[magenta]Listing messages from CustomChatMessageStore: {len(self._messages)}[/magenta]"
        )
        return self._messages

    @classmethod
    async def deserialize(
        cls, serialized_store_state: Any, **kwargs: Any
    ) -> "CustomChatMessageStore":
        """Create a new instance from serialized state."""
        print(f"[red]Deserialize messages from CustomChatMessageStore[/red]")
        store = cls()
        await store.update_from_state(serialized_store_state, **kwargs)
        return store

    async def update_from_state(
        self, serialized_store_state: Any, **kwargs: Any
    ) -> None:
        """Update this instance from serialized state."""
        print(f"[red]Deserialize messages from CustomChatMessageStore[/red]")
        if serialized_store_state:
            state = ChatMessageStoreState.from_dict(serialized_store_state, **kwargs)
            if state.messages:
                self._messages.extend(state.messages)

    async def serialize(self, **kwargs: Any) -> Any:
        """Serialize this store's state."""
        print(f"[magenta]Serialize messages from CustomChatMessageStore[/magenta]")
        state = ChatMessageStoreState(messages=self._messages)
        return state.to_dict(**kwargs)


async def main():
    async with (
        AzureCliCredential() as credential,
        get_client().create_agent(
            name="HelperAgent",
            instructions="You are a helpful assistant.",
            chat_message_store_factory=CustomChatMessageStore,
            store=None,
        ) as agent,
    ):
        # Start a new thread for the agent conversation.

        thread = agent.get_new_thread()

        query = "Who is Tom Cruise?"
        print(f"[bold]Test 1: {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query, thread=thread)
        print(f"[green]{result.text}\n[/green]")

        serialized_thread = await thread.serialize()
        resumed_thread = await agent.deserialize_thread(serialized_thread)

        # You can now use the resumed_thread to continue the conversation.
        query = "and where is he from?"
        print(f"[bold]Test 2: {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query, thread=resumed_thread)
        print(f"[green]{result.text}\n[/green]")


if __name__ == "__main__":
    asyncio.run(main())
