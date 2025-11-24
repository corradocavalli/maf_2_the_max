"""
Custom Context Provider for Dynamic Context Management
=======================================================

This script demonstrates how to create a custom ContextProvider that modifies
the inference context before it's passed to the LLM and processes the response
after the LLM completes the request.

Key Concepts:
-------------
- **ContextProvider**: Interface for pre/post-processing LLM interactions
- **invoking()**: Modify context BEFORE sending to the LLM
- **invoked()**: Process response AFTER receiving from the LLM
- **Dynamic Instructions**: Inject context-aware instructions based on state
- **Information Extraction**: Use LLM to extract and cache structured data

How It Works:
-------------
1. Before LLM Call (invoking):
   - Check if user's name and age are known
   - If missing, inject instructions to request this information
   - Add context instructions that guide the LLM's behavior

2. After LLM Call (invoked):
   - Analyze user messages for name and age information
   - Use structured output (Pydantic model) to extract data
   - Cache extracted information for future interactions
   - Update internal state to avoid re-asking

Example Flow:
-------------
User: "What is the square root of 9?"
→ invoking: "Ask for name and age, decline to answer until provided"
Agent: "Before I help, may I know your name and age?"

User: "My name is Bob"
→ invoked: Extract name="Bob" from message, cache it
→ invoking: "User's name is Bob. Still need age."
Agent: "Thanks Bob! May I also know your age?"

User: "I am 27 years old"
→ invoked: Extract age=27 from message, cache it
→ invoking: "User's name is Bob, age is 27" (no more requests)
Agent: "Got it, Bob! How can I help you today?"

ContextProvider Methods:
------------------------
- **invoking(messages, **kwargs)**: Pre-process before LLM call
  - Modify system instructions
  - Add contextual information
  - Enforce requirements or policies
  - Returns: Context object with additional instructions

- **invoked(request, response, exception, **kwargs)**: Post-process after LLM call
  - Extract information from user messages
  - Update internal state/memory
  - Log interactions
  - Trigger side effects or workflows

- **serialize()**: Persist provider state for thread serialization
  - Save cached data for session recovery
  - Enable cross-session memory
U
"""

import asyncio
from collections.abc import MutableSequence, Sequence
from typing import Any

from agent_framework import (
    ChatClientProtocol,
    ChatMessage,
    ChatOptions,
    Context,
    ContextProvider,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print

load_dotenv()


# Define a Pydantic model to hold user information structure (for model structured output)
class UserInfo(BaseModel):
    name: str | None = None
    age: int | None = None


class UserInfoMemory(ContextProvider):
    def __init__(
        self,
        chat_client: ChatClientProtocol,
        user_info: UserInfo | None = None,
        **kwargs: Any,
    ):
        """Create the memory.

        If you pass in kwargs, they will be attempted to be used to create a UserInfo object.
        """

        self._chat_client = chat_client
        if user_info:
            self.user_info = user_info
        elif kwargs:
            self.user_info = UserInfo.model_validate(kwargs)
        else:
            self.user_info = UserInfo()

    async def invoking(
        self, messages: ChatMessage | MutableSequence[ChatMessage], **kwargs: Any
    ) -> Context:
        """Provide user information context before each agent call."""

        print(f"[cyan]->Invoking>[/cyan]")
        for msg in messages:
            print(
                f"[blue]Message Role:[/blue] {msg.role}, [blue]Content:[/blue] {msg.text}"
            )

        instructions: list[str] = []

        if self.user_info.name is None:
            instructions.append(
                "Ask the user for their name and politely decline to answer any questions until they provide it."
            )
        else:
            instructions.append(f"The user's name is {self.user_info.name}.")

        if self.user_info.age is None:
            instructions.append(
                "Ask the user for their age and politely decline to answer any questions until they provide it."
            )
        else:
            instructions.append(f"The user's age is {self.user_info.age}.")

        # Return context with additional instructions
        return Context(instructions=" ".join(instructions))

    def serialize(self) -> str:
        """Serialize the user info for thread persistence."""
        return self.user_info.model_dump_json()

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract user information from messages after each agent call."""
        print(f"[cyan]->Invoked[/cyan]")
        # Check if we need to extract user info from user messages
        user_messages = [
            msg
            for msg in request_messages
            if hasattr(msg, "role") and msg.role.value == "user"
        ]
        if (
            self.user_info.name is None or self.user_info.age is None
        ) and user_messages:
            try:
                # Use the chat client to extract structured information
                result = await self._chat_client.get_response(
                    messages=request_messages,  # type: ignore
                    chat_options=ChatOptions(
                        instructions="Extract the user's name and age from the message if present. If not present return nulls.",
                        response_format=UserInfo,  # We use Structured Output to extract the info
                    ),
                )

                # Update user info with extracted data
                if result.value and isinstance(result.value, UserInfo):
                    if self.user_info.name is None and result.value.name:
                        self.user_info.name = result.value.name
                    if self.user_info.age is None and result.value.age:
                        self.user_info.age = result.value.age

            except Exception:
                pass  # Failed to extract, continue without updating


async def main():
    async with AzureCliCredential() as credential:
        chat_client = AzureAIAgentClient(async_credential=credential)

        # Create the context provider
        memory_provider = UserInfoMemory(chat_client)

        async with chat_client.create_agent(
            instructions="you are a friendly assistant. Always address the user by their name.",
            context_providers=[memory_provider],
        ) as agent:

            # Start a new thread for the agent conversation.
            thread = agent.get_new_thread()

            query = "Hello, what is the square root of 9?"
            print(f"[bold]1: {query}[/bold]")
            print("-" * 50)
            result = await agent.run(query, thread=thread)
            print(f"[green]{result.text}\n[/green]")

            query = "My Name is Bob"
            print(f"[bold]2: {query}[/bold]")
            print("-" * 50)
            result = await agent.run(query, thread=thread)
            print(f"[green]{result.text}\n[/green]")

            query = "And where I am 27 years old"
            print(f"[bold]3: {query}[/bold]")
            print("-" * 50)
            result = await agent.run(query)
            print(f"[green]{result.text}\n[/green]")

            # Access the memory component via the thread's get_service method and inspect the memories
            user_info_memory = thread.context_provider.providers[0]  # type: ignore
            if user_info_memory:
                print()
                print(f"MEMORY - User Name: {user_info_memory.user_info.name}")  # type: ignore
                print(f"MEMORY - User Age: {user_info_memory.user_info.age}")  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
