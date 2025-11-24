import asyncio
import time
from collections.abc import Awaitable, Callable
from random import randint
from typing import Annotated

from agent_framework import (
    AgentRunContext,
    ChatContext,
    ChatMessage,
    ChatResponse,
    FunctionInvocationContext,
    FunctionMiddleware,
    Role,
    chat_middleware,
    function_middleware,
)
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from pydantic import Field
from rich import print

load_dotenv()


# The weather function to be used by the agent
async def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    print(f"[blue][get_weather] Getting weather for {location}[/blue]")
    await asyncio.sleep(0.5)  # Simulate an async API call
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


# Chat middleware intercepts chat requests sent to AI models
@chat_middleware
async def security_middleware(
    context: ChatContext,
    next: Callable[[ChatContext], Awaitable[None]],
) -> None:
    """Process chat messages to filter sensitive information."""
    print("[cyan][SecurityMiddleware] Processing input...[/cyan]")

    # Security check - block messages containing certain bitcoin
    blocked_term = "bitcoin"

    for message in context.messages:
        if message.text:
            message_lower = message.text.lower()
            if blocked_term in message_lower:
                print(
                    f"[red][SecurityMiddleware] BLOCKED: Found '{blocked_term}' in message: '{message.text}'[/red]"
                )
                # Override the response instead of calling AI
                context.result = ChatResponse(
                    messages=[
                        ChatMessage(
                            role=Role.ASSISTANT,
                            text="Sorry, I cannot assist with that request.",
                        )
                    ]
                )

                # Set terminate flag to stop execution
                context.terminate = True
                return

    # Continue to next middleware or AI execution
    await next(context)


# This time we create a FunctionMiddleware class
class LoggingFunctionMiddleware(FunctionMiddleware):
    """Function middleware that logs function execution."""

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        # Pre-processing: Log before function execution
        print(f"[yellow]Calling function: {context.function.name}[/yellow]")
        start_time = time.time()
        await next(context)
        duration = time.time() - start_time
        print(f"[yellow]Function call completed in {duration:.2f} seconds[/yellow]")


async def agent_middleware(
    context: AgentRunContext,
    next: Callable[[AgentRunContext], Awaitable[None]],
) -> None:
    """Agent middleware that logs run invocations."""

    print(f"[magenta][AgentMiddleware] Started[/magenta]")

    await next(context)
    print(f"[magenta][AgentMiddleware] Completed[/magenta]")


async def main():
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            name="HelperAgent",
            instructions="You are a helpful assistant.",
            tools=[get_weather],
            middleware=[
                agent_middleware,
                security_middleware,
                LoggingFunctionMiddleware(),
            ],
        ) as agent,
    ):
        query = "I want to know about bitcoin."
        print(f"[bold]1: {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query)
        print(f"[green]{result.text}\n[/green]")

        query = "What is the weather in Zurich"
        print(f"[bold]2: {query}[/bold]")
        print("-" * 50)
        result = await agent.run(query)
        print(f"[green]{result.text}\n[/green]")


if __name__ == "__main__":
    asyncio.run(main())
