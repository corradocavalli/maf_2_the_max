import asyncio
from typing import Annotated, Any

from agent_framework import AgentRunResponse, ChatMessage, Role, ai_function
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from pydantic import Field
from rich import print

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


@ai_function(
    name="get_loan",
    description="Get a loan for a given user.",
    approval_mode="always_require",
)
def _get_loan(
    amount: Annotated[float, Field(description="The amount to request.")],
) -> str:
    global balance
    balance += amount
    return f"Processing loan request of ${amount}."


async def main():
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential).create_agent(
            instructions="""
            You are a helpful assistant, your task is to help user making orders online.
            You allow users to check their balance and make payments.
            If the balance is not sufficient, you should inform the user and not proceed with the payment.
            """,
            name="PaymentAgent",
            tools=[get_balance, _make_payment],
        ) as agent,
    ):
        query = "Hello, I would like to check my balance."
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")
        result = await agent.run(query)
        print(f"[green]{result.text}[/green]")

        query = "Make a payment of $45 for my internet subscription."
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")
        result = await agent.run(query)
        print(f"[yellow]{result.text}[/yellow]")

        query = "What's my new balance?"
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")
        result = await agent.run(query)
        print(f"[green]{result.text}[/green]")

        # Attempt a payment that exceeds the balance
        query = "I want to make a payment of $150 for my monthly Gym subscription."
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")
        result = await agent.run(query)
        print(f"[yellow]{result.text}[/yellow]")

        # Get a loan to cover the payment (requires approval)
        query = "I want to request a loan of $150 to cover my gym subscription payment."
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")

        result = await agent.run(
            query, tools=[_get_loan]
        )  # adding the loan tool for this request only

        while len(result.user_input_requests) > 0:
            # Start with the original query
            new_inputs: list[Any] = [query]

            for user_input_needed in result.user_input_requests:
                print(
                    f"\n[bold][cyan]User Input Request for function from {agent.name}:"
                    f"\n Function: {user_input_needed.function_call.name}"
                    f"\n Arguments: {user_input_needed.function_call.arguments}[/cyan][/bold]"
                )

                # Add the assistant message with the approval request
                new_inputs.append(
                    ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed])
                )

                # Get user approval
                user_approval = await asyncio.to_thread(
                    input, "\nApprove function call? (y/n): "
                )

                # Add the user's approval response
                new_inputs.append(
                    ChatMessage(
                        role=Role.USER,
                        contents=[
                            user_input_needed.create_response(
                                user_approval.lower() == "y"
                            )
                        ],
                    )
                )

            # Run again with all the context
            print("\n[white]Re-running with user approvals...[white]")
            result = await agent.run(new_inputs, tools=[_get_loan])
            print(f"[yellow]{result.text}[/yellow]")
            print(
                f"DEBUG: Remaining user_input_requests: {len(result.user_input_requests)}\n"
            )

        query = "What's my balance after the loan?"
        print(f"\n[bold][yellow]User:[/bold][/yellow] {query}")


if __name__ == "__main__":
    asyncio.run(main())
