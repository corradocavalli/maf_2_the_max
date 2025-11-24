"""
Simple Sequential Workflow - No AI Agents Involved
===================================================

This script demonstrates a basic sequential workflow using the Agent Framework's
workflow capabilities WITHOUT any AI agents. It shows data transformation through
a pipeline of executors.

Executor Definition Methods:
----------------------------
1. Class-based Executor (UpperCase):
   - Inherit from Executor base class
   - Define handlers using @handler decorator
   - Useful for stateful executors or complex logic
   - Example: class UpperCase(Executor)

2. Function-based Executor (reverse_text):
   - Use @executor decorator on async functions
   - Simpler for stateless operations
   - More concise for single-purpose transformations
   - Example: @executor(id="reverse_text_executor")

3. Lambda/Inline Executor (not shown here):
   - Can be defined inline for simple operations
   - Useful for quick transformations in the workflow

The workflow follows a simple pattern:
Input → UpperCase → ReverseText → Output
"""

import asyncio

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
)
from rich import print
from typing_extensions import Never


# Class-based executor with explicit inheritance
class UpperCase(Executor):
    def __init__(self, id: str):
        super().__init__(id=id)

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        """Convert the input to uppercase and forward it to the next node.

        Note: The WorkflowContext is parameterized with the type this handler will
        emit. Here WorkflowContext[str] means downstream nodes should expect str.
        """
        result = text.upper()

        # Send the result to the next executor in the workflow.
        await ctx.send_message(result)


# Example 2: A standalone function-based executor
@executor(id="reverse_text_executor")
async def reverse_text(text: str, ctx: WorkflowContext[Never, str]) -> None:
    """Reverse the input string and yield the workflow output.

    This node yields the final output using ctx.yield_output(result).
    The workflow will complete when it becomes idle (no more work to do).

    The WorkflowContext is parameterized with two types:
    - T_Out = Never: this node does not send messages to downstream nodes.
    - T_W_Out = str: this node yields workflow output of type str.
    """
    result = text[::-1]

    # Yield the output - the workflow will complete when idle
    await ctx.yield_output(result)


async def main():
    upper_case = UpperCase(id="upper_case_executor")
    workflow = (
        WorkflowBuilder()
        .add_edge(upper_case, reverse_text)
        .set_start_executor(upper_case)
        .build()
    )
    events = await workflow.run("hello world")
    # workflow emits events during its execution, we are interested in the final state only
    print(f"\n[cyan]Events:[/cyan]")
    for event in events:
        print(f"[yellow]Event: {event}[/yellow]")

    print(f"\n[green]Outputs: {events.get_outputs()}[/green]")
    # Summarize the final run state (e.g., IDLE)
    print(f"[blue]Final state: {events.get_final_state().value}[/blue]")


if __name__ == "__main__":
    asyncio.run(main())
