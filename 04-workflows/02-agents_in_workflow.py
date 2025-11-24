# Copyright (c) Microsoft. All rights reserved.

"""
AI Agents in Sequential Workflow
==================================

This example demonstrates how AI agents can be easily composed into a workflow.
Unlike the basic workflow example (01-basic.py) which uses simple data transformations,
this script shows how to integrate AI agents as executors in a workflow pipeline.

Key Features:
-------------
- AI agents as workflow executors
- Sequential agent processing (Writer → Reviewer)
- Automatic message passing between agents
- Clean resource management with async context managers

Workflow Pattern:
-----------------
User Input → Writer Agent → Reviewer Agent → Output

The Writer agent creates content based on the prompt, and the Reviewer agent
analyzes and scores the output, demonstrating how agents can work together
in a coordinated workflow.
"""

import asyncio

from agent_framework import AgentRunEvent, WorkflowBuilder
from agent_framework_azure_ai import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv
from rich import print

load_dotenv()


async def main():
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential) as chat_client,
    ):
        async with (
            chat_client.create_agent(
                instructions=(
                    "You are good at creating catchy short emails."
                    "Email content can vary from very sad to very happy."
                ),
                name="writer",
            ) as writer_agent,
            chat_client.create_agent(
                instructions=(
                    "You are an excellent email reviewer."
                    "You analyze the email content and tone, and return a value from 1 to 10 indicating the happiness of the email."
                ),
                name="sentiment_reviewer",
            ) as reviewer_agent,
        ):
            # Build the workflow using the fluent builder.
            workflow = (
                WorkflowBuilder()
                .set_start_executor(writer_agent)
                .add_edge(writer_agent, reviewer_agent)
                .build()
            )

            # Run the workflow with the user's initial message.
            events = await workflow.run("Create a nice email for my friend")
            # Print agent run events
            for event in events:
                if isinstance(event, AgentRunEvent):
                    print(f"{event.executor_id}: {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
