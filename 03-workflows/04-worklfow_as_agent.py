"""
Workflow as Agent - Concurrent Workflow Pattern
================================================

This script demonstrates how a concurrent workflow can be converted into a reusable
agent using the `workflow.as_agent()` method. This powerful pattern allows complex
multi-agent workflows to be encapsulated and used like a single agent.

Key Concepts:
-------------
1. **ConcurrentBuilder**: Creates workflows where multiple agents run in parallel
2. **workflow.as_agent()**: Converts a workflow into an agent interface
3. **Agent Reusability**: The workflow-as-agent can be used in other workflows or called directly

Workflow Pattern:
-----------------
                    ┌─────────────────┐
                    │   User Prompt   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
      ┌─────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
      │ Researcher  │  │  Marketer  │  │   Legal    │
      │   Agent     │  │   Agent    │  │   Agent    │
      └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Aggregated     │
                    │  Response       │
                    └─────────────────┘

Agents in this Example:
-----------------------
- **Researcher**: Provides market insights, opportunities, and risks
- **Marketer**: Creates value propositions and messaging strategies
- **Legal**: Reviews compliance, constraints, and policy concerns

All three agents process the same input concurrently, and their responses
are aggregated into a single output.

Use Cases:
----------
- Multi-perspective analysis (research, marketing, legal review)
- Consensus building with multiple expert agents
- Parallel processing for faster response times
- Modular workflow composition
"""

import asyncio

from agent_framework import ConcurrentBuilder
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework_azure_ai import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:

    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential) as chat_client,
    ):

        researcher = chat_client.create_agent(
            instructions=(
                "You're an expert market and product researcher. Given a prompt, provide concise, factual insights,"
                " opportunities, and risks."
            ),
            name="researcher",
        )

        marketer = chat_client.create_agent(
            instructions=(
                "You're a creative marketing strategist. Craft compelling value propositions and target messaging"
                " aligned to the prompt."
            ),
            name="marketer",
        )

        legal = chat_client.create_agent(
            instructions=(
                "You're a cautious legal/compliance reviewer. Highlight constraints, disclaimers, and policy concerns"
                " based on the prompt."
            ),
            name="legal",
        )

        workflow = (
            ConcurrentBuilder().participants([researcher, marketer, legal]).build()
        )

        # Expose the concurrent workflow as an agent
        agent = workflow.as_agent(name="ConcurrentWorkflowAgent")
        prompt = (
            "We are launching a new budget-friendly electric bike for urban commuters."
        )
        agent_response = await agent.run(prompt)

        if agent_response.messages:
            print("\n===== Aggregated Messages =====")
            for i, msg in enumerate(agent_response.messages, start=1):
                role = getattr(msg.role, "value", msg.role)
                name = msg.author_name if msg.author_name else role
                print(f"{'-' * 60}\n\n{i:02d} [{name}]:\n{msg.text}")


if __name__ == "__main__":
    asyncio.run(main())
