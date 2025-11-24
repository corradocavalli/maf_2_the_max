"""
Agent Framework with Azure AI Foundry SDK Features
===================================================

This script demonstrates how Microsoft Agent Framework seamlessly integrates with
Azure AI Foundry SDK features, giving you access to the full capabilities of both
frameworks. It shows how to use native Foundry SDK classes alongside Agent Framework.

Key Concepts:
-------------
1. **AIProjectClient**: Direct access to Azure AI Foundry project resources
2. **AgentsClient**: Low-level agent creation and management from Foundry SDK
3. **Persistent Agents**: Creating agents that persist in Azure AI Foundry
4. **ChatAgent Wrapper**: Using Agent Framework's ChatAgent with Foundry agents
5. **Resource Management**: Proper cleanup of created agents

"""

import asyncio
import os

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.ai.agents.aio import AgentsClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
from rich import print


async def main():
    async with (
        AzureCliCredential() as credential,
        AIProjectClient(
            endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential
        ) as project_client,
        AgentsClient(
            endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential
        ) as agents_client,
    ):
        print("Listing deployments in the project:")
        async for deployment in project_client.deployments.list():
            print(f"[blue]- {deployment.name}[/blue]")

        # Create a persistent agent
        created_agent = await agents_client.create_agent(
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            name="PersistentAgent",
            instructions="You are a helpful assistant. You answer questions in French.",
        )
        print(f"Created agent with ID: [magenta]{created_agent.id}[/magenta]")

        try:
            # Use the agent
            async with ChatAgent(
                chat_client=AzureAIAgentClient(
                    agents_client=agents_client, agent_id=created_agent.id
                )
            ) as agent:
                result = await agent.run("What is the capital of France?")
                print(f"[green]{result.text}[/green]")
        finally:
            # Clean up the agent
            await agents_client.delete_agent(created_agent.id)


if __name__ == "__main__":
    asyncio.run(main())
