If you come from other frameworks you will probably be used to use something called a client and the build on top of it.

In MSF everything is an Agent, and we have different kinds.

-Azure AI Froundry agents
They map 1:1 to Ai Foundry agents, the can be persisted and support service manage conversation threads.
you need env var
and this package module ``pip install `agent-framework[azure-ai]``

The `01-basic_ai_fondry_agent.py` show a basic example of creating an agent that uses a model deplayed in AI Foundry.

You can also create the AzureAIAgentClient details this way:  
run
```python
AzureAIAgentClient(
            project_endpoint="https://<your-project>.services.ai.azure.com/api/projects/<project-id>",
            model_deployment_name="gpt-4o-mini",
            async_credential=credential,
            agent_name="HelperAgent"
        )
```
Note that the method `create_agent` returns a ChatClient

The `02-existing_ai_foundry.py` shows how to use AIProjectClient to create a new AI Foundry agent, use it, and delete it.
In case you want to use an existing AI Foundry Agent you can get by its Id using:

``python
AzureAIAgentClient(
            project_endpoint="https://<your-project>.services.ai.azure.com/api/projects/<project-id>",
            model_deployment_name="gpt-4o-mini",
            async_credential=credential,
            agent_id="<existing-agent-id>"
        )
```