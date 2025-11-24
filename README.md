# Microsoft Agent Framework Introduction Demo

## Prerequisites
1. You need Azure CLI `az login`
2. You need a Microsoft Foundry project and a model deployed.
3. You need [uv](https://github.com/astral-sh/uv) package manager installed
   
## Setup
- use `uv sync --prerelease=allow` to create the Python virtual environment with all depencencies installed.
- Authenticate using `az login` command

## Repository Overview

### 01-basics
- `01-basic_ai_foundry_agent.py` - Simplest way to invoke an AI agent deployed to Azure AI Foundry
- `02-streaming_agent.py` - Streaming agent responses incrementally for responsive user experience
- `03-create_ai_foundry_agent.py` - Integration with Azure AI Foundry SDK for persistent agent management

### 02-tools_and_mcp
- `01-local_tools.py` - Create agents that use local Python functions as tools
- `02-mcp_tools.py` - Interact with remote MCP tools via StreamableHTTP and combine multiple MCP servers
- `03-agent_mcp_client.py` - Connect to an MCP server that wraps an agent via StreamableHTTP transport
  ### mcp-servers
- `server.py` - Basic MCP server exposing payment tools (get_balance, make_payment)
- `agent_mcp_server.py` - Turn an Agent Framework agent into an MCP server accessible remotely

### 03-extras
- `01-threads.py` - Maintain conversation context and serialize/deserialize threads for persistence
- `02-chat_message_store.py` - Implement custom message stores to persist conversations in any storage backend
- `03-context_provider.py` - Create custom ContextProviders to modify inference context before/after LLM calls
- `04-middleware.py` - Use middleware to intercept agent operations, chat requests, and function calls

### 04-workflows
- `01-basic.py` - Basic sequential workflow with data transformation pipeline (no AI agents)
- `02-agents_in_workflow.py` - Compose AI agents as executors in a sequential workflow pipeline
- `03-control_flow.py` - Conditional workflow with spam detection and dynamic routing based on agent responses
- `04-worklfow_as_agent.py` - Convert a concurrent multi-agent workflow into a reusable single agent

### 05-dev_ui
- `01-demo.py` - Visualize and debug Agent Framework workflows with the Dev-UI tool
