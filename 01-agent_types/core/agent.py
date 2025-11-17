import os

from agent_framework.azure import AzureOpenAIChatClient
from dotenv import load_dotenv

load_dotenv()


def get_client():
    return AzureOpenAIChatClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
