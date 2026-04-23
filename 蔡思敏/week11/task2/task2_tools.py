import re
from typing import Annotated, Union
import requests
import os
import openai
TOKEN = "6d997a997fbf"

os.environ["OPENAI_API_KEY"] = "sk-e365d480f719416e8f4e317b7fa03ca1"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = openai.Client(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
def named_entity_recognition(text: Annotated[str, "Text to perform NER on"]):
    """Performs Named Entity Recognition using external NLP API."""
    response = client.completions.create(model="qwen-max", prompt=text)
    return response.outputs[0].text

@mcp.tool
def get_animal_info(animal_name: Annotated[str, "Name of the animal, e.g., 'panda', '大熊猫', 'lion'"]):
    """Retrieves information about a given animal."""
    try:
        return requests.get(
            f"https://zh.wikipedia.org/api/rest_v1/page/summary/{animal_name}").json()
    except:
        return []

@mcp.tool
def get_planet_info(planet_name: Annotated[str, "Name of the planet, e.g., 'earth', 'venus', 'mars'"]):
    """Retrieves information about a given planet."""
    try:
        return requests.get(
            f"https://api.le-systeme-solaire.net/rest/bodies/{planet_name.lower()}").json()
    except:
        return []
