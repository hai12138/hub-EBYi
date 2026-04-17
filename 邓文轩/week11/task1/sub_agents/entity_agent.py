import os
from agents import Agent, ModelSettings, AsyncOpenAI, OpenAIChatCompletionsModel

external_client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

ENTITY_PROMPT = (
    "你是一个实体识别专家。你的任务是从用户输入的文本中识别出所有命名实体（如人名、地名、机构名等）。"
    "以列表形式输出识别到的实体及其类型。"
)

entity_agent = Agent(
    name="entity_agent",
    instructions=ENTITY_PROMPT,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
    model_settings=ModelSettings(parallel_tool_calls=False)
)
