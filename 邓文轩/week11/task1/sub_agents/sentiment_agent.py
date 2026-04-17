import os
from agents import Agent, ModelSettings, AsyncOpenAI, OpenAIChatCompletionsModel

external_client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SENTIMENT_PROMPT = (
    "你是一个情感分类专家。你的任务是对用户输入的文本进行情感分类。"
    "只输出分类结果：positive、negative 或 neutral。"
)

sentiment_agent = Agent(
    name="sentiment_agent",
    instructions=SENTIMENT_PROMPT,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
    model_settings=ModelSettings(parallel_tool_calls=False)
)
