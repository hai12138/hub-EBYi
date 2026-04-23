"""
作业1: 安装openai-agents框架，实现如下的一个程序：
有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
子agent 1: 对文本进行情感分类
子agent 2: 对文本进行实体识别
"""

import os

os.environ["OPENAI_API_KEY"] = "sk-e0e141c7108a4cabaf4cced1f749ae89"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from fastmcp import FastMCP
import asyncio
from enum import Enum
from agents import Agent, Runner, GuardrailFunctionOutput
from agents import set_default_openai_api, set_tracing_disabled
from pydantic import BaseModel
from openai.types.responses import ResponseTextDeltaEvent

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class EmotionLabel(Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class EmotionClassificationOutput(BaseModel):
    """用于判断用户请求是否属于功课或学习类问题的结构"""
    emotion: EmotionLabel


emotion_agent = Agent(
    name="emotion_agent",
    model="qwen-max",
    instructions=(
        "You are a sentiment analysis agent. "
        "Analyze the overall sentiment of the given text, considering all parts of the text. "
        "Possible values for 'emotion' are 'positive', 'negative', or 'neutral'. "
        "Return the result in JSON format, with key 'emotion' and value as one of the three options. "
        "Do not use Chinese in the JSON keys or values. "
        "Examples:\n"
        "Input: '我今天早上弄丢了钱包，但是下午有好心人给我送回来了'\n"
        "Output: {\"emotion\": \"positive\"}\n"
        "Input: '我今天很难过，因为钱包丢了'\n"
        "Output: {\"emotion\": \"negative\"}\n"
        "Input: '我今天去上班了'\n"
        "Output: {\"emotion\": \"neutral\"}\n"
    ),
    output_type=EmotionClassificationOutput,
)


class EntityItem(BaseModel):
    entity: str
    type: str

class NEROutput(BaseModel):
    entities: list[EntityItem]

NER_agent = Agent(
    name="NER_agent",
    model="qwen-max",
    instructions=(
        "You are a NER agent. You analyze the entities in the given text. "
        "Return the result in JSON format, as an object with a key 'entities', whose value is a list of objects, each with keys 'entity' and 'type', and all values in English. "
        "Do not use Chinese in the JSON keys or values. Example: {\"entities\": [{\"entity\": \"Beijing\", \"type\": \"Location\"}]}"
    ),
    output_type=NEROutput,
)

triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Your task is to select an agent to handle the request based on the language of the request.",
    handoffs=[emotion_agent, NER_agent],
)


async def main():
    msg = input("请输入文本，我将分析其中的情感和实体信息：")

    result = Runner.run_streamed(
        triage_agent,
        msg,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print(f"\n分析结果: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
