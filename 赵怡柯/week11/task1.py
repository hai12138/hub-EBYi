'''
作业1: 安装openai-agents框架，实现如下的一个程序：
• 有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
• 子agent 1: 对文本进行情感分类
• 子agent 2: 对文本进行实体识别
'''
import os

os.environ["OPENAI_API_KEY"] = "sk-54703c491c0c42bb9dddbc6db21b78da"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent, ResponseAudioDoneEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace

from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

tc_agent = Agent(
    name="TextClassification_agent",
    model="qwen-max",
    instructions="你是小文，一个文本情感分类专家，回答问题的时候先告诉我你是谁。"
)

ner_agent = Agent(
    name="NER_agent",
    model="qwen-max",
    instructions="你是小李，一个文本实体识别专家，回答问题的时候先告诉我你是谁。"
)

triage_agent = Agent(
    name="Triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[tc_agent, ner_agent],
)

async def run_agent():
    ass_prompt = input("你好，我可以帮你进行文本的情感分类或者文本的实体识别，你有什么问题吗？\n")
    agent = triage_agent
    inputs:list[TResponseInputItem]=[{"content":ass_prompt, "role":"user"}]


    while True:
        result = Runner.run_streamed(
            agent,
            input=inputs,
        )
        async for event in result.stream_events():
            if not isinstance(event, RawResponsesStreamEvent):
                continue
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                print(data.delta, end="", flush=True)
            elif isinstance(data, ResponseContentPartDoneEvent):
                print("\n")
            # elif isinstance(data, ResponseDoneEvent):
            #     print("对话结束。")

        inputs = result.to_input_list() # 每轮都保存完整对话历史
        print()
        user_query = input("请输入你的问题：\n")
        inputs.append({"content":user_query, "role":"user"})
        # 切换到被转交的agent
        agent = result.current_agent

if __name__ == '__main__':
    asyncio.run(run_agent())