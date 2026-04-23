'''
安装openai-agents框架，实现如下的一个程序：
• 有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
• 子agent 1: 对文本进行情感分类
• 子agent 2: 对文本进行实体识别
'''
import os

os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


sentiment_classification_agent = Agent(
    name="sentiment_classification_agent",
    model="qwen-max",
    instructions="你是情感专家，擅长对情感进行分类，回答问题时先告诉我你是谁。"
)

entity_recognition_agent = Agent(
    name="entity_recognition_agent",
    model="qwen-max",
    instructions="你是实体识别专家，擅长识别文本中人名、地名、机构名，回答问题时先告诉我你是谁。"
)

triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions=
    '''
    你是任务分发器。
    - 如果用户想要分析情感（如“这句话表达了什么情绪”），移交给 sentiment_classification_agent。
    - 如果用户想要识别实体（如“这句话里有哪些人名”），移交给 entity_recognition_agent。
    - 不要直接回答用户的问题，你的任务只是分发。"
    ''',
    handoffs=[sentiment_classification_agent, entity_recognition_agent]
)

triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[sentiment_classification_agent, entity_recognition_agent],
)

async def main():
    conversation_id = str(uuid.uuid4().hex[:16])
    msg = input("你好，我可以帮你分析情感和识别实体，你有什么问题？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    while True:
        with trace("Routing example", group_id=conversation_id):
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

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent

if __name__ == "__main__":
    asyncio.run(main())