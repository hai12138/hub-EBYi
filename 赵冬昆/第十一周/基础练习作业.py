import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-0c18954972f8450c95480d5c3c0a82f6"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled
from agents import Agent, TResponseInputItem

from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.stream_events import RawResponsesStreamEvent

# 情感 代理
emotion_tutor_agent = Agent(
    name="Emotion Tutor",
    model="qwen-max",
    handoff_description="负责处理所有情感问题的专家代理。",
    instructions="您是专业的情感分析专家。请协助回答问题，并且清晰的分析用户提出问题的情感状态以及变化。",
)
entity_recognition_agent = Agent(
    name="Entity Recognition",
    model="qwen-max",
    handoff_description="负责处理实体识别的专家代理。",
    instructions="您是一个专业的实体识别专家。请协助回答问题，并且能够清晰的从一段非结构化的文本中，自动找出具有特定意义的“实体”（Entity），并将其归类到预定义的类别中。"
)

# triage 定义的的名字 默认的功能用户提问 指派其他agent进行完成
triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[emotion_tutor_agent, entity_recognition_agent],
)


async def main():
    # print(f"\n 接受用户的输入:'{input_data}'")
    conversation_id = str(uuid.uuid4().hex[:16])
    msg = input("你好，我可以帮你识别情感以及提取关键信息问题，你有什么问题？")
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


# result = await Runner.run(emotion_tutor_agent, input_data,context=ctx.context)


if __name__ == "__main__":
    asyncio.run(main())
