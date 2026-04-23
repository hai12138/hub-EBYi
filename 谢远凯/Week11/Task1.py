'''1: 安装openai-agents框架，实现如下的一个程序：
• 有一个主agent，接受用户的请求输入，选择其中的一个agent 回答
• 子agent 1: 对文本进行情感分类
• 子agent 2: 对文本进行实体识别'''
import asyncio

from agents import Agent, Runner, RawResponsesStreamEvent, TResponseInputItem
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent

from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

classification_agent = Agent(
    name='sentiment classification agent',
    instructions='擅长对文本进行情感分类',
    handoff_description='对文本进行情感分类',)
ner_agent = Agent(
    name='Named Entity Recognition agent',
    instructions='擅长对文本进行实体识别',
    handoff_description='对文本进行实体识别',)

triage_age = Agent(
    name='triage agent',
    instructions='Handoff to the appropriate agent based on input text',
    handoffs=[classification_agent, ner_agent]
)
async def main():
    msg = input("你好，我可以帮你进行情感分类或者实体识别，请输入你要进行任务和文本？")
    # inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    result = Runner.run_streamed(
        classification_agent,
        input=msg,
    )
    async for event in result.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        data = event.data
        if isinstance(data, ResponseTextDeltaEvent):
            print(data.delta, end="", flush=True)
        elif isinstance(data, ResponseContentPartDoneEvent):
            print("\n")
if __name__ == "__main__":
    asyncio.run(main())
