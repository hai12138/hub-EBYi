from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
import os
from dotenv import load_dotenv
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel,Field
set_tracing_disabled(True)

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
url = os.getenv("OPENAI_API_URL")
external_client = AsyncOpenAI(
    api_key=key,
    base_url=url,
)

class Entity(BaseModel):
    entity: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")
    start: int = Field(..., description="起始位置")
    end: int = Field(..., description="结束位置")
class EntityList(BaseModel):
    entities: list[Entity] = Field(..., description="实体列表")

agent_sentiment_classification = Agent(
    name="SentimentClassification",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client
    ),
    instructions="""
            你是一个智能情感分类器，负责判断输入的文本的情感。
        """,
    output_type=str,
)

agent_entity_recognition = Agent(
    name="EntityRecognition",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client
    ),
    instructions="""
            你是一个智能实体识别器，负责识别输入的文本中的实体。最终输出为json。
        """,
    output_type=EntityList
)

agent_main = Agent(
    name="MainCoordinator",
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client
    ),
    instructions="""
            你是一个智能管家，负责理解用户需求并决定如何处理请求。你可以判断输入文本的情感，也可以对文本中的实体进行识别。
        """,
    handoffs=[agent_sentiment_classification, agent_entity_recognition]
)


async def main():
    # text = input("Press Enter to start...")

    try:
        result = Runner.run_streamed(agent_main, "我今天很开心，我取得了期末考试的第一名")
        # result = Runner.run_streamed(agent_main, "中国最长的河流是长江")

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
    except Exception as e:
        print(f"\n请求失败: {type(e).__name__}")
        print(f"错误信息: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
