import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-aa76fcf6520f48d38b356ae436c16af0"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

async def run(mcp_server: MCPServer):
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    agent = Agent(
        name="Assistant",
        instructions="你是qwen，擅长回答各类问题。",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model="qwen-flash",
            openai_client=external_client,
        )
    )

    # 调用 get_today_daily_news 工具 -》 得到结果 -》 大模型message格式 汇总为输入-》 qwen-flash -》 结果
    message = "最近有什么新闻？"
    print(f"Running: {message}")

    """
    [
        {
            "role": "system",
            "content": "你是qwen，擅长回答各类问题。"
        },
        {
            "role": "user",
            "content": "最近有什么新闻？ get_today_daily_news 调用结果"
        },
    ]
    """

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


    # 调用 get_city_weather 工具 -》 得到结果 -》 大模型message格式 汇总为输入-》 qwen-flash -》 结果
    message = "武汉最近的天气怎么样？"
    print(f"Running: {message}")

    result = Runner.run_streamed(agent, input=message)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


async def main():
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
    )as server:
        await run(server)

if __name__ == "__main__":
    asyncio.run(main())
