import asyncio
from fastmcp import Client

client = Client("http://localhost:8900/sse")

async def call_tool():
    # async with client:
    #     result = await client.call_tool("get_today_daily_news")
    #     print(result)

    async with client:
        result = await client.list_tools() # 列举mcp server 中所有的tool
        print(result)

        result = await client.call_tool("get_today_daily_news") # 通过client调用 mcp server中某一个tool
        print(result)

asyncio.run(call_tool())
