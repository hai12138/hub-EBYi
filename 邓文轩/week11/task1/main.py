import asyncio
from manager import TaskManager



async def main():
    query = input("请输入: \n")
    mg = TaskManager()
    await mg.run(query)


if __name__ == "__main__":
    asyncio.run(main())