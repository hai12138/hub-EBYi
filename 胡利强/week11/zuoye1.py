import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatModel, RouterTool

# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 1. 子Agent 1：情感分类
sentiment_agent = Agent(
    name="情感分类助手",
    model=OpenAIChatModel("gpt-3.5-turbo"),
    instructions="""
    你只做文本情感分类：
    输出只能是：正面/负面/中性
    不要多余解释，直接给结果。
    """,
)

# 2. 子Agent 2：实体识别
ner_agent = Agent(
    name="实体识别助手",
    model=OpenAIChatModel("gpt-3.5-turbo"),
    instructions="""
    你只做实体识别：
    识别文本中的人名、地名、机构名、时间、物品等实体，
    用清晰列表输出，不要多余废话。
    """,
)

# 3. 主Agent + 路由工具（自动选子Agent）
router_tool = RouterTool(
    agents=[sentiment_agent, ner_agent],
    description="根据用户输入自动选择：情感分类 或 实体识别",
)

main_agent = Agent(
    name="主路由Agent",
    model=OpenAIChatModel("gpt-3.5-turbo"),
    instructions="你是总调度，根据用户需求自动调用情感分类或实体识别Agent",
    tools=[router_tool],
)

# 运行测试
if __name__ == "__main__":
    print("=== 第十一周作业1：主Agent路由子Agent ===")
    user_input = input("请输入文本：")
    result = Runner.run_sync(main_agent, user_input)
    print("\n最终结果：")
    print(result.final_output)
