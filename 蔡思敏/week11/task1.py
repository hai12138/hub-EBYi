"""
@Author  :  CAISIMIN
@Date    :  2026/4/12 19:59
"""

import os
import asyncio
from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled

os.environ["OPENAI_API_KEY"] = "sk-e365d480f719416e8f4e317b7fa03ca1"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

sentiment_classifer_agent = Agent(
    name="sentiment_classifer_agent",
    model="qwen-max",
    instructions="你是一个情感分类工具，请对输入的文本进行情感分类，给出对应的情感标签，如积极、消极、中立"
)

ner_classifer_agent = Agent(
    name="ner_classifer_agent",
    model="qwen-max",
    instructions="你是一个命名实体识别工具，请对输入的文本进行命名实体识别，给出对应的命名实体标签，如人名、地名、组织机构名、时间、数字等"
)


triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="你的任务是根据用户的提问内容，判断应该将请求分派给 'sentiment_classifer_agent' 还是 'ner_classifer_agent'。",
    handoffs=[sentiment_classifer_agent, ner_classifer_agent],
)

async def main():
    query = "我很开心!!"
    print(f"**用户提问：** {query}")
    result = await Runner.run(triage_agent, query)
    print(result.final_output)

    print("=" * 50)

    query = "北京是中国的首都"
    print(f"**用户提问：** {query}")
    result = await Runner.run(triage_agent, query)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())


# 运行结果

# **用户提问：** 我很开心!!
# 积极
# ==================================================
# **用户提问：** 北京是中国的首都
# 北京：地名
# 中国：地名
#
# Process finished with exit code 0
