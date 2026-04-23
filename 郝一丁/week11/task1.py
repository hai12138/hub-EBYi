import asyncio
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled

# 使用 chat_completions 接口
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 子 agent 1：情感分类
classification_agent = Agent(
    name="Sentiment Classification Agent",
    instructions=(
        "你是一个情感分类助手。"
        "用户给出一句文本后，你需要判断其情感类别。"
        "只输出分类结果和一句简短说明。"
        "情感类别限定为：积极、消极、中性。"
    ),
    handoff_description="当用户要求做情感分类、判断文本情绪、分析文本是正面还是负面时，交给这个 agent。"
)

# 子 agent 2：实体识别
ner_agent = Agent(
    name="Named Entity Recognition Agent",
    instructions=(
        "你是一个命名实体识别助手。"
        "用户给出一句文本后，你需要识别其中的实体。"
        "重点识别：人名、地名、组织机构名、时间。"
        "请按清晰格式输出识别结果。"
    ),
    handoff_description="当用户要求做实体识别、提取人名地名机构名时间等信息时，交给这个 agent。"
)

# 主 agent：负责分发
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "你是一个任务分发助手。"
        "如果用户想做情感分类，就转交给情感分类 agent。"
        "如果用户想做实体识别，就转交给实体识别 agent。"
        "不要自己完成任务，直接选择最合适的 agent。"
    ),
    handoffs=[classification_agent, ner_agent]
)

async def main():
    msg = input("你好，我可以帮你做情感分类或实体识别，请输入任务和文本：\n")

    result = await Runner.run(
        triage_agent,
        input=msg
    )

    print("\n处理结果：")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
    