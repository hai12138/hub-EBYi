import os

from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled


os.environ.setdefault("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

os.environ.setdefault("OPENAI_API_KEY", "sk-69dd3e7fd02e49d3a38498daedaab73d")

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = os.environ.get("AGENT_MODEL", "qwen-max")


sentiment_agent = Agent(
    name="sentiment_agent",
    model=MODEL_NAME,
    handoff_description="负责文本情感分类，判断积极、消极或中性。",
    instructions=(
        "你是一个文本情感分类专家。"
        "当用户要求你分析一句话、一段话或评论的情感时，"
        "请判断其情感标签为：积极、消极、中性 三者之一。"
        "请使用如下格式输出：\n"
        "情感标签：<积极/消极/中性>\n"
        "理由：<一句简短说明>"
    ),
)


ner_agent = Agent(
    name="ner_agent",
    model=MODEL_NAME,
    handoff_description="负责文本实体识别，抽取人名、地名、机构、时间等实体。",
    instructions=(
        "你是一个命名实体识别专家。"
        "当用户要求你从文本中识别人名、地名、机构、时间、产品、事件等实体时，"
        "请抽取实体，并按如下格式输出：\n"
        "实体识别结果：\n"
        "1. <实体> - <类型>\n"
        "2. <实体> - <类型>\n"
        "如果没有明显实体，请输出：实体识别结果：未识别到明显实体。"
    ),
)


triage_agent = Agent(
    name="triage_agent",
    model=MODEL_NAME,
    instructions=(
        "你是主路由 Agent，负责根据用户意图把请求分派给最合适的子 Agent。"
        "如果用户要做情感分类、情绪判断、正负面分析、评论倾向分析，"
        "就交给 sentiment_agent。"
        "如果用户要做实体识别、命名实体抽取、人名地名机构时间抽取，"
        "就交给 ner_agent。"
        "你不要自己完成最终任务，必须把请求 handoff 给一个子 Agent。"
    ),
    handoffs=[sentiment_agent, ner_agent],
)


def main() -> None:
    print("=" * 60)
    print("Handoffs 路由示例：主 Agent -> 情感分类 Agent / 实体识别 Agent")
    print(f"当前模型：{MODEL_NAME}")
    print("示例1：请判断这句话的情感：这家餐厅环境很好，下次还会再来。")
    print("示例2：请识别这段文本中的实体：马云于1999年在杭州创立了阿里巴巴。")
    print("输入 quit / exit 结束。")
    print("=" * 60)

    while True:
        user_input = input("\n请输入你的任务：").strip()
        if not user_input:
            print("输入不能为空，请重新输入。")
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("程序结束。")
            break

        try:
            result = Runner.run_sync(triage_agent, user_input)
            print(f"\n[路由结果] 当前由：{result.last_agent.name}")
            print(result.final_output)
        except Exception as exc:
            print(f"\n运行失败：{exc}")


if __name__ == "__main__":
    main()
