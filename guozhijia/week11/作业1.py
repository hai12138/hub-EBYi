import os

# 配置API
os.environ["OPENAI_API_KEY"] = "sk-f988e27a26444967885d84f2326d4f83d"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL = "qwen3.5-flash"

# 子Agent 1: 情感分类
sentiment_agent = Agent(
    name="sentiment_agent",
    model=MODEL,
    instructions="""你是一个文本情感分类专家。你的任务是分析用户输入的文本情感倾向。
请将文本情感分为以下三类之一：
- 积极 (Positive): 文本表达喜悦、满意、赞美等正向情绪
- 消极 (Negative): 文本表达愤怒、失望、厌恶等负向情绪
- 中性 (Neutral): 文本没有明显的情感倾向

请给出分类结果并简要说明理由。"""
)

# 子Agent 2: 实体识别
ner_agent = Agent(
    name="ner_agent",
    model=MODEL,
    instructions="""你是一个命名实体识别专家。你的任务是识别用户输入文本中的命名实体。
请识别以下类型的实体：
- 人名 (PER)
- 地名 (LOC)
- 组织机构名 (ORG)
- 时间 (TIME)
- 数值 (NUM)

请以列表形式输出识别到的实体及其类型，如果没有识别到某类实体则不输出。"""
)

# 主Agent: 接受用户请求，路由到合适的子Agent
triage_agent = Agent(
    name="triage_agent",
    model=MODEL,
    instructions="""你是一个智能路由助手。根据用户的请求内容，将请求转发给最合适的子Agent处理：
- 如果用户想对文本进行情感分析/情感分类，请转发给 sentiment_agent
- 如果用户想对文本进行实体识别/命名实体提取，请转发给 ner_agent
- 如果用户同时需要两种分析，请转发给 sentiment_agent（它处理完后会回到你这里，再转发给 ner_agent）

请根据用户意图选择合适的Agent。""",
    handoffs=[sentiment_agent, ner_agent],
)


def run_tests():
    """运行测试用例"""

    # 测试1: 情感分类
    print("=" * 60)
    print("测试1: 情感分类")
    print("=" * 60)
    test_input_1 = "请对以下文本进行情感分类：这个产品真的太棒了，我非常喜欢！"
    print(f"用户输入: {test_input_1}")
    print("-" * 40)
    result = Runner.run_sync(triage_agent, test_input_1)
    print(f"Agent输出: {result.final_output}")
    print()

    # 测试2: 实体识别
    print("=" * 60)
    print("测试2: 实体识别")
    print("=" * 60)
    test_input_2 = "请对以下文本进行实体识别：马云于2019年9月10日卸任阿里巴巴集团董事局主席。"
    print(f"用户输入: {test_input_2}")
    print("-" * 40)
    result = Runner.run_sync(triage_agent, test_input_2)
    print(f"Agent输出: {result.final_output}")
    print()

    # 测试3: 综合测试 - 需要路由判断
    print("=" * 60)
    print("测试3: 综合测试（自动路由）")
    print("=" * 60)
    test_input_3 = "帮我分析一下这句话的情感：北京今天的天气真糟糕，让人心情很差。"
    print(f"用户输入: {test_input_3}")
    print("-" * 40)
    result = Runner.run_sync(triage_agent, test_input_3)
    print(f"Agent输出: {result.final_output}")


if __name__ == "__main__":
    run_tests()
